"""Compare NVFP4 routed-input sharing and split compute through serving bindings.

Synthetic GLM-5.3 TP4 experts have K4096, I512, E288, and top-8 routing.
Every arm uses identical source tensors, caller-owned scratch, and fresh bindings.
The per-expert control disables only the preparation-time equality proof; its
scale vectors are unchanged. CUDA-event samples alternate arm order and include
an optional L2 flush outside the timed region. Results are kernel diagnostics,
not model-serving throughput.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl as impl
from b12x.moe._shared.kernels.reference import compare_to_reference
from tests._reference.helpers import prepare_tp_moe_fp4_experts
from tests.moe.test_cute_migration_moe_standard_corpus import (
    _make_inputs,
    _make_nvfp4_weights,
    _nvfp4_oracle,
)


def hardware() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,pstate,clocks.current.graphics,clocks.current.memory,"
            "power.draw,power.limit,temperature.gpu,clocks_event_reasons.active",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def compare(actual, expected):
    actual = actual.float()
    expected = expected.float()
    metrics = compare_to_reference(actual, expected)
    rms = expected.square().mean().sqrt().item()
    return {
        "finite": bool(actual.isfinite().all().item()),
        "nonzero": bool(actual.abs().max().item() > 0),
        "cosine": metrics.cos,
        "normalized_rmse": metrics.rmse / rms,
        "maximum_absolute_error": (actual - expected).abs().max().item(),
        "bit_exact": torch.equal(actual, expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=192)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve independent measurement records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    geometry = {
        "num_experts": args.experts,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
    }
    report = {
        "status": "research-only",
        "complete": False,
        "conditions": vars(args)
        | {
            "output": str(args.output),
            "reference_file": str(args.reference_file),
        },
        "source": str(Path(impl.__file__).resolve()),
        "torch": torch.__version__,
        "hardware_before": hardware(),
        "arms": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    weights = _make_nvfp4_weights(device, seed=353, **geometry)
    # Non-unit but exactly uniform vectors exercise the serving representation
    # independently of the model's calibrated activation range.
    weights = replace(
        weights,
        a1_scale=torch.full((args.experts,), 1.25, device=device),
        a2_scale=torch.full((args.experts,), 0.75, device=device),
    )
    inputs = _make_inputs(
        device,
        m=args.tokens,
        seed=354,
        route_shift=0,
        num_experts=args.experts,
        hidden_size=args.hidden_size,
        topk=args.topk,
    )
    print("Computing independent NVFP4 oracle", flush=True)
    reference_identity = {
        **geometry,
        "tokens": args.tokens,
        "topk": args.topk,
        "weight_seed": 353,
        "input_seed": 354,
        "a1": 1.25,
        "a2": 0.75,
        "oracle_sha256": hashlib.sha256(
            (
                Path(impl.__file__).parents[1] / "_shared/kernels/reference.py"
            ).read_bytes()
        ).hexdigest(),
    }
    if args.reference_file is not None and args.reference_file.exists():
        cached = torch.load(args.reference_file, map_location="cpu", weights_only=True)
        assert cached["identity"] == reference_identity
        reference = cached["reference"].to(device)
    else:
        reference = _nvfp4_oracle(weights, inputs, **geometry)
        if args.reference_file is not None:
            torch.save(
                {"identity": reference_identity, "reference": reference.cpu()},
                args.reference_file,
            )
    print("Oracle complete", flush=True)
    graphs = []
    retained = []
    baseline = None
    arm_specs = (
        ("per-expert-monolithic", False, False),
        ("shared-input-monolithic", True, False),
        ("shared-input-split", True, True),
    )
    for name, shared, split in arm_specs[:1] if args.baseline_only else arm_specs:
        os.environ["B12X_NVFP4_DYNAMIC_MATERIALIZED"] = str(int(split))
        experts = prepare_tp_moe_fp4_experts(
            a=inputs.a,
            a1_gscale=weights.a1_scale,
            w1_fp4=weights.w1_fp4,
            w1_blockscale=weights.w1_scale,
            w1_alphas=weights.w1_alpha,
            a2_gscale=weights.a2_scale,
            w2_fp4=weights.w2_fp4,
            w2_blockscale=weights.w2_scale,
            w2_alphas=weights.w2_alpha,
            quant_mode="nvfp4",
            source_format="modelopt_nvfp4",
        )
        if hasattr(experts, "immutable_input_scales"):
            experts = replace(experts, immutable_input_scales=True)
            assert experts.can_share_input(input_scales_static=True)
        else:
            assert args.baseline_only
        assert experts.a1_gscale.data_ptr() == weights.a1_scale.data_ptr()
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=args.tokens,
                num_topk=args.topk,
                device=device,
                weight_plan=experts.plan,
                quant_mode="nvfp4",
                core_token_counts=(args.tokens,),
                frozen=True,
            )
        )
        config = plan.launch_plan.policy_resolution.config
        assert config.backend == "dynamic" and config.dynamic_tile_m == 128, config
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        output = torch.empty_like(inputs.a)

        def execute():
            if shared:
                assert experts.can_share_input(input_scales_static=True), (
                    "sharing invalidated before bind",
                    experts._a1_scale_version,
                    experts.a1_gscale._version,
                )
            binding = fused_moe.bind(
                plan,
                scratch=scratch,
                a=inputs.a,
                experts=experts,
                topk_ids=inputs.topk_ids,
                topk_weights=inputs.topk_weights,
                output=output,
                input_scales_static=True,
                fast_math=True,
            )
            # Preserve the zero-copy direct-scale contract in all three arms.
            assert binding.input_gs.data_ptr() == experts.a1_gscale.data_ptr()
            assert binding.down_input_scale.data_ptr() == experts.a2_gscale.data_ptr()
            if shared:
                assert experts.can_share_input(input_scales_static=True), (
                    "sharing invalidated by bind",
                    experts._a1_scale_version,
                    experts.a1_gscale._version,
                )
            fused_moe.run(binding=binding)
            if shared:
                assert experts.can_share_input(input_scales_static=True), (
                    "sharing invalidated by run",
                    experts._a1_scale_version,
                    experts.a1_gscale._version,
                )

        row = {
            "name": name,
            "shared_input": shared,
            "split": split,
            "policy": str(config),
            "samples_ms": [],
        }
        report["arms"].append(row)
        save()
        print(f"Warmup {name}", flush=True)
        choice = (
            nullcontext()
            if shared or not hasattr(experts, "can_share_input")
            else patch.object(
                impl.B12XFP4ExpertWeights, "can_share_input", return_value=False
            )
        )
        gate = getattr(impl, "_nvfp4_dynamic_materialized_enabled", None)
        seen = []
        compile_kernel = impl.b12x_compile

        def observed_compile(target, *compile_args, **compile_kwargs):
            kernel = getattr(target, "_kernel", None)
            facts = {
                "shared": getattr(kernel, "share_input_across_experts", None),
                "materialized": getattr(kernel, "materialize_intermediate", None),
                "external_fc1": getattr(kernel, "external_materialized_fc1", None),
                "target_type": type(target).__name__,
                "spec": str(compile_kwargs.get("compile_spec")),
            }
            row.setdefault("compile_facts", []).append(facts)
            save()
            print(json.dumps(facts), flush=True)
            return compile_kernel(target, *compile_args, **compile_kwargs)

        def observed_gate(**kwargs):
            result = gate(**kwargs)
            seen.append(result)
            return result

        started = time.monotonic()
        gate_observer = (
            nullcontext()
            if gate is None
            else patch.object(
                impl, "_nvfp4_dynamic_materialized_enabled", observed_gate
            )
        )
        with (
            choice,
            gate_observer,
            patch.object(impl, "b12x_compile", observed_compile),
        ):
            for _ in range(3):
                execute()
            torch.cuda.synchronize()
            row["compile_warmup_seconds"] = time.monotonic() - started
            assert (gate is None and not split) or (seen and any(seen) == split), (
                name,
                seen,
            )
            row["oracle"] = compare(output, reference)
            row["strict_cosine_0_9999_pass"] = row["oracle"]["cosine"] >= 0.9999
            # These are the existing standard-MoE GPU oracle limits from
            # test_standard_moe_dynamic_prefill_live_graph_oracle. The tighter
            # PR353 small-shape criterion is recorded separately: the serving
            # control at K4096/I512 already falls slightly below 0.9999.
            row["oracle_gate"] = {"min_cosine": 0.999, "max_normalized_rmse": 0.03}
            save()
            assert row["oracle"]["finite"] and row["oracle"]["nonzero"], row
            assert row["oracle"]["cosine"] >= 0.999, row
            assert row["oracle"]["normalized_rmse"] <= 0.03, row
            if baseline is None:
                baseline = output.clone()
            row["against_per_expert"] = compare(output, baseline)
            row["per_expert_strict_parity_pass"] = (
                row["against_per_expert"]["cosine"] >= 0.9999
                and row["against_per_expert"]["normalized_rmse"] <= 0.015
            )
            save()
            # Record exact-reference and cross-kernel parity independently.
            # A strict cross-kernel failure does not prevent diagnostic timing;
            # it remains an explicit, unresolved qualification result.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
        before = torch.cuda.memory_allocated()
        for _ in range(5):
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            metrics = compare(output, reference)
            assert metrics["finite"] and metrics["nonzero"]
            assert metrics["cosine"] >= 0.999 and metrics["normalized_rmse"] <= 0.03
        row["replay_checks"] = 5
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            graph.replay()
            torch.cuda.synchronize()
        row["cuda_kernel_names"] = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        save()
        assert (
            any("Nvfp4MaterializedPhase1" in n for n in row["cuda_kernel_names"])
            == split
        ), row
        assert (
            any("Nvfp4MaterializedPhase2" in n for n in row["cuda_kernel_names"])
            == split
        ), row
        graphs.append(graph)
        retained.append((experts, plan, scratch, output))
        save()
        print(json.dumps(row), flush=True)

    flush = torch.empty(args.flush_mib * 1024 * 1024, dtype=torch.uint8, device=device)
    for graph in graphs:
        for _ in range(20):
            graph.replay()
    torch.cuda.synchronize()
    for repetition in range(args.repeats):
        order = (
            range(len(graphs)) if repetition % 2 == 0 else reversed(range(len(graphs)))
        )
        for index in order:
            timings = []
            for _ in range(args.iterations):
                if args.flush_mib:
                    flush.fill_(repetition % 256)
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                timings.append(begin.elapsed_time(end))
            report["arms"][index]["samples_ms"].append(timings)
            report["arms"][index].setdefault("hardware_samples", []).append(hardware())
        save()
    for row in report["arms"]:
        row["repeat_medians_ms"] = [statistics.median(x) for x in row["samples_ms"]]
        row["median_ms"] = statistics.median(row["repeat_medians_ms"])
    reference_ms = report["arms"][0]["median_ms"]
    for row in report["arms"]:
        row["latency_change_percent"] = 100 * (row["median_ms"] / reference_ms - 1)
        row["throughput_change_percent"] = 100 * (reference_ms / row["median_ms"] - 1)
    report["hardware_after"] = hardware()
    report["complete"] = True
    save()
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"samples_ms", "cuda_kernel_names", "hardware_samples"}
                }
                for row in report["arms"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
