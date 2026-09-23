"""QSA compiled program identity does not depend on the cache pool's page counts.

The number of main and compressed cache pages sizes the caller's pool and
changes between starts whenever the available memory differs. No QSA program
takes a page count as a compile-time constant or a launch argument, so two
declarations that differ only in those counts plan the same compiled programs
and a cached restart with a differently sized pool recompiles nothing.

Compile planning runs in a spawned offline compiler worker, the same process
shape the compile pool uses, so the host test needs no GPU and leaves the
test process's CUDA state untouched.
"""
from __future__ import annotations

import ast
import json
import multiprocessing
import os
import pathlib

import pytest

CORPUS = pathlib.Path(__file__).parent.parent / "preparation" / "corpora"
CORPUS_FILE = CORPUS / "qwen3_8_flash_next_tp2_rank0.jsonl"
DEVICE_UUID = "48d28d14-08f3-f3d1-cb99-20c3fa5eca41"
PRODUCT_NAME = "NVIDIA GeForce RTX 5090"

# Page counts on both sides of Triton's integer specialization boundaries: a
# multiple of 16 and its neighbour; the smallest legal pool of each query joins them.
PAGE_COUNTS = ((2, 2), (16, 16), (21536, 5384), (21537, 5385))


@pytest.mark.parametrize("starts,rows,suffix,status,expected", [
    ((0, 3), 2, 2, 1, False),
    ((3, 4), 2, 0, 1, False),
    ((-1, 1), 2, 0, 1, False),
    ((1, 0), 2, 0, 1, False),
    ((0, 0), 2, 0, 0, False),
    ((0, 2), 2, 0, 1, False),
    ((0, 2), 2, 0, 0, True),
    ((0, 1), 1, 0, 0, True),
    ((0, 0), 0, 0, 0, False),
    ((0, 3), 2, 2, 0, False),
    ((0, 16), 16, 7, 0, True),
    ((0, 16), 16, 8, 0, False),
])
def test_raw_ring_commit_checks_boundaries_before_row_loads(
    starts, rows, suffix, status, expected,
):
    """Execute the production load guards with bounds-checked host pointers.

    This catches an invalid read even when CUDA's allocator would happen to
    leave accessible memory beyond a tensor. GPU tests cover the full launch.
    """
    from types import SimpleNamespace

    path = pathlib.Path(__file__).parents[2] / "b12x/attention/qsa/_kernels.py"
    kernel = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_commit_raw_ring_kernel")
    prefix = []
    for statement in kernel.body:
        prefix.append(statement)
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "real_request"
            for target in statement.targets
        ):
            break

    class HostScalarCasts(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                return node.func.value
            return node

    class Pointer:
        def __init__(self, values, offset=0):
            self.values, self.offset = values, offset

        def __add__(self, offset):
            return Pointer(self.values, self.offset + offset)

    def load(pointer, mask=True, other=None):
        if not mask:
            return other
        assert 0 <= pointer.offset < len(pointer.values), "out-of-bounds row load"
        if pointer.values is request_values:
            assert status == 0, "errored request was dereferenced"
        return pointer.values[pointer.offset]

    request_values = [0] * rows
    env = dict(
        tl=SimpleNamespace(program_id=lambda axis: 0 if axis == 0 else suffix,
                           load=load, minimum=min),
        query_start_loc=Pointer(starts), state_errors=Pointer([status] * rows),
        request_ids=Pointer(request_values), rows=rows, RING_CAPACITY=8,
    )
    code = HostScalarCasts().visit(ast.Module(body=prefix, type_ignores=[]))
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), env)
    assert bool(env["real_request"]) is expected


def _smallest_legal_page_counts(payload: dict[str, object]) -> tuple[int, int]:
    max_seq_len = int(payload["max_seq_len"])
    groups = -(-max_seq_len // int(payload["compress_ratio"]))
    return (
        -(-max_seq_len // int(payload["main_page_size"])),
        -(-groups // int(payload["compressed_page_size"])),
    )


def _qsa_corpus_queries() -> list[dict[str, object]]:
    queries, seen = [], set()
    for line in CORPUS_FILE.read_text().splitlines():
        record = json.loads(line)
        if record["component"] != "attention.qsa" or record["request"] in seen:
            continue
        seen.add(record["request"])
        queries.append(record["query"])
    return queries


def _program_keys_by_geometry(payload: dict[str, object] | None, connection) -> None:
    """Report the planned program keys of each page-count variant of one query."""
    try:
        from b12x._lib import compile_pool
        from b12x._lib.compile_pool import CompileJob, describe_compilation
        from b12x.attention.qsa._tuning import TUNING, QsaQuery
        from b12x.preparation import DeviceIdentity

        activity = multiprocessing.get_context("spawn").Array("q", (0, 0))
        compile_pool._initialize_worker(
            0, (12, 0), DEVICE_UUID, PRODUCT_NAME, 170, 232448, 232448, activity,
        )
        identity = DeviceIdentity("nvidia", (12, 0), 170, PRODUCT_NAME)
        if payload is None:
            import torch

            from b12x.attention.qsa._contract import Caps, _query_from_caps
            from b12x.preparation import FrozenMapping

            caps = Caps(
                device=torch.device("cuda:0"),
                max_batch=4,
                max_raw_state_slots=4,
                max_q_rows=4096,
                max_seq_len=262144,
                num_main_cache_pages=1400,
                num_compressed_cache_pages=1400,
                main_page_size=1504,
                compressed_page_size=376,
                max_speculative_tokens=3,
                q_heads=12,
                kv_heads=1,
                index_heads=4,
                position_axes=3,
                mrope_interleaved=True,
                mrope_sections=(11, 11, 10),
            )
            payload = _query_from_caps(caps, FrozenMapping()).to_dict()
        keys = {}
        minimum_main, minimum_compressed = _smallest_legal_page_counts(payload)
        for main_pages, compressed_pages in (
            *PAGE_COUNTS, _smallest_legal_page_counts(payload),
        ):
            if main_pages < minimum_main or compressed_pages < minimum_compressed:
                continue
            query = QsaQuery(**{
                **payload,
                "num_main_cache_pages": main_pages,
                "num_compressed_cache_pages": compressed_pages,
            })
            config = TUNING.default_config(query, identity)
            job = CompileJob.create(
                "b12x.attention.qsa._contract:compile_qsa",
                query.to_dict(), TUNING.encode_config(config), 0,
            )
            plan = describe_compilation(job)
            keys[(main_pages, compressed_pages)] = sorted(
                (program.dialect, program.key, program.name) for program in plan.programs
            )
        connection.send(("ok", keys))
    except Exception as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _plan_in_offline_worker(payload: dict[str, object], cache_dir: pathlib.Path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "B12X_COMPILE_CACHE_DIR": str(cache_dir / "compile"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
    }
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        process = context.Process(target=_program_keys_by_geometry, args=(payload, child))
        process.start()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    child.close()
    try:
        status, result = parent.recv()
    finally:
        process.join(timeout=600)
    if status != "ok":
        raise AssertionError(f"offline QSA compile planning failed: {result}")
    return result


@pytest.mark.parametrize(
    "payload", _qsa_corpus_queries(),
    ids=lambda payload: f"m{payload['max_q_rows']}-context{payload['max_seq_len']}",
)
def test_qsa_program_keys_ignore_the_cache_page_counts(payload, tmp_path) -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")
    keys = _plan_in_offline_worker(payload, tmp_path)
    reference_geometry, reference = next(iter(keys.items()))
    assert reference, "QSA compile planning reported no programs"
    for geometry, programs in keys.items():
        assert programs == reference, (
            f"QSA programs for page counts {geometry} differ from {reference_geometry}: "
            f"{sorted(set(map(tuple, programs)) ^ set(map(tuple, reference)))}"
        )


def test_qsa_multi_chunk_programs_are_retained(tmp_path) -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")
    keys = _plan_in_offline_worker(None, tmp_path)
    reference = next(iter(keys.values()))
    for programs in keys.values():
        assert programs == reference
    assert sum(program[2] == "_stage_topk_carry_kernel" for program in reference) > 1
    names = {program[2] for program in reference}
    assert not names.intersection({
        "_remap_topk_group_ids_kernel", "_stable_topk_threshold_kernel",
        "_count_stable_topk_candidates_kernel", "_emit_stable_topk_kernel",
    })
    assert any(program[2] == "attention.qsa.stable_selection" for program in reference)


def test_compile_qsa_enumerates_shared_storage_validators() -> None:
    """Prepared support programs cover bindings whose caches share one pool.

    The compile transaction never overlaps compressed and raw storage, so the
    shared-pool page-table variant and the ownership validator must be compiled
    explicitly and keyed apart from the separate-storage variant; otherwise a
    shared-pool binding either misses its programs or runs the wrong variant.
    """
    from collections.abc import Mapping

    qsa_root = pathlib.Path(__file__).parents[2] / "b12x/attention/qsa"
    compiler = next(
        n for n in ast.parse((qsa_root / "_contract.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "compile_qsa"
    )
    block = next(
        n for n in ast.walk(compiler)
        if isinstance(n, ast.With)
        and ast.unparse(n.items[0].context_expr) == "_support_context(support, compiling=True)"
    )
    calls = {
        n.func.id: n for n in ast.walk(block)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "launch_validate_shared_pool_ownership" in calls
    keywords = {k.arg: k.value for k in calls["launch_validate_page_tables"].keywords}
    shared = keywords["shared_compressed_raw_pool"]
    assert isinstance(shared, ast.Constant) and shared.value is True

    key_function = next(
        n for n in ast.parse((qsa_root / "_kernels.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_support_kernel_key"
    )
    page_tables, score = object(), object()
    env = {
        "Mapping": Mapping,
        "_SUPPORT_KERNEL_KEYS": {page_tables: "_validate_page_tables_kernel"},
        "_score_representatives_kernel": score,
    }
    exec(
        compile(ast.Module(body=[key_function], type_ignores=[]), "_support_kernel_key", "exec"),
        env,
    )
    support_key = env["_support_kernel_key"]
    assert support_key(page_tables, {"SHARED_COMPRESSED_RAW_POOL": False}) == "_validate_page_tables_kernel"
    assert support_key(page_tables, {}) == "_validate_page_tables_kernel"
    assert support_key(page_tables, {"SHARED_COMPRESSED_RAW_POOL": True}) == "_validate_page_tables_kernel/shared"
