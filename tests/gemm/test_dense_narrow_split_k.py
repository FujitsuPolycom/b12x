"""Narrow dense FP8 split-K policy preserves capacity and ordered reductions."""

from dataclasses import asdict

import cutlass
import pytest

import b12x._lib.dense_gemm as dense


@pytest.mark.parametrize(
    "m,n,k,tile,sm,slices",
    [
        (1, 512, 4096, (16, 128), 48, 8),
        (4, 512, 7168, (16, 64), 48, 4),
        (8, 512, 4096, (16, 128), 48, 8),
        (4, 2560, 4608, (16, 64), 48, 1),
        (4, 2560, 4608, (16, 128), 48, 2),
        (4, 768, 4352, (16, 128), 48, 2),
        (9, 512, 4096, (16, 128), 48, 1),
        (4, 512, 3968, (16, 128), 48, 1),
        (4, 512, 4096, (16, 128), 1, 1),
        (4, 1536, 7168, (16, 128), 188, 8),
    ],
)
def test_narrow_split_uses_divisible_resident_slices(
    monkeypatch, m, n, k, tile, sm, slices
):
    monkeypatch.setattr(dense, "_B12X_DENSE_NARROW_SPLITK_MAX", 8, raising=False)
    monkeypatch.setattr(dense, "_B12X_DENSE_SPLITK_TURBO", True)
    policy = dense._dense_gemm_policy_for(
        m=m,
        n=n,
        k=k,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=tile,
        cluster_shape_mn=(1, 1),
        sm_count=sm,
        expected_m=m,
        generalize_mxfp8_split_k=True,
    )
    assert policy.split_k_slices == slices
    if slices > 1:
        assert not policy.split_k_atomic_bf16
        assert (k // 128) % slices == 0
        assert ((n + tile[1] - 1) // tile[1]) * slices <= sm


@pytest.mark.parametrize(
    "limit,expected", [(0, 1), (1, 1), (2, 2), (3, 2), (4, 4), (8, 8)]
)
@pytest.mark.parametrize("turbo", [False, True])
def test_narrow_split_limit_never_selects_atomic_math(
    monkeypatch, limit, expected, turbo
):
    monkeypatch.setattr(dense, "_B12X_DENSE_NARROW_SPLITK_MAX", limit, raising=False)
    monkeypatch.setattr(dense, "_B12X_DENSE_SPLITK_TURBO", turbo)
    policy = dense._dense_gemm_policy_for(
        m=4,
        n=512,
        k=4096,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(16, 128),
        cluster_shape_mn=(1, 1),
        sm_count=48,
        expected_m=4,
        generalize_mxfp8_split_k=True,
    )
    assert policy.split_k_slices == expected
    if expected > 1:
        assert policy.split_k_atomic_bf16 is False


@pytest.mark.parametrize("capacity", [8, 16, 128])
def test_narrow_policy_uses_planned_capacity_not_live_rows(monkeypatch, capacity):
    monkeypatch.setattr(dense, "_B12X_DENSE_NARROW_SPLITK_MAX", 8, raising=False)
    policies = [
        dense._dense_gemm_policy_for(
            m=live,
            n=512,
            k=4096,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=(16, 128),
            cluster_shape_mn=(1, 1),
            sm_count=48,
            expected_m=capacity,
            generalize_mxfp8_split_k=True,
        )
        for live in (1, 2, 4, 8)
    ]
    assert all(asdict(policy) == asdict(policies[0]) for policy in policies)
    assert policies[0].split_k_slices == (8 if capacity == 8 else 1)


@pytest.mark.parametrize(
    "m,n,k,dtype",
    [
        (4, 4096, 4096, cutlass.BFloat16),
        (8, 8192, 7168, cutlass.BFloat16),
        (4, 512, 4096, cutlass.Float16),
        (128, 2560, 4608, cutlass.BFloat16),
    ],
)
def test_other_geometries_keep_existing_policy(monkeypatch, m, n, k, dtype):
    args = dict(
        m=m,
        n=n,
        k=k,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=dtype,
        mma_tiler_mn=(16, 128),
        cluster_shape_mn=(1, 1),
        sm_count=48,
        expected_m=m,
        generalize_mxfp8_split_k=True,
    )
    monkeypatch.setattr(dense, "_B12X_DENSE_NARROW_SPLITK_MAX", 0, raising=False)
    prior = dense._dense_gemm_policy_for(**args)
    monkeypatch.setattr(dense, "_B12X_DENSE_NARROW_SPLITK_MAX", 8)
    assert dense._dense_gemm_policy_for(**args) == prior


@pytest.mark.parametrize("slices", [2, 4, 8])
@pytest.mark.parametrize("turbo", [False, True])
def test_prepared_narrow_config_roundtrips_ordered_policy(monkeypatch, slices, turbo):
    from b12x.gemm._preparation import _configured_lowering
    from b12x.gemm._tuning import DenseGemmQuery, TUNING
    from b12x.preparation import DeviceIdentity

    monkeypatch.setattr(dense, "_B12X_DENSE_NARROW_SPLITK_MAX", 8)
    monkeypatch.setattr(dense, "_B12X_DENSE_SPLITK_TURBO", turbo)
    query = DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=4,
        in_features=4096,
        out_features=512,
        expected_m=4,
        sm_count=48,
        output_mode="provided",
        overrides={
            "mma_tiler_mn": (16, 128),
            "load_path": "tma",
            "swap_ab": False,
            "_split_k_slices_override": slices,
        },
    )
    device = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")
    config = TUNING.configure(query, device=device).default
    recovered = TUNING.decode_config(TUNING.config_payload(config))
    lowering = _configured_lowering(query, recovered, device)
    assert recovered == config
    assert lowering.policy.split_k_slices == slices
    assert lowering.policy.split_k_atomic_bf16 is False
    assert lowering.kernel_c_l == slices


@pytest.mark.parametrize(
    "capacity,n,k", [(16, 512, 4096), (4, 2560, 4608), (4, 4096, 4096)]
)
def test_eight_slices_cannot_escape_tiny_resident_narrow_scope(
    monkeypatch, capacity, n, k
):
    from b12x.gemm._tuning import DenseGemmQuery, TUNING
    from b12x.preparation import DeviceIdentity

    monkeypatch.setattr(dense, "_B12X_DENSE_NARROW_SPLITK_MAX", 8)
    query = DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=capacity,
        in_features=k,
        out_features=n,
        expected_m=capacity,
        sm_count=48,
        overrides={
            "mma_tiler_mn": (16, 128),
            "load_path": "tma",
            "swap_ab": False,
            "_split_k_slices_override": 8,
        },
    )
    with pytest.raises(ValueError, match="eight-way|split-K|outside"):
        TUNING.configure(
            query, device=DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")
        )


@pytest.mark.parametrize("slices", [4, 8])
def test_owned_ordered_reducer_reuses_runtime_rows_without_resolution(
    monkeypatch, slices
):
    from types import SimpleNamespace

    calls = []
    reducer = lambda *args: calls.append(args)
    partials = SimpleNamespace(data_ptr=lambda: 16)
    output = SimpleNamespace(data_ptr=lambda: 32)
    monkeypatch.setattr(dense, "make_ptr", lambda dtype, pointer, *a, **kw: pointer)
    monkeypatch.setattr(dense, "current_cuda_stream", lambda: "current-stream")
    monkeypatch.setattr(
        dense,
        "_compile_dense_split_reduction",
        lambda *args: pytest.fail("runtime reducer resolution"),
    )
    for rows in (1, 4, 8):
        dense._run_dense_split_reduction(
            reducer, partials, output, rows, 512, slices, None
        )
    assert calls == [(16, 32, rows, "current-stream") for rows in (1, 4, 8)]
