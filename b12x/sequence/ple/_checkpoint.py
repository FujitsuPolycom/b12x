"""Prepared interior PLE checkpoint export; device offsets never select kernels."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared


@triton.jit
def _export_checkpoint_kernel(
    normalized_u_ptr,
    gathered_state_ptr,
    query_start_loc_ptr,
    checkpoint_offsets_ptr,
    checkpoint_slots_ptr,
    request_is_prefill_ptr,
    num_seqs_ptr,
    conv_state_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    MAX_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    element = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = request < tl.load(num_seqs_ptr)
    offset = tl.load(checkpoint_offsets_ptr + request, live, other=0)
    slot = tl.load(checkpoint_slots_ptr + request, live, other=-1).to(tl.int64)
    prefill = tl.load(request_is_prefill_ptr + request, live, other=False)
    start = tl.load(query_start_loc_ptr + request, live, other=0)
    end = tl.load(query_start_loc_ptr + request + 1, live, other=0)
    valid = (
        live
        & prefill
        & (slot >= 0)
        & (slot < MAX_SLOTS)
        & (start >= 0)
        & (end <= MAX_TOKENS)
        & (offset > 0)
        & (offset < end - start)
    )
    channel = element // STATE_CAPACITY
    position = element % STATE_CAPACITY
    relative = offset.to(tl.int64) - STATE_LENGTH + position.to(tl.int64)
    payload = (channel < CHANNELS) & (position < STATE_LENGTH)
    token = start.to(tl.int64) + relative
    query = tl.load(
        normalized_u_ptr + token * CHANNELS + channel.to(tl.int64),
        valid & payload & (relative >= 0),
        other=0,
    )
    history = tl.load(
        gathered_state_ptr
        + (request.to(tl.int64) * CHANNELS + channel.to(tl.int64)) * STATE_LENGTH
        + offset.to(tl.int64)
        + position.to(tl.int64),
        valid & payload & (relative < 0),
        other=0,
    )
    tl.store(
        conv_state_ptr + slot * STATE_STRIDE + element.to(tl.int64),
        tl.where(relative >= 0, query, history),
        valid & (element < CHANNELS * STATE_CAPACITY),
    )


@dataclass(frozen=True)
class _CompilePointer:
    dtype: torch.dtype
    alignment: int

    def data_ptr(self):
        return self.alignment


def compile_checkpoint(query, device):
    """Prepare the checkpoint program with scalar-aligned metadata pointers."""
    channels = query.streams * query.hidden_size
    length = query.dilation * (query.kernel_size - 1)
    with torch.cuda.device(device):
        bf16 = _CompilePointer(torch.bfloat16, 2)
        i32 = _CompilePointer(torch.int32, 4)
        program = _export_checkpoint_kernel.warmup(
            bf16,
            bf16,
            i32,
            i32,
            _CompilePointer(torch.int64, 8),
            _CompilePointer(torch.bool, 1),
            i32,
            bf16,
            CHANNELS=channels,
            STATE_LENGTH=length,
            STATE_CAPACITY=length + query.max_speculative_tokens,
            STATE_STRIDE=query.state_strides[0],
            MAX_TOKENS=query.max_tokens,
            MAX_SLOTS=query.max_state_slots,
            BLOCK=256,
            num_warps=4,
            grid=(
                query.max_seqs,
                triton.cdiv(channels * (length + query.max_speculative_tokens), 256),
            ),
        )
        return program


@torch.library.custom_op("b12x::ple_export_checkpoint", mutates_args=("conv_state",))
def _export_checkpoint_op(
    normalized_u: torch.Tensor,
    gathered_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    request_is_prefill: torch.Tensor,
    num_seqs: torch.Tensor,
    conv_state: torch.Tensor,
    plan_handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(plan_handle), "sequence.ple", conv_state.device
    )
    if not state.mixed or len(state.programs) != 6:
        raise ValueError("PLE checkpoint export requires a prepared mixed program")
    q = state.query
    if tuple(conv_state.stride()) != q.state_strides:
        raise ValueError("PLE checkpoint state strides differ from preparation")
    with torch.cuda.device(conv_state.device):
        state.programs[5][
            (q.max_seqs, triton.cdiv(state.channels * state.state_capacity, 256), 1)
        ](
            normalized_u,
            gathered_state,
            query_start_loc,
            offsets,
            slots,
            request_is_prefill,
            num_seqs,
            conv_state,
            state.channels,
            state.state_length,
            state.state_capacity,
            q.state_strides[0],
            q.max_tokens,
            q.max_state_slots,
            256,
        )


@_export_checkpoint_op.register_fake
def _export_checkpoint_fake(
    normalized_u: torch.Tensor,
    gathered_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    request_is_prefill: torch.Tensor,
    num_seqs: torch.Tensor,
    conv_state: torch.Tensor,
    plan_handle: int,
) -> None:
    pass


def export_layer_checkpoint(binding, *, offsets, slots):
    if binding.plan is None:
        raise ValueError("PLE checkpoint export requires a prepared binding")
    torch.ops.b12x.ple_export_checkpoint(
        binding.normalized_u,
        binding.gathered_state,
        binding.query_start_loc,
        offsets,
        slots,
        binding.request_is_prefill,
        binding.num_seqs,
        binding.conv_state,
        binding.plan.handle,
    )
