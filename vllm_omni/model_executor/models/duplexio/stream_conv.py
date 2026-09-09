# SPDX-License-Identifier: Apache-2.0
"""Packed six-cell convolution with FP32 accumulation and fused SiLU."""

import torch
from torch import Tensor
from vllm.triton_utils import tl, triton


@triton.jit
def stream_conv_kernel(
    x, weight, bias, state, slots, boundaries, has_state, chunks, output,
    channels: tl.constexpr, history: tl.constexpr,
    x_row: tl.constexpr, x_col: tl.constexpr, weight_row: tl.constexpr,
    state_slot: tl.constexpr, state_channel: tl.constexpr, state_time: tl.constexpr,
    has_bias: tl.constexpr, block_d: tl.constexpr,
):
    request = tl.load(chunks + tl.program_id(0) * 2)
    chunk = tl.load(chunks + tl.program_id(0) * 2 + 1)
    start, end = tl.load(boundaries + request), tl.load(boundaries + request + 1)
    slot, valid_state = tl.load(slots + request), tl.load(has_state + request)
    tokens = start + chunk * 64 + tl.arange(0, 64)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    valid = (tokens[:, None] < end) & (dims[None, :] < channels)
    acc = tl.zeros((64, block_d), tl.float32)
    for offset in tl.static_range(history // 6 + 1):
        source = tokens - offset * 6
        current = tl.load(x + source[:, None] * x_row + dims[None, :] * x_col,
                          mask=valid & (source[:, None] >= start), other=0).to(tl.float32)
        previous = tl.load(state + slot * state_slot + dims[None, :] * state_channel
                           + (source[:, None] - start + history) * state_time,
                           mask=valid & (source[:, None] < start) & valid_state, other=0).to(tl.float32)
        w = tl.load(weight + dims * weight_row + history - offset * 6,
                    mask=dims < channels, other=0).to(tl.float32)
        acc += (current + previous) * w[None, :]
    if has_bias:
        acc += tl.load(bias + dims, mask=dims < channels, other=0).to(tl.float32)[None, :]
    acc = acc * (1.0 / (1.0 + tl.exp(-acc)))
    tl.store(output + tokens[:, None] * channels + dims[None, :], acc, mask=valid)


@triton.jit
def update_stream_conv_state_kernel(
    x, state, slots, boundaries, has_state,
    channels: tl.constexpr, history: tl.constexpr, x_row: tl.constexpr, x_col: tl.constexpr,
    state_slot: tl.constexpr, state_channel: tl.constexpr, state_time: tl.constexpr,
    block_t: tl.constexpr, block_d: tl.constexpr,
):
    request = tl.program_id(0)
    start, end = tl.load(boundaries + request), tl.load(boundaries + request + 1)
    slot, valid_state = tl.load(slots + request), tl.load(has_state + request)
    times = tl.arange(0, block_t)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    source = end - history + times
    valid = (times[:, None] < history) & (dims[None, :] < channels)
    current = tl.load(x + source[:, None] * x_row + dims[None, :] * x_col,
                      mask=valid & (source[:, None] >= start), other=0)
    previous = tl.load(state + slot * state_slot + dims[None, :] * state_channel
                       + (source[:, None] - start + history) * state_time,
                       mask=valid & (source[:, None] < start) & valid_state, other=0)
    output = tl.where(source[:, None] >= start, current, previous)
    # History shifts in place: every warp must finish reading the old history
    # before another warp overwrites it. Each CTA owns distinct channels.
    tl.debug_barrier()
    tl.store(state + slot * state_slot + dims[None, :] * state_channel + times[:, None] * state_time,
             output, mask=valid)


def stream_causal_conv(
    x: Tensor, weight: Tensor, bias: Tensor | None, state: Tensor, slots: Tensor,
    boundaries: Tensor, has_state: Tensor, chunk_indices: Tensor,
) -> Tensor:
    """Use GDN's existing 64-token chunk index, then advance request-local state.

    ``x`` is packed (tokens, channels); ``weight`` has taps spaced six cells
    apart; ``state`` is (cache_slots, channels, history_cells). State advances
    in a separate launch so late chunks cannot overwrite an early chunk's reads.
    """
    channels, history = x.shape[1], state.shape[2]
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    stream_conv_kernel[(chunk_indices.shape[0], triton.cdiv(channels, 32))](
        x, weight, bias, state, slots, boundaries, has_state, chunk_indices, output,
        channels, history, *x.stride(), weight.stride(0), *state.stride(), bias is not None, 32,
    )
    update_stream_conv_state_kernel[(boundaries.shape[0] - 1, triton.cdiv(channels, 32))](
        x, state, slots, boundaries, has_state, channels, history, *x.stride(), *state.stride(),
        triton.next_power_of_2(history), 32,
    )
    return output
