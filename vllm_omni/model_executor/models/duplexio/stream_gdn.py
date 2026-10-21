# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803  (Triton constexpr arguments keep their kernel names)
"""Per-stream GatedDeltaNet over flat positions, reading and writing each request's cache slot.

A request's slot holds, for each of its six streams, the last three pre-conv
inputs (``(kernel - 1) * streams`` conv columns, stream-major) and the FP32
recurrent state (``streams * value_heads`` heads). Each position steps its own
stream from that stream's state: a real position commits the step, a query or
no-op computes it and keeps nothing. The conv and recurrence follow
``duplexio.decode.gdn_decode``: FP32 conv with SiLU rounded to the activation
dtype, then flash-linear-attention's fused recurrent rule with in-kernel Q/K L2
normalization (MIT; see GDN_LICENSE).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.triton_utils import tl, triton

from vllm_omni.model_executor.models.duplexio.frame_layout import NUM_GDN_STREAMS

# Columns of the per-position GDN metadata.
STREAM, REAL = 0, 1
GDN_COLUMNS = 2
# Columns of the per-request GDN metadata: its positions [start, end) in the
# pass, its row in the step's state indices, and whether its slot is unowned.
REQUEST_START, REQUEST_END, REQUEST_ROW, REQUEST_FRESH = range(4)
GDN_REQUEST_COLUMNS = 4


def gdn_cache_dtypes(dtype: torch.dtype) -> tuple[torch.dtype, ...]:
    """Convolution history and FP32 recurrent state."""
    return dtype, torch.float32


def gdn_cache_shapes(
    tp_size: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    conv_kernel_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Every stream's convolution history and recurrent heads, in one slot."""
    conv_shape, recurrent_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_size,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        (conv_kernel_size - 1) * NUM_GDN_STREAMS + 1,
        0,
    )
    return conv_shape, (recurrent_shape[0] * NUM_GDN_STREAMS, *recurrent_shape[1:])


def gdn_gates(a: Tensor, b: Tensor, a_log: Tensor, dt_bias: Tensor) -> tuple[Tensor, Tensor]:
    """FP32 decay ``g`` and the write strength ``beta`` in the activation dtype, as Qwen computes them."""
    return -a_log.float().exp() * F.softplus(a.float() + dt_bias), b.sigmoid()


@triton.jit
def flat_gdn_kernel(
    x,
    weight,
    g,
    beta,
    output,
    conv_state,
    recurrent_state,
    slots,
    positions,
    requests,
    scale,
    stride_x,
    stride_weight,
    stride_conv_slot,
    stride_conv_channel,
    stride_conv_time,
    stride_state_slot,
    stride_positions,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    STREAMS: tl.constexpr,
):
    """One (request, stream, value head, value tile): step the stream's positions in order."""
    i_v, i_segment = tl.program_id(0), tl.program_id(1)
    segment, i_hv = i_segment // HV, i_segment % HV
    request, stream = segment // STREAMS, segment % STREAMS
    i_h = i_hv // (HV // H)
    start = tl.load(requests + request * 4)
    end = tl.load(requests + request * 4 + 1)
    slot = tl.load(slots + tl.load(requests + request * 4 + 2)).to(tl.int64)
    # A padded request has no slot: it reads zeros and writes nothing.
    owned = slot >= 0
    fresh = (tl.load(requests + request * 4 + 3) != 0) | ~owned
    slot = tl.maximum(slot, 0)

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    q_channel = i_h * K + o_k
    k_channel = H * K + i_h * K + o_k
    v_channel = 2 * H * K + i_hv * V + o_v

    history = conv_state + slot * stride_conv_slot + stream * 3 * stride_conv_time
    keep = ~fresh
    q0 = tl.load(history + q_channel * stride_conv_channel, mask=keep, other=0).to(tl.float32)
    q1 = tl.load(history + q_channel * stride_conv_channel + stride_conv_time, mask=keep, other=0).to(tl.float32)
    q2 = tl.load(history + q_channel * stride_conv_channel + 2 * stride_conv_time, mask=keep, other=0).to(tl.float32)
    k0 = tl.load(history + k_channel * stride_conv_channel, mask=keep, other=0).to(tl.float32)
    k1 = tl.load(history + k_channel * stride_conv_channel + stride_conv_time, mask=keep, other=0).to(tl.float32)
    k2 = tl.load(history + k_channel * stride_conv_channel + 2 * stride_conv_time, mask=keep, other=0).to(tl.float32)
    v0 = tl.load(history + v_channel * stride_conv_channel, mask=keep, other=0).to(tl.float32)
    v1 = tl.load(history + v_channel * stride_conv_channel + stride_conv_time, mask=keep, other=0).to(tl.float32)
    v2 = tl.load(history + v_channel * stride_conv_channel + 2 * stride_conv_time, mask=keep, other=0).to(tl.float32)
    # Taps oldest first: output = w0 * x[t-3] + w1 * x[t-2] + w2 * x[t-1] + w3 * x[t].
    wq = weight + q_channel * stride_weight
    wk = weight + k_channel * stride_weight
    wv = weight + v_channel * stride_weight
    wq0 = tl.load(wq).to(tl.float32)
    wq1 = tl.load(wq + 1).to(tl.float32)
    wq2 = tl.load(wq + 2).to(tl.float32)
    wq3 = tl.load(wq + 3).to(tl.float32)
    wk0 = tl.load(wk).to(tl.float32)
    wk1 = tl.load(wk + 1).to(tl.float32)
    wk2 = tl.load(wk + 2).to(tl.float32)
    wk3 = tl.load(wk + 3).to(tl.float32)
    wv0 = tl.load(wv).to(tl.float32)
    wv1 = tl.load(wv + 1).to(tl.float32)
    wv2 = tl.load(wv + 2).to(tl.float32)
    wv3 = tl.load(wv + 3).to(tl.float32)

    p_h = recurrent_state + slot * stride_state_slot + ((stream * HV + i_hv) * K + o_k[:, None]) * V + o_v[None, :]
    b_h = tl.load(p_h, mask=keep, other=0).to(tl.float32)
    # A fresh slot writes every stream, so streams it does not step start from zero later.
    committed = fresh.to(tl.int32)
    for p in tl.range(start, end):
        meta = positions + p * stride_positions
        if tl.load(meta) == stream:
            row = x + p * stride_x
            xq = tl.load(row + q_channel).to(tl.float32)
            xk = tl.load(row + k_channel).to(tl.float32)
            xv = tl.load(row + v_channel).to(tl.float32)
            b_q = wq0 * q0 + wq1 * q1 + wq2 * q2 + wq3 * xq
            b_k = wk0 * k0 + wk1 * k1 + wk2 * k2 + wk3 * xk
            b_v = wv0 * v0 + wv1 * v1 + wv2 * v2 + wv3 * xv
            # SiLU, rounded to the activation dtype as the conv output is.
            b_q = (b_q * tl.sigmoid(b_q)).to(output.dtype.element_ty).to(tl.float32)
            b_k = (b_k * tl.sigmoid(b_k)).to(output.dtype.element_ty).to(tl.float32)
            b_v = (b_v * tl.sigmoid(b_v)).to(output.dtype.element_ty).to(tl.float32)
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_g = tl.load(g + p * HV + i_hv).to(tl.float32)
            b_beta = tl.load(beta + p * HV + i_hv).to(tl.float32)
            stepped = b_h * tl.exp(b_g)
            b_v = b_beta * (b_v - tl.sum(stepped * b_k[:, None], 0))
            stepped += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(stepped * b_q[:, None], 0)
            tl.store(output + (p * HV + i_hv) * V + o_v, b_o.to(output.dtype.element_ty))
            if tl.load(meta + 1) != 0:
                b_h = stepped
                q0, q1, q2 = q1, q2, xq
                k0, k1, k2 = k1, k2, xk
                v0, v1, v2 = v1, v2, xv
                committed = committed | 1
    tl.store(p_h, b_h, mask=owned & (committed != 0))


@triton.jit
def flat_conv_history_kernel(
    x,
    conv_state,
    slots,
    positions,
    requests,
    channels,
    stride_x,
    stride_conv_slot,
    stride_conv_channel,
    stride_conv_time,
    stride_positions,
    STREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Shift each stream's last three committed inputs into its history, after every read of it."""
    segment, block = tl.program_id(0), tl.program_id(1)
    request, stream = segment // STREAMS, segment % STREAMS
    start = tl.load(requests + request * 4)
    end = tl.load(requests + request * 4 + 1)
    slot = tl.load(slots + tl.load(requests + request * 4 + 2)).to(tl.int64)
    owned = slot >= 0
    fresh = (tl.load(requests + request * 4 + 3) != 0) | ~owned
    slot = tl.maximum(slot, 0)
    channel = block * BLOCK + tl.arange(0, BLOCK)
    inside = channel < channels
    history = conv_state + slot * stride_conv_slot + channel * stride_conv_channel + stream * 3 * stride_conv_time
    keep = inside & ~fresh
    h0 = tl.load(history, mask=keep, other=0)
    h1 = tl.load(history + stride_conv_time, mask=keep, other=0)
    h2 = tl.load(history + 2 * stride_conv_time, mask=keep, other=0)
    changed = fresh.to(tl.int32)
    for p in tl.range(start, end):
        meta = positions + p * stride_positions
        if (tl.load(meta) == stream) & (tl.load(meta + 1) != 0):
            h0, h1 = h1, h2
            h2 = tl.load(x + p * stride_x + channel, mask=inside, other=0)
            changed = changed | 1
    store = inside & owned & (changed != 0)
    tl.store(history, h0, mask=store)
    tl.store(history + stride_conv_time, h1, mask=store)
    tl.store(history + 2 * stride_conv_time, h2, mask=store)


def flat_gdn(
    mixed_qkv: Tensor,
    weight: Tensor,
    g: Tensor,
    beta: Tensor,
    conv_state: Tensor,
    recurrent_state: Tensor,
    slots: Tensor,
    positions: Tensor,
    requests: Tensor,
    num_requests: int,
    *,
    key_heads: int,
    key_dim: int,
    value_dim: int,
) -> Tensor:
    """Run each position of ``mixed_qkv`` ``(tokens, conv_dim)`` on its stream; returns its core outputs.

    The outputs are ``(tokens, value_heads, value_dim)``.

    ``weight`` is the ``(conv_dim, 4)`` causal kernel, ``conv_state`` a
    ``(slots, conv_dim, 3 * streams)`` view and ``recurrent_state``
    ``(slots, streams * value_heads, key_dim, value_dim)``; ``slots`` holds the
    step's state index per request, ``positions`` ``(tokens, GDN_COLUMNS)`` rows (any row stride) and
    ``requests`` ``(num_requests, GDN_REQUEST_COLUMNS)`` int32.
    """
    tokens, channels = mixed_qkv.shape
    value_heads = g.shape[1]
    assert weight.shape == (channels, 4) and weight.stride(1) == 1
    assert mixed_qkv.stride(1) == 1 and g.is_contiguous() and beta.is_contiguous()
    assert recurrent_state.shape[1:] == (NUM_GDN_STREAMS * value_heads, key_dim, value_dim)
    assert recurrent_state.stride()[1:] == (key_dim * value_dim, value_dim, 1)
    output = torch.zeros((tokens, value_heads, value_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    segments = num_requests * NUM_GDN_STREAMS
    block_v = min(32, triton.next_power_of_2(value_dim))
    flat_gdn_kernel[(triton.cdiv(value_dim, block_v), segments * value_heads)](
        mixed_qkv,
        weight,
        g,
        beta,
        output,
        conv_state,
        recurrent_state,
        slots,
        positions,
        requests,
        key_dim**-0.5,
        mixed_qkv.stride(0),
        weight.stride(0),
        *conv_state.stride(),
        recurrent_state.stride(0),
        positions.stride(0),
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BV=block_v,
        STREAMS=NUM_GDN_STREAMS,
        num_warps=2,
        num_stages=1,
    )
    block = 256
    flat_conv_history_kernel[(segments, triton.cdiv(channels, block))](
        mixed_qkv,
        conv_state,
        slots,
        positions,
        requests,
        channels,
        mixed_qkv.stride(0),
        *conv_state.stride(),
        positions.stride(0),
        STREAMS=NUM_GDN_STREAMS,
        BLOCK=block,
        num_warps=4,
    )
    return output


__all__ = ["GDN_COLUMNS", "GDN_REQUEST_COLUMNS", "flat_gdn", "gdn_cache_dtypes", "gdn_cache_shapes", "gdn_gates"]
