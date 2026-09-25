# SPDX-License-Identifier: Apache-2.0
"""Packed GDN prefill and recurrent six-cell decode."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from torch import Tensor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

from vllm_omni.model_executor.models.duplexio.numerics import call_compiled_function


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
    """Use vLLM's convolution and recurrent-cache layout."""
    return MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_size, key_heads, value_heads, key_dim, value_dim,
        (conv_kernel_size - 1) * 6 + 1, 0,
    )


@torch.compile(dynamic=True, fullgraph=True)
def normalize_gdn_qk(x: Tensor) -> Tensor:
    value = x.float()
    return (value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


@torch.compile(dynamic=True, fullgraph=True)
def gdn_gates(b: Tensor, a: Tensor, a_log: Tensor, dt_bias: Tensor) -> tuple[Tensor, Tensor]:
    beta = b.sigmoid()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
    return beta, g


def prepare_gdn_inputs(
    qkv: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    key_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Match training's Q/K normalization and FP32 decay gate arithmetic."""
    key_width = key_heads * key_dim
    q, k, v = qkv.split((key_width, key_width, qkv.shape[1] - key_width * 2), -1)
    q = normalize_gdn_qk(q.view(qkv.shape[0], key_heads, key_dim))
    k = normalize_gdn_qk(k.view(qkv.shape[0], key_heads, key_dim))
    v = v.view(qkv.shape[0], -1, value_dim)
    beta, g = call_compiled_function(gdn_gates, b, a, a_log, dt_bias)
    return q, k, v, g, beta


@torch.compile(dynamic=True, fullgraph=True)
def initial_gdn_state(cache: Tensor, slots: Tensor, has_state: Tensor) -> Tensor:
    """Ignore unowned cache bytes when a request starts or reuses a slot."""
    return torch.where(has_state[:, None, None, None], cache[slots], 0)


@triton.jit
def slot_recurrent_gdn_kernel(
    q, k, v, g, beta, o, state, slots, has_state, cu_seqlens, scale,
    stride_q, stride_k, stride_v, stride_slot,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """fla's fused recurrent gated delta rule, reading and writing each request's cache slot in place.

    The arithmetic is fla's (head-wise beta and decay, key-major state), so
    results match it bit for bit; only the state addressing differs.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_g = g + bos * HV + i_hv
    p_beta = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_h = state + tl.load(slots + i_n).to(tl.int64) * stride_slot + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h & (tl.load(has_state + i_n) != 0), other=0).to(tl.float32)
    for _ in tl.range(0, eos - bos):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q * scale
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= tl.exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_g += HV
        p_beta += HV
        p_o += HV * V
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def slot_recurrent_gdn(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    cache: Tensor, slots: Tensor, boundaries: Tensor, has_state: Tensor,
) -> Tensor:
    """Advance each request's recurrent state in its cache slot; return the outputs."""
    tokens, heads, key_dim = k.shape
    value_heads, value_dim = v.shape[1:]
    assert q.stride()[1:] == k.stride()[1:] == (key_dim, 1) and v.stride()[1:] == (value_dim, 1)
    assert g.is_contiguous() and beta.is_contiguous() and g.shape == beta.shape == (tokens, value_heads)
    assert cache.shape[1:] == (value_heads, key_dim, value_dim) and cache.stride()[1:] == (
        key_dim * value_dim, value_dim, 1,
    )
    output = torch.empty((tokens, value_heads, value_dim), dtype=v.dtype, device=v.device)
    block_v = min(8, triton.next_power_of_2(value_dim))
    grid = (triton.cdiv(value_dim, block_v), slots.shape[0] * value_heads)
    slot_recurrent_gdn_kernel[grid](
        q, k, v, g, beta, output, cache, slots, has_state, boundaries, key_dim ** -0.5,
        q.stride(0), k.stride(0), v.stride(0), cache.stride(0),
        H=heads, HV=value_heads, K=key_dim, V=value_dim, BK=triton.next_power_of_2(key_dim), BV=block_v,
        num_warps=1, num_stages=3,
    )
    return output


def append_gdn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    cache: Tensor,
    slots: Tensor,
    boundaries: Tensor,
    has_state: Tensor,
    chunks: Tensor,
) -> Tensor:
    """Append packed whole-frame requests, resetting newly allocated slots.

    Each request contains at least six cells. Six cells per request therefore
    identifies a decode-only batch without reading sequence lengths on the CPU.
    Q/K are normalized at preparation; cached recurrence always remains FP32.
    """
    if q.shape[0] == slots.shape[0] * 6:
        return slot_recurrent_gdn(q, k, v, g, beta, cache, slots, boundaries, has_state)
    initial = initial_gdn_state(cache, slots, has_state)
    # The autograd wrapper ignores precomputed chunks and rebuilds them with a host sync.
    # Its forward assumes the contiguous layout the wrapper enforces; V is a strided split of QKV.
    q, k, v, g, beta = (tensor.unsqueeze(0).contiguous() for tensor in (q, k, v, g, beta))
    _, output, _, final, _, _ = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, k.shape[-1] ** -0.5, initial, True,
        cu_seqlens=boundaries, chunk_indices=chunks,
    )
    # Cache views alias vLLM's mixed-dtype allocation; mutate outside compilation.
    cache[slots] = final
    return output[0]
