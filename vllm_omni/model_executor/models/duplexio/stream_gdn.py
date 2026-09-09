# SPDX-License-Identifier: Apache-2.0
"""Packed GDN prefill and recurrent six-cell decode."""

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
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
    initial = initial_gdn_state(cache, slots, has_state)
    q, k, v, g, beta = (tensor.unsqueeze(0) for tensor in (q, k, v, g, beta))
    if q.shape[1] == slots.shape[0] * 6:
        output, final = fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=initial,
            output_final_state=True, cu_seqlens=boundaries,
        )
    else:
        output, final = chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=initial,
            output_final_state=True, cu_seqlens=boundaries, chunk_indices=chunks,
        )
    # Cache views alias vLLM's mixed-dtype allocation; mutate outside compilation.
    cache[slots] = final
    return output[0]
