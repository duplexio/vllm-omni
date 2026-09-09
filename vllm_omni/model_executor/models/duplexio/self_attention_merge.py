"""Layout-invariant self/history attention arithmetic and its gradient."""

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def merge_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    history_ptr,
    lse_ptr,
    output_ptr,
    weights_ptr,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    length,
    width: tl.constexpr,
    qb,
    qh,
    qt,
    qd,
    kb,
    kh,
    kt,
    kd,
    vb,
    vh,
    vt,
    vd,
    hb,
    hh,
    ht,
    hd,
    lb,
    lh,
    lt,
    scale: tl.constexpr,
    block: tl.constexpr,
):
    position, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = head // (heads // kv_heads)
    row = (batch * heads + head) * length + position
    channel = tl.arange(0, block)
    query = tl.load(
        query_ptr + batch * qb + head * qh + position * qt + channel * qd, channel < width, 0
    ).to(tl.float32)
    key = tl.load(
        key_ptr + batch * kb + kv_head * kh + position * kt + channel * kd, channel < width, 0
    ).to(tl.float32)
    value = tl.load(
        value_ptr + batch * vb + kv_head * vh + position * vt + channel * vd, channel < width, 0
    ).to(tl.float32)
    history = tl.load(
        history_ptr + batch * hb + head * hh + position * ht + channel * hd, channel < width, 0
    ).to(tl.float32)
    lse = tl.load(lse_ptr + batch * lb + head * lh + position * lt)
    ratio = tl.sum(query * key, 0) * scale - lse
    own_weight, history_weight = tl.sigmoid(ratio), tl.sigmoid(-ratio)
    tl.store(
        output_ptr + row * width + channel, history * history_weight + value * own_weight, channel < width
    )
    tl.store(weights_ptr + row * 2, history_weight)
    tl.store(weights_ptr + row * 2 + 1, own_weight)


class SelfAttentionMerge(torch.autograd.Function):
    """One CTA per query fixes both reduction and pointwise rounding."""

    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        history: Tensor,
        lse: Tensor,
        scale: float,
    ) -> Tensor:
        batch, heads, length, width = query.shape
        output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
        weights = torch.empty((batch, heads, length, 2), device=query.device, dtype=torch.float32)
        launch: Any = merge_kernel[(length, heads, batch)]
        launch(
            query,
            key,
            value,
            history,
            lse,
            output,
            weights,
            heads,
            key.shape[1],
            length,
            width,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *history.stride(),
            *lse.stride(),
            scale,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(query, key, value, history, weights)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: Any, gradient: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None]:
        query, key, value, history, weights = ctx.saved_tensors
        batch, heads, length, width = query.shape
        kv_heads = key.shape[1]
        shape = (batch, kv_heads, heads // kv_heads, length, width)
        history_weight, own_weight = weights.unbind(-1)
        history_weight = history_weight.view(*shape[:-1], 1)
        own_weight = own_weight.view(*shape[:-1], 1)
        grad = gradient.view(shape).float()
        history_values = history.view(shape).float()
        own_values = value.float().unsqueeze(2)
        ratio_gradient = (
            grad
            * (
                own_values * own_weight * (1 - own_weight)
                - history_values * history_weight * (1 - history_weight)
            )
        ).sum(-1)
        scaled = ratio_gradient.unsqueeze(-1) * ctx.scale
        query_gradient = (scaled * key.float().unsqueeze(2)).flatten(1, 2)
        key_gradient = (scaled * query.view(shape).float()).sum(2)
        value_gradient = (grad * own_weight).sum(2)
        history_gradient = (grad * history_weight).flatten(1, 2)
        return (
            query_gradient.to(query.dtype),
            key_gradient.to(key.dtype),
            value_gradient.to(value.dtype),
            history_gradient.to(history.dtype),
            -ratio_gradient.flatten(1, 2),
            None,
        )


@torch.compile(dynamic=True, fullgraph=True)
def merge_self_attention(
    query: Tensor,
    self_key: Tensor,
    self_value: Tensor,
    history: Tensor,
    history_lse: Tensor,
    scale: float,
) -> Tensor:
    """Combine disjoint history/self softmaxes for (B, H, T, D) GQA tensors."""
    return SelfAttentionMerge.apply(query, self_key, self_value, history, history_lse, scale)
