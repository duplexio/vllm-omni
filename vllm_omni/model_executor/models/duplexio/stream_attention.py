# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training-ordered attention reductions for the native paged cache."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


@torch.compile(dynamic=True, fullgraph=True)
def cached_rotary_pos_emb(
    query: Tensor, key: Tensor, positions: Tensor, cos_sin_cache: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply training's fused rotation to (T, H, D) Q/K using cached phases."""
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    query, key = apply_rotary_pos_emb(
        query.transpose(0, 1)[None],
        key.transpose(0, 1)[None],
        torch.cat((cos, cos), dim=-1)[None],
        torch.cat((sin, sin), dim=-1)[None],
    )
    # Row-major again, and contiguous: the cache-write kernel indexes each
    # token's heads as one packed run.
    return (
        query[0].transpose(0, 1).contiguous(),
        key[0].transpose(0, 1).contiguous(),
    )


@torch.compile(dynamic=True, fullgraph=True)
def gated_attention_output(output: Tensor, gate: Tensor) -> Tensor:
    """Use the same fused sigmoid/product rounding in training and decoding."""
    return output * gate.sigmoid()


@torch.compile(fullgraph=True)
def paged_history_attention(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    block_mask: BlockMask,
    scale: float,
    kernel_options: dict[str, int | bool],
) -> tuple[Tensor, Tensor]:
    """Attend to prior rows straight out of paged storage.

    Query is (tokens, heads, dim) and K/V are (slots, kv_heads, dim) spanning the
    whole cache pool. Each KV head's grouped query heads become consecutive
    rows of one query block, so a KV tile is loaded once for all of them. The
    block mask's batch dimension splits each block's tiles; the query is
    repeated over it. Returns the split histories (splits, tokens, kv_heads,
    groups, dim) and their natural-log softmax normalizers (splits, tokens,
    kv_heads, groups), which the caller merges with the query-local diagonal
    the cache does not hold yet.
    """
    tokens, _, dim = query.shape
    kv_heads = keys.shape[1]
    splits = block_mask.kv_num_blocks.shape[0]
    # Inductor cannot lower a flex query broadcast over a symbolic batch, so each
    # split gets its own copy; it fuses into the packing copy.
    rows = query.unflatten(1, (kv_heads, -1)).transpose(0, 1).reshape(1, kv_heads, -1, dim)
    history, auxiliary = flex_attention(
        rows.expand(splits, -1, -1, -1).contiguous(),
        keys.transpose(0, 1)[None],
        values.transpose(0, 1)[None],
        block_mask=block_mask,
        scale=scale,
        return_aux=AuxRequest(lse=True),
        kernel_options=kernel_options,
    )
    assert auxiliary.lse is not None
    return (
        history.unflatten(2, (tokens, -1)).transpose(1, 2),
        auxiliary.lse.unflatten(2, (tokens, -1)).transpose(1, 2),
    )


@torch.compile(dynamic=True, fullgraph=True)
def merge_self_attention(
    query: Tensor,
    self_key: Tensor,
    self_value: Tensor,
    history: Tensor,
    lse: Tensor,
    scale: float,
) -> Tensor:
    """Add each query's own cell to its split cached history.

    Query is (tokens, heads, dim), self K/V (tokens, kv_heads, dim), and
    ``history``/``lse`` are (splits, tokens, kv_heads, groups[, dim]) partial
    reductions with their natural-log normalizers. Grouped query heads reduce
    against their own KV head without expanding it, and the partials merge in
    FP32. A split with no visible key has ``lse == -inf`` and weight zero, so a
    query with no history returns its own cell exactly.
    """
    kv_heads = self_key.shape[1]
    groups = query.shape[1] // kv_heads
    grouped = query.unflatten(1, (kv_heads, groups)).float()
    scores = (grouped * self_key.unsqueeze(2).float()).sum(-1) * scale
    lse = lse.float()
    top = torch.maximum(lse.amax(0), scores)
    weights = torch.exp(lse - top)
    own = torch.exp(scores - top)
    merged = (
        (history.float() * weights.unsqueeze(-1)).sum(0)
        + self_value.unsqueeze(2).float() * own.unsqueeze(-1)
    ) / (weights.sum(0) + own).unsqueeze(-1)
    return merged.flatten(1, 2).to(query.dtype)
