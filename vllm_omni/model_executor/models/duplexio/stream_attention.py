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
    enable_gqa: bool,
    kernel_options: dict[str, int | bool],
) -> tuple[Tensor, Tensor]:
    """Attend to prior rows straight out of paged storage.

    Query is (tokens, heads, dim) and K/V are (slots, kv_heads, dim) spanning the
    whole cache pool; the block mask resolves which pages and slots each query
    owns. Returns the history in query layout plus its natural-log softmax
    normalizer (tokens, heads), which the caller needs to merge the query-local
    diagonal that the cache does not hold yet.
    """
    history, auxiliary = flex_attention(
        query.transpose(0, 1)[None],
        keys.transpose(0, 1)[None],
        values.transpose(0, 1)[None],
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_aux=AuxRequest(lse=True),
        kernel_options=kernel_options,
    )
    assert auxiliary.lse is not None
    return history[0].transpose(0, 1), auxiliary.lse[0].transpose(0, 1)


@torch.compile(dynamic=True, fullgraph=True)
def merge_self_attention(
    query: Tensor,
    self_key: Tensor,
    self_value: Tensor,
    history: Tensor,
    lse: Tensor,
    scale: float,
) -> Tensor:
    """Add each query's own cell to its cached history.

    Query/history are (tokens, heads, dim), self K/V (tokens, kv_heads, dim) and
    ``lse`` (tokens, heads) is the natural-log normalizer of ``history``. Grouped
    query heads reduce against their own KV head without expanding it. A query
    with no visible history has ``lse == -inf`` and returns its own cell exactly.
    """
    kv_heads = self_key.shape[1]
    groups = query.shape[1] // kv_heads
    grouped = query.unflatten(1, (kv_heads, groups)).float()
    scores = (grouped * self_key.unsqueeze(2).float()).sum(-1) * scale
    ratio = (scores - lse.unflatten(1, (kv_heads, groups)).float()).unsqueeze(-1)
    merged = (
        history.unflatten(1, (kv_heads, groups)).float() * torch.sigmoid(-ratio)
        + self_value.unsqueeze(2).float() * torch.sigmoid(ratio)
    )
    return merged.flatten(1, 2).to(query.dtype)
