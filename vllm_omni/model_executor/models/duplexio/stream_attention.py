# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training-ordered attention reductions for the native paged cache."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from vllm_omni.model_executor.models.duplexio.row_semantics import DUPLEXIO_NUM_CELLS


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


@torch.compile(dynamic=True, fullgraph=True)
def merge_row_attention(
    query: Tensor,
    self_key: Tensor,
    self_value: Tensor,
    keys: Tensor,
    values: Tensor,
    extra_slots: Tensor,
    audio: Tensor,
    audio_lse: Tensor,
    persistent: Tensor,
    persistent_lse: Tensor,
    empty: Tensor,
    scale: float,
) -> Tensor:
    """Join a row's cached partials with the keys its cache read left out.

    Query is (tokens, heads, dim), self K/V (tokens, kv_heads, dim), and
    ``keys``/``values`` (slots, kv_heads, dim) the whole paged cache, read at
    each row's ``extra_slots`` (two per row, -1 where none). The audio and
    persistent partials are FlashAttention's packed rows, (rows, kv_heads *
    cells * groups, dim) with natural-log normalizers (kv_heads * cells *
    groups, rows), and ``empty`` (rows * 2) flags the rows whose audio or
    persistent read was empty, whose partials are dropped. Grouped query heads
    reduce against their own KV head without expanding it, and the partials
    merge in FP32. A partial with no key has weight zero, so a query with no
    history returns its own cell exactly.
    """
    tokens, heads, dim = query.shape
    kv_heads = self_key.shape[1]
    groups = heads // kv_heads
    rows = tokens // DUPLEXIO_NUM_CELLS
    grouped = query.view(rows, DUPLEXIO_NUM_CELLS, kv_heads, groups, dim).float()

    empty = empty.view(rows, 2)

    def partial(output: Tensor, lse: Tensor, empty: Tensor) -> tuple[Tensor, Tensor]:
        lse = lse.view(kv_heads, DUPLEXIO_NUM_CELLS, groups, rows).permute(3, 1, 0, 2)
        output = output.view(rows, kv_heads, DUPLEXIO_NUM_CELLS, groups, dim).transpose(1, 2)
        empty = empty[:, None, None, None]
        return (
            output.float().masked_fill(empty[..., None], 0),
            lse.masked_fill(empty, -torch.inf),
        )

    audio, audio_lse = partial(audio, audio_lse, empty[:, 0])
    persistent, persistent_lse = partial(persistent, persistent_lse, empty[:, 1])
    own_key = self_key.view(rows, DUPLEXIO_NUM_CELLS, kv_heads, 1, dim).float()
    own_value = self_value.view(rows, DUPLEXIO_NUM_CELLS, kv_heads, 1, dim).float()
    own = (grouped * own_key).sum(-1) * scale
    # (rows, 1, kv_heads, 1, extra, dim)
    slots = extra_slots.view(rows, -1)
    # A missing slot reads slot zero, which may hold anything: mask its score
    # and its value, since zero weight times a stale NaN is still NaN.
    missing = (slots < 0)[:, :, None, None]
    extra_key = keys[slots.clamp_min(0)].float()
    extra_value = values[slots.clamp_min(0)].float().masked_fill(missing, 0)
    extra_key, extra_value = (
        tensor.transpose(1, 2)[:, None, :, None] for tensor in (extra_key, extra_value)
    )
    extra = ((grouped.unsqueeze(-2) * extra_key).sum(-1) * scale).masked_fill(
        (slots < 0)[:, None, None, None], -torch.inf
    )
    top = torch.maximum(torch.maximum(audio_lse, persistent_lse), torch.maximum(own, extra.amax(-1)))
    audio_weight = torch.exp(audio_lse - top)
    persistent_weight = torch.exp(persistent_lse - top)
    own_weight = torch.exp(own - top)
    extra_weight = torch.exp(extra - top.unsqueeze(-1))
    merged = (
        audio * audio_weight.unsqueeze(-1)
        + persistent * persistent_weight.unsqueeze(-1)
        + own_value * own_weight.unsqueeze(-1)
        + (extra_value * extra_weight.unsqueeze(-1)).sum(-2)
    ) / (audio_weight + persistent_weight + own_weight + extra_weight.sum(-1)).unsqueeze(-1)
    return merged.reshape(tokens, heads, dim).to(query.dtype)
