# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803  (Triton constexpr arguments keep their kernel names)
"""Attention of flat positions over DuplexIO's paged cache.

Every position reads three logical ranges of its request's cache layout
(``kv_reclamation.DuplexIOKVLayout``): its audio window in the audio ring, the
BOS frames' audio keys in the window, and the persistent keys. A step's real
keys are written before attention, and each range ends at what had been
written up to the position, so within a step attention is causal over real
positions. A query or no-op writes nothing and adds its own key instead.

The positions of one frame share a program, so a key tile is loaded once for
all of them; their ranges differ only at the end. Programs split the frame's
keys, and a second kernel merges the splits and the self key.
"""

from __future__ import annotations

import torch
from torch import Tensor
from vllm.triton_utils import tl, triton

from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_POSITIONS

# Columns of the per-position attention metadata.
REQUEST = 0  # The request's row in the step's block table.
AUDIO_START, AUDIO_END, BOS_START, BOS_END, PERSISTENT_START, PERSISTENT_END = range(1, 7)
SELF_KEY = 7  # 1 when the position attends to its own unwritten key.
ATTENTION_COLUMNS = 8
LOG2E = 1.4426950408889634


@triton.jit
def flat_attention_partial_kernel(
    q,
    key_cache,
    value_cache,
    tables,
    meta,
    partial,
    partial_stats,
    scale_log2,
    ring_slots,
    bos_base,
    persistent_base,
    block_size,
    stride_q_token,
    stride_q_head,
    stride_k_block,
    stride_k_slot,
    stride_k_head,
    stride_v_block,
    stride_v_slot,
    stride_v_head,
    stride_table,
    stride_meta,
    G: tl.constexpr,
    D: tl.constexpr,
    FRAME: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
):
    frame = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    kv_heads = tl.num_programs(1)
    rows = tl.arange(0, ROWS)
    position = rows // G
    valid = position < FRAME
    token = frame * FRAME + tl.minimum(position, FRAME - 1)
    dims = tl.arange(0, D)

    row_meta = meta + token * stride_meta
    audio_start = tl.load(row_meta + 1)
    audio_end = tl.load(row_meta + 2)
    bos_start = tl.load(row_meta + 3)
    bos_end = tl.load(row_meta + 4)
    persistent_start = tl.load(row_meta + 5)
    persistent_end = tl.load(row_meta + 6)
    request = tl.load(meta + frame * FRAME * stride_meta)

    big = 2**30
    # Every position's range in a region lies within the frame's union.
    union_audio = tl.min(tl.where(valid, audio_start, big), 0)
    audio_length = tl.maximum(tl.max(tl.where(valid, audio_end, 0), 0) - union_audio, 0)
    union_bos = tl.min(tl.where(valid, bos_start, big), 0)
    bos_length = tl.maximum(tl.max(tl.where(valid, bos_end, 0), 0) - union_bos, 0)
    union_persistent = tl.min(tl.where(valid, persistent_start, big), 0)
    persistent_length = tl.maximum(tl.max(tl.where(valid, persistent_end, 0), 0) - union_persistent, 0)
    total = audio_length + bos_length + persistent_length
    per_split = tl.cdiv(total, SPLITS)
    begin = split * per_split
    stop = tl.minimum(begin + per_split, total)

    head = kv_head * G + rows % G
    query = tl.load(q + token[:, None] * stride_q_token + head[:, None] * stride_q_head + dims[None, :])
    maximum = tl.full((ROWS,), -1.0e30, tl.float32)
    normalizer = tl.zeros((ROWS,), tl.float32)
    accumulator = tl.zeros((ROWS, D), tl.float32)
    table = tables + request.to(tl.int64) * stride_table
    for tile in tl.range(begin, stop, BLOCK_N):
        index = tile + tl.arange(0, BLOCK_N)
        present = index < stop
        in_audio = index < audio_length
        in_bos = (index >= audio_length) & (index < audio_length + bos_length)
        audio_logical = union_audio + index
        bos_logical = union_bos + index - audio_length
        persistent_logical = union_persistent + index - audio_length - bos_length
        logical = tl.where(in_audio, audio_logical, tl.where(in_bos, bos_logical, persistent_logical))
        slot = tl.where(
            in_audio,
            audio_logical % ring_slots,
            tl.where(in_bos, bos_base + bos_logical % ring_slots, persistent_base + persistent_logical),
        )
        page = tl.load(table + slot // block_size, mask=present, other=0).to(tl.int64)
        offset = (slot % block_size).to(tl.int64)
        keys = tl.load(
            key_cache
            + page[:, None] * stride_k_block
            + offset[:, None] * stride_k_slot
            + kv_head * stride_k_head
            + dims[None, :],
            mask=present[:, None],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(keys)) * scale_log2
        start = tl.where(
            in_audio[None, :],
            audio_start[:, None],
            tl.where(in_bos[None, :], bos_start[:, None], persistent_start[:, None]),
        )
        end = tl.where(
            in_audio[None, :], audio_end[:, None], tl.where(in_bos[None, :], bos_end[:, None], persistent_end[:, None])
        )
        visible = present[None, :] & valid[:, None] & (logical[None, :] >= start) & (logical[None, :] < end)
        scores = tl.where(visible, scores, float("-inf"))
        updated = tl.maximum(maximum, tl.max(scores, 1))
        weights = tl.exp2(scores - updated[:, None])
        rescale = tl.exp2(maximum - updated)
        values = tl.load(
            value_cache
            + page[:, None] * stride_v_block
            + offset[:, None] * stride_v_slot
            + kv_head * stride_v_head
            + dims[None, :],
            mask=present[:, None],
            other=0.0,
        )
        normalizer = normalizer * rescale + tl.sum(weights, 1)
        accumulator = accumulator * rescale[:, None] + tl.dot(weights.to(values.dtype), values)
        maximum = updated
    program = (frame * kv_heads + kv_head) * SPLITS + split
    tl.store(partial + (program * ROWS + rows[:, None]) * D + dims[None, :], accumulator)
    tl.store(partial_stats + (program * ROWS + rows) * 2, maximum)
    tl.store(partial_stats + (program * ROWS + rows) * 2 + 1, normalizer)


@triton.jit
def flat_attention_merge_kernel(
    q,
    self_key,
    self_value,
    meta,
    partial,
    partial_stats,
    output,
    scale_log2,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_o_token,
    stride_o_head,
    stride_meta,
    G: tl.constexpr,
    D: tl.constexpr,
    FRAME: tl.constexpr,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    kv_heads = tl.num_programs(1) // G
    frame, position = token // FRAME, token % FRAME
    kv_head = head // G
    row = position * G + head % G
    dims = tl.arange(0, D)
    splits = tl.arange(0, SPLITS)
    program = (frame * kv_heads + kv_head) * SPLITS + splits
    maximum = tl.load(partial_stats + (program * ROWS + row) * 2)
    normalizer = tl.load(partial_stats + (program * ROWS + row) * 2 + 1)
    accumulator = tl.load(partial + (program[:, None] * ROWS + row) * D + dims[None, :])
    own = tl.load(meta + token * stride_meta + 7) != 0
    query = tl.load(q + token * stride_q_token + head * stride_q_head + dims).to(tl.float32)
    key = tl.load(self_key + token * stride_k_token + kv_head * stride_k_head + dims).to(tl.float32)
    value = tl.load(self_value + token * stride_v_token + kv_head * stride_v_head + dims).to(tl.float32)
    own_score = tl.where(own, tl.sum(query * key) * scale_log2, -1.0e30)
    top = tl.maximum(tl.max(maximum, 0), own_score)
    weights = tl.exp2(maximum - top)
    own_weight = tl.where(own, tl.exp2(own_score - top), 0.0)
    numerator = tl.sum(accumulator * weights[:, None], 0) + own_weight * value
    denominator = tl.sum(normalizer * weights, 0) + own_weight
    result = numerator / tl.maximum(denominator, 1.0e-30)
    tl.store(output + token * stride_o_token + head * stride_o_head + dims, result.to(output.dtype.element_ty))


def attention_splits(frames: int, kv_heads: int) -> int:
    """Key splits per frame, a power of two: enough programs to fill the GPU; one for long prefills."""
    return min(32, triton.next_power_of_2(triton.cdiv(256, frames * kv_heads)))


def flat_paged_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    tables: Tensor,
    meta: Tensor,
    *,
    scale: float,
    ring_slots: int,
    bos_base: int,
    persistent_base: int,
    block_size: int,
) -> Tensor:
    """Attend ``(tokens, heads, dim)`` queries of whole frames to the paged cache and their own keys.

    ``key``/``value`` are the tokens' own ``(tokens, kv_heads, dim)``; the
    caches are ``(blocks, block_size, kv_heads, dim)`` views; ``tables`` the
    step's block table; ``meta`` ``(tokens, ATTENTION_COLUMNS)`` int32 rows (any row stride).
    """
    tokens, heads, dim = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    frames = tokens // FRAME_POSITIONS
    rows = triton.next_power_of_2(max(16, FRAME_POSITIONS * groups))
    splits = attention_splits(frames, kv_heads)
    programs = frames * kv_heads * splits
    partial = torch.empty((programs, rows, dim), dtype=torch.float32, device=query.device)
    stats = torch.empty((programs, rows, 2), dtype=torch.float32, device=query.device)
    scale_log2 = scale * LOG2E
    flat_attention_partial_kernel[(frames, kv_heads, splits)](
        query,
        key_cache,
        value_cache,
        tables,
        meta,
        partial,
        stats,
        scale_log2,
        ring_slots,
        bos_base,
        persistent_base,
        block_size,
        query.stride(0),
        query.stride(1),
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        tables.stride(0),
        meta.stride(0),
        G=groups,
        D=dim,
        FRAME=FRAME_POSITIONS,
        ROWS=rows,
        BLOCK_N=32,
        SPLITS=splits,
        num_warps=4,
        num_stages=2,
    )
    output = torch.empty_like(query)
    flat_attention_merge_kernel[(tokens, heads)](
        query,
        key,
        value,
        meta,
        partial,
        stats,
        output,
        scale_log2,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        output.stride(0),
        output.stride(1),
        meta.stride(0),
        G=groups,
        D=dim,
        FRAME=FRAME_POSITIONS,
        ROWS=rows,
        SPLITS=splits,
        num_warps=4,
    )
    return output


__all__ = ["ATTENTION_COLUMNS", "REQUEST", "SELF_KEY", "flat_paged_attention"]
