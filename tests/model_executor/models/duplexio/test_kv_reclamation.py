# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.nn.attention.flex_attention import BlockMask
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOKVCacheManager,
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    _decode_uint32,
    _encode_uint32,
    duplexio_compact_key_visible,
    duplexio_primary_compact_slots,
    duplexio_primary_write_slots,
    update_duplexio_attention_metadata,
    update_duplexio_graph_block_mask,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    duplexio_attention_visible,
)


def test_compact_attention_matches_logical_rows_across_audio_eviction() -> None:
    torch.manual_seed(11)
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=2,
        max_model_len=12 * DUPLEXIO_NUM_CELLS,
    )
    dim = 7
    epoch = 19
    cached_keys = torch.zeros(layout.max_compact_slots, dim)
    cached_values = torch.zeros_like(cached_keys)
    cached_epochs = torch.zeros(layout.max_compact_slots, dtype=torch.long)
    cached_positions = torch.zeros(layout.max_compact_slots, dtype=torch.long)
    cached_ordinals = torch.zeros(layout.max_compact_slots, dtype=torch.long)
    cached_active = torch.zeros(layout.max_compact_slots, dtype=torch.bool)
    logical_keys: list[torch.Tensor] = []
    logical_values: list[torch.Tensor] = []
    logical_active: list[bool] = []
    active_text_tokens = 0

    for frame in range(8):
        positions = torch.arange(
            frame * DUPLEXIO_NUM_CELLS,
            (frame + 1) * DUPLEXIO_NUM_CELLS,
        )
        frame_keys = torch.randn(DUPLEXIO_NUM_CELLS, dim)
        frame_values = torch.randn(DUPLEXIO_NUM_CELLS, dim)
        text_active = torch.tensor(
            [
                frame % 2 == 0,
                frame % 3 != 0,
                frame % 4 == 1,
                frame in (2, 5, 7),
            ]
        )
        key_active = torch.cat((text_active, torch.ones(2, dtype=torch.bool)))
        frame_ordinals = torch.zeros(DUPLEXIO_NUM_CELLS, dtype=torch.long)
        active_ranks = torch.cumsum(text_active.to(torch.long), dim=0)
        frame_ordinals[:DUPLEXIO_NUM_TEXT_CELLS] = torch.where(
            text_active,
            active_text_tokens + active_ranks,
            0,
        )
        active_text_tokens += int(text_active.sum())

        primary_slots = duplexio_primary_compact_slots(positions, layout)
        cached_keys[primary_slots] = frame_keys
        cached_values[primary_slots] = frame_values
        cached_epochs[primary_slots] = epoch
        cached_positions[primary_slots] = positions
        cached_ordinals[primary_slots] = frame_ordinals
        cached_active[primary_slots] = key_active
        for cell, ordinal in enumerate(frame_ordinals.tolist()):
            if ordinal == 0:
                continue
            slot = layout.persistent_text_slot(ordinal - 1)
            cached_keys[slot] = frame_keys[cell]
            cached_values[slot] = frame_values[cell]
            cached_epochs[slot] = epoch
            cached_positions[slot] = positions[cell]
            cached_ordinals[slot] = ordinal
            cached_active[slot] = True

        logical_keys.extend(frame_keys)
        logical_values.extend(frame_values)
        logical_active.extend(key_active.tolist())
        all_keys = torch.stack(logical_keys)
        all_values = torch.stack(logical_values)
        all_positions = torch.arange(all_keys.shape[0])
        live_slots = layout.live_compact_slots(active_text_tokens)
        compact_positions = torch.arange(live_slots)

        queries = torch.randn(DUPLEXIO_NUM_CELLS, dim)
        for query, query_position in zip(queries, positions, strict=True):
            logical_visible = duplexio_attention_visible(
                query_position,
                all_positions,
                torch.tensor(logical_active),
                audio_attention_window_frames=layout.audio_window_frames,
            )
            logical_scores = query @ all_keys[logical_visible].T
            logical_output = (
                logical_scores.softmax(dim=-1)
                @ all_values[logical_visible]
            )

            compact_visible = duplexio_compact_key_visible(
                query_position,
                compact_positions,
                torch.tensor(epoch),
                cached_epochs[:live_slots],
                cached_positions[:live_slots],
                cached_ordinals[:live_slots],
                cached_active[:live_slots],
                layout,
            )
            compact_scores = query @ cached_keys[:live_slots][compact_visible].T
            compact_output = (
                compact_scores.softmax(dim=-1)
                @ cached_values[:live_slots][compact_visible]
            )
            torch.testing.assert_close(compact_output, logical_output)


def test_audio_ring_usage_stays_constant_for_sustained_session() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=8,
        max_model_len=600_000,
    )
    frames = torch.arange(100_000).repeat_interleave(2)
    cells = torch.tensor([4, 5]).repeat(100_000)
    positions = frames * DUPLEXIO_NUM_CELLS + cells
    slots = duplexio_primary_compact_slots(positions, layout)

    assert slots.unique().numel() == 2 * (layout.audio_window_frames + 1)
    assert int(slots.max()) < layout.audio_slots
    assert layout.live_compact_slots(0) == (
        layout.persistent_text_base + DUPLEXIO_NUM_TEXT_CELLS
    )


def test_bulk_prefill_keeps_only_final_transient_text_writes() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=8,
        max_model_len=60,
    )
    positions = torch.arange(3 * DUPLEXIO_NUM_CELLS)
    request_indices = torch.zeros_like(positions)

    slots = duplexio_primary_write_slots(
        positions,
        request_indices,
        layout,
    ).view(3, DUPLEXIO_NUM_CELLS)

    assert torch.equal(slots[:2, :DUPLEXIO_NUM_TEXT_CELLS], torch.full((2, 4), -1))
    assert torch.equal(
        slots[2, :DUPLEXIO_NUM_TEXT_CELLS],
        torch.arange(
            layout.transient_text_base,
            layout.transient_text_base + DUPLEXIO_NUM_TEXT_CELLS,
        ),
    )
    assert torch.equal(
        slots[:, DUPLEXIO_NUM_TEXT_CELLS:],
        duplexio_primary_compact_slots(positions, layout)
        .view(3, DUPLEXIO_NUM_CELLS)[:, DUPLEXIO_NUM_TEXT_CELLS:],
    )


def test_cache_epoch_excludes_reused_pages_from_canceled_session() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=2,
        max_model_len=60,
    )
    visible = duplexio_compact_key_visible(
        torch.tensor(4),
        torch.tensor(layout.audio_slot(0, 0)),
        torch.tensor(2),
        torch.tensor(1),
        torch.tensor(4),
        torch.tensor(0),
        torch.tensor(True),
        layout,
    )

    assert not bool(visible)


def test_compact_attention_keeps_inactive_audio_only_for_current_self() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=2,
        max_model_len=60,
    )
    audio_position = torch.tensor(4)
    audio_slot = torch.tensor(layout.audio_slot(0, 0))

    current_visible = duplexio_compact_key_visible(
        audio_position,
        audio_slot,
        torch.tensor(1),
        torch.tensor(1),
        audio_position,
        torch.tensor(0),
        torch.tensor(False),
        layout,
    )
    next_frame_visible = duplexio_compact_key_visible(
        torch.tensor(DUPLEXIO_NUM_CELLS),
        audio_slot,
        torch.tensor(1),
        torch.tensor(1),
        audio_position,
        torch.tensor(0),
        torch.tensor(False),
        layout,
    )

    assert bool(current_visible)
    assert not bool(next_frame_visible)


def test_cache_metadata_round_trips_full_uint32_domain() -> None:
    values = torch.tensor([0, 255, 65_537, 2**32 - 1], dtype=torch.long)

    encoded = _encode_uint32(values, torch.bfloat16)

    assert torch.equal(_decode_uint32(encoded), values)


def test_cache_metadata_does_not_change_attention_scores() -> None:
    torch.manual_seed(13)
    query = torch.randn(5, 7)
    key = torch.randn(5, 7)
    epochs = torch.tensor([1, 2, 3, 4, 5])
    positions = torch.tensor([0, 6, 12, 18, 24])
    ordinals = torch.tensor([0, 1, 2, 3, 4])
    query_metadata = torch.cat(
        (_encode_uint32(epochs, query.dtype), query.new_zeros(5, 16)),
        dim=-1,
    )
    key_metadata = torch.cat(
        (
            key.new_zeros(5, 4),
            _encode_uint32(epochs, key.dtype),
            _encode_uint32(positions, key.dtype),
            _encode_uint32(ordinals, key.dtype),
            _encode_uint32(torch.ones_like(epochs), key.dtype),
        ),
        dim=-1,
    )

    augmented_scores = torch.cat((query, query_metadata), dim=-1) @ torch.cat(
        (key, key_metadata),
        dim=-1,
    ).T

    torch.testing.assert_close(augmented_scores, query @ key.T)


@dataclass(frozen=True)
class _Block:
    block_id: int


class _BlockPool:
    def __init__(self) -> None:
        self.next_block = 0
        self.freed: list[object] = []

    def get_new_blocks(self, count: int) -> list[_Block]:
        blocks = [
            _Block(block_id)
            for block_id in range(self.next_block, self.next_block + count)
        ]
        self.next_block += count
        return blocks

    def free_blocks(self, blocks: Iterable[object]) -> None:
        self.freed.extend(blocks)


def test_cache_manager_reserves_once_and_releases_every_block_on_teardown() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=8,
        max_model_len=600,
    )
    pool = _BlockPool()
    manager = cast(Any, object.__new__(DuplexIOKVCacheManager))
    manager.layout = layout
    manager.req_to_blocks = defaultdict(list)
    manager.num_cached_block = {}
    manager._partial_hit_reqs = {}
    manager.block_pool = pool
    manager.new_block_ids = []

    assert manager.get_num_blocks_to_allocate(
        "session",
        num_tokens=6,
        new_computed_blocks=[],
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=6,
    ) == layout.max_blocks
    allocated = manager.allocate_new_blocks("session", 6, 6)
    assert len(allocated) == layout.max_blocks
    assert manager.take_new_block_ids() == [
        block.block_id for block in allocated
    ]
    assert manager.take_new_block_ids() == []
    assert manager.get_num_blocks_to_allocate(
        "session",
        num_tokens=500,
        new_computed_blocks=[],
        total_computed_tokens=494,
        num_local_computed_tokens=494,
        num_tokens_main_model=500,
    ) == 0

    manager.free("session")

    assert "session" not in manager.req_to_blocks
    assert pool.freed == list(reversed(allocated))


def test_cache_spec_registers_native_manager_and_preserves_page_contract() -> None:
    base = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=80,
        head_size_v=80,
        dtype=torch.bfloat16,
    )

    spec = make_duplexio_kv_cache_spec(
        base,
        audio_window_frames=8,
        max_model_len=600,
    )

    assert KVCacheSpecRegistry.get_manager_class(spec) is DuplexIOKVCacheManager
    assert spec.page_size_bytes == base.page_size_bytes
    assert spec.max_num_blocks_per_req(
        cast(Any, object()),
        1,
    ) == spec.layout.max_blocks
    assert DuplexIOKVCacheSpec.merge([spec, spec]) is spec


def test_runtime_metadata_scans_only_live_compact_pages() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=8,
        max_model_len=600,
    )
    metadata = cast(Any, object.__new__(FlexAttentionMetadata))
    metadata.duplexio_layout = layout
    metadata.seq_lens = torch.full((2,), layout.max_compact_slots)
    metadata.num_blocks_per_seq = torch.full((2,), layout.max_blocks)
    metadata.block_table = torch.arange(
        1,
        1 + 2 * layout.max_blocks,
        dtype=torch.int32,
    ).view(2, layout.max_blocks)
    metadata.block_size = layout.block_size
    metadata.physical_to_logical = torch.empty(
        2,
        1 + 2 * layout.max_blocks,
        dtype=torch.long,
    )
    metadata.block_mask = object()

    update_duplexio_attention_metadata(
        {"first": metadata, "shared": metadata},
        [5, 19],
    )

    expected_lengths = torch.tensor(
        [layout.live_compact_slots(5), layout.live_compact_slots(19)]
    )
    assert torch.equal(metadata.seq_lens, expected_lengths)
    assert torch.equal(
        metadata.num_blocks_per_seq,
        torch.tensor(
            [
                (length + layout.block_size - 1) // layout.block_size
                for length in expected_lengths.tolist()
            ]
        ),
    )
    assert metadata.block_mask is None
    first_unused_block = metadata.block_table[
        0,
        metadata.num_blocks_per_seq[0],
    ]
    assert metadata.physical_to_logical[0, first_unused_block] == -1


def test_graph_block_mask_updates_stable_physical_candidates() -> None:
    metadata = cast(Any, object.__new__(FlexAttentionMetadata))
    metadata.block_table = torch.tensor([[3, 4]], dtype=torch.int32)
    metadata.block_size = 16
    metadata.kv_block_size = 8
    metadata.duplexio_graph_block_offsets = torch.arange(3, dtype=torch.int32)
    block_mask = BlockMask(
        seq_lengths=(6, 128),
        kv_num_blocks=torch.zeros((1, 1, 1), dtype=torch.int32),
        kv_indices=torch.full((1, 1, 1, 7), -1, dtype=torch.int32),
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=None,
        q_indices=None,
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(16, 8),
        mask_mod=lambda _b, _h, _q, _kv: torch.tensor(True),
    )
    metadata.duplexio_graph_block_mask = block_mask
    indices_pointer = block_mask.kv_indices.data_ptr()

    update_duplexio_graph_block_mask(metadata, 2)

    assert block_mask.kv_indices.data_ptr() == indices_pointer
    assert block_mask.kv_num_blocks.item() == 4
    assert torch.equal(
        block_mask.kv_indices[0, 0, 0],
        torch.tensor([6, 7, 8, 9, -1, -1, -1], dtype=torch.int32),
    )
