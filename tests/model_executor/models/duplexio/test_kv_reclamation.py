# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOFrameMetadata,
    DuplexIOKVCacheManager,
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    duplexio_attention_visible,
)


def frame_metadata(layout: DuplexIOKVLayout, rows: int = 1) -> DuplexIOFrameMetadata:
    return DuplexIOFrameMetadata(layout, rows * DUPLEXIO_NUM_CELLS, torch.device("cpu"))


def install_row(
    frame: DuplexIOFrameMetadata,
    *,
    text_active: torch.Tensor,
    emitted_before: int,
    audio_frame: int,
    audio_active: bool = True,
) -> None:
    """Install one live row, mirroring what ``frame_inputs`` computes for it.

    ``audio_frame`` is the one-based audio-time position the row's own audio
    occupies (or, for a frozen row, the last position already recorded) and
    ``emitted_before`` how many text cells the session emitted before the row.
    """
    ordinals = torch.where(
        text_active,
        emitted_before + text_active.cumsum(0, dtype=torch.int32),
        0,
    )
    window = frame.layout.audio_window_frames
    frame.update(
        key_active=torch.cat((text_active, torch.full((2,), audio_active))),
        text_ordinal=torch.cat((ordinals, torch.zeros(2, dtype=torch.int32))),
        text_last=torch.full((DUPLEXIO_NUM_CELLS,), emitted_before, dtype=torch.int32),
        audio_first=torch.full(
            (DUPLEXIO_NUM_CELLS,), max(audio_frame - window, 1), dtype=torch.int32
        ),
        audio_last=torch.full(
            (DUPLEXIO_NUM_CELLS,),
            audio_frame - 1 if audio_active else audio_frame,
            dtype=torch.int32,
        ),
    )


def test_compact_cache_attention_matches_logical_rows_across_audio_eviction() -> None:
    torch.manual_seed(11)
    # Five ring frames for a two-frame window: the ring wraps every five rows,
    # so resident keys are both overwritten and outlived by the window.
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=2,
        max_model_len=5 * DUPLEXIO_NUM_CELLS,
    )
    frame = frame_metadata(layout)
    dim = 7
    cached_keys = torch.zeros(layout.max_compact_slots, dim)
    cached_values = torch.zeros_like(cached_keys)
    slots = torch.arange(layout.max_compact_slots)
    logical_keys: list[torch.Tensor] = []
    logical_values: list[torch.Tensor] = []
    logical_active: list[bool] = []
    logical_audio: list[int] = []
    emitted = 0

    for row in range(8):
        audio_frame = row + 1
        text_active = torch.tensor(
            [row % 2 == 0, row % 3 != 0, row % 4 == 1, row in (2, 5, 7)]
        )
        install_row(
            frame,
            text_active=text_active,
            emitted_before=emitted,
            audio_frame=audio_frame,
        )
        emitted += int(text_active.sum())

        keys = torch.randn(DUPLEXIO_NUM_CELLS, dim)
        values = torch.randn(DUPLEXIO_NUM_CELLS, dim)
        written = frame.write_slots(DUPLEXIO_NUM_CELLS)
        writing = written >= 0
        cached_keys[written[writing]] = keys[writing]
        cached_values[written[writing]] = values[writing]

        logical_keys.extend(keys)
        logical_values.extend(values)
        logical_active.extend(text_active.tolist() + [True, True])
        logical_audio.extend([audio_frame] * DUPLEXIO_NUM_CELLS)
        all_keys = torch.stack(logical_keys)
        all_values = torch.stack(logical_values)
        all_positions = torch.arange(all_keys.shape[0])

        for cell in range(DUPLEXIO_NUM_CELLS):
            query = torch.randn(dim)
            position = row * DUPLEXIO_NUM_CELLS + cell
            logical_visible = duplexio_attention_visible(
                torch.tensor(position),
                all_positions,
                torch.tensor(audio_frame),
                torch.tensor(logical_audio),
                torch.tensor(logical_active),
                audio_attention_window_frames=layout.audio_window_frames,
            )
            logical_scores = query @ all_keys[logical_visible].T
            expected = logical_scores.softmax(-1) @ all_values[logical_visible]

            visible = frame.visible(None, None, torch.tensor(cell), slots)
            # Self K/V comes from the query itself, never from a cache slot.
            compact_keys = torch.cat((cached_keys[visible], keys[cell][None]))
            compact_values = torch.cat((cached_values[visible], values[cell][None]))
            scores = query @ compact_keys.T
            torch.testing.assert_close(scores.softmax(-1) @ compact_values, expected)


def test_audio_ring_reuses_a_bounded_set_of_slots() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=8,
        max_model_len=600_000,
    )
    rows = 500
    tokens = rows * DUPLEXIO_NUM_CELLS
    frame = frame_metadata(layout, rows)
    cells = torch.arange(tokens)
    frame.update(
        key_active=torch.ones(tokens, dtype=torch.bool),
        text_ordinal=torch.zeros(tokens, dtype=torch.int32),
        text_last=torch.zeros(tokens, dtype=torch.int32),
        audio_first=torch.ones(tokens, dtype=torch.int32),
        audio_last=torch.div(cells, DUPLEXIO_NUM_CELLS, rounding_mode="floor").int(),
    )

    written = frame.write_slots(tokens)
    audio_slots = written[cells % DUPLEXIO_NUM_CELLS >= DUPLEXIO_NUM_TEXT_CELLS]

    assert audio_slots.unique().numel() == 2 * (layout.audio_window_frames + 64)
    assert int(audio_slots.max()) < layout.audio_slots
    assert layout.audio_slots <= layout.persistent_text_base


def test_frozen_audio_rows_write_emitted_text_but_no_audio() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=8,
        max_model_len=60,
    )
    frame = frame_metadata(layout)
    install_row(
        frame,
        text_active=torch.tensor([True, False, False, False]),
        emitted_before=3,
        audio_frame=7,
        audio_active=False,
    )

    written = frame.write_slots(DUPLEXIO_NUM_CELLS)

    # Frozen audio must not write: it would clobber the live key already
    # resident at the same audio position.
    assert torch.equal(
        written,
        torch.tensor([layout.persistent_text_base + 3, -1, -1, -1, -1, -1]),
    )


def test_live_row_writes_its_own_audio_frame_into_the_ring() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=8,
        max_model_len=60,
    )
    frame = frame_metadata(layout)
    install_row(
        frame,
        text_active=torch.zeros(DUPLEXIO_NUM_TEXT_CELLS, dtype=torch.bool),
        emitted_before=0,
        audio_frame=3,
    )

    written = frame.write_slots(DUPLEXIO_NUM_CELLS)

    assert torch.equal(written, torch.tensor([-1, -1, -1, -1, 6, 7]))


def test_padded_graph_tokens_neither_write_nor_see_a_slot() -> None:
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=2,
        max_model_len=60,
    )
    frame = frame_metadata(layout, rows=2)
    install_row(
        frame,
        text_active=torch.ones(DUPLEXIO_NUM_TEXT_CELLS, dtype=torch.bool),
        emitted_before=4,
        audio_frame=5,
    )

    tokens = 2 * DUPLEXIO_NUM_CELLS
    written = frame.write_slots(tokens)
    padded = torch.arange(DUPLEXIO_NUM_CELLS, tokens)
    visible = frame.visible(
        None,
        None,
        padded[:, None],
        torch.arange(layout.max_compact_slots)[None],
    )

    assert torch.equal(
        written[DUPLEXIO_NUM_CELLS:], torch.full((DUPLEXIO_NUM_CELLS,), -1)
    )
    assert not visible.any()


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

    assert (
        manager.get_num_blocks_to_allocate(
            "session",
            num_tokens=6,
            new_computed_blocks=[],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=6,
        )
        == layout.max_blocks
    )
    allocated = manager.allocate_new_blocks("session", 6, 6)
    assert len(allocated) == layout.max_blocks
    assert manager.take_new_block_ids() == [block.block_id for block in allocated]
    assert manager.take_new_block_ids() == []
    assert (
        manager.get_num_blocks_to_allocate(
            "session",
            num_tokens=500,
            new_computed_blocks=[],
            total_computed_tokens=494,
            num_local_computed_tokens=494,
            num_tokens_main_model=500,
        )
        == 0
    )

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

    spec = make_duplexio_kv_cache_spec(base, audio_window_frames=8, max_model_len=600)

    assert KVCacheSpecRegistry.get_manager_class(spec) is DuplexIOKVCacheManager
    assert spec.page_size_bytes == base.page_size_bytes
    assert spec.max_num_blocks_per_req(cast(Any, object()), 1) == spec.layout.max_blocks
    assert DuplexIOKVCacheSpec.merge([spec, spec]) is spec


def test_cache_spec_rejects_pages_flex_cannot_tile() -> None:
    base = FullAttentionSpec(
        block_size=528,
        num_kv_heads=4,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="power-of-two"):
        make_duplexio_kv_cache_spec(base, audio_window_frames=8, max_model_len=600)
