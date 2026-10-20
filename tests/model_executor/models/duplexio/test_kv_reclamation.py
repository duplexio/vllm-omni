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
    DuplexIOKVCacheManager,
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)


def test_layout_regions_are_page_aligned_and_bounded() -> None:
    layout = DuplexIOKVLayout(block_size=16, audio_window_frames=8, max_model_len=600)
    # Each ring keeps the window plus the frame being written, two keys a frame.
    assert layout.ring_slots == 18
    assert layout.bos_base == 32 and layout.persistent_base == 64
    assert layout.max_blocks == (64 + 600 + 15) // 16


@dataclass(frozen=True)
class _Block:
    block_id: int


class _BlockPool:
    def __init__(self) -> None:
        self.next_block = 0
        self.freed: list[object] = []

    def get_new_blocks(self, count: int) -> list[_Block]:
        blocks = [_Block(block_id) for block_id in range(self.next_block, self.next_block + count)]
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


def test_cache_spec_rejects_pages_flash_attention_cannot_tile() -> None:
    base = FullAttentionSpec(
        block_size=520,
        num_kv_heads=4,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="divisible by 16"):
        make_duplexio_kv_cache_spec(base, audio_window_frames=8, max_model_len=600)
