# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded paged-KV layout for DuplexIO full-attention layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from typing_extensions import Self
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
)


@dataclass(frozen=True)
class DuplexIOKVLayout:
    """Physical slots owned by one admitted DuplexIO request.

    The first region is a ring for the two audio cells in each frame. The
    second region holds the four current text cells so inactive text still has
    exact self-attention. The final region stores only active text cells and is
    never reclaimed during the request.
    """

    block_size: int
    audio_window_frames: int
    max_model_len: int

    @property
    def max_frames(self) -> int:
        return self.max_model_len // DUPLEXIO_NUM_CELLS

    @property
    def audio_ring_frames(self) -> int:
        # The ring is keyed by audio position (a non-decreasing cumsum that
        # advances only on frames carrying real audio). A query at audio
        # position p sees audio positions p-window through p, inclusive, so
        # window + 1 distinct positions must stay resident. Audio positions
        # advance at most once per frame, so max_frames stays an upper bound.
        return min(self.audio_window_frames + 1, self.max_frames)

    @property
    def audio_slots(self) -> int:
        return self.audio_ring_frames * self.num_audio_cells

    @property
    def num_audio_cells(self) -> int:
        return DUPLEXIO_NUM_CELLS - DUPLEXIO_NUM_TEXT_CELLS

    @property
    def transient_text_base(self) -> int:
        return cdiv(self.audio_slots, self.block_size) * self.block_size

    @property
    def persistent_text_base(self) -> int:
        return self.transient_text_base + self.block_size

    @property
    def max_persistent_text_tokens(self) -> int:
        return self.max_frames * DUPLEXIO_NUM_TEXT_CELLS

    @property
    def max_compact_slots(self) -> int:
        return self.persistent_text_base + self.max_persistent_text_tokens

    @property
    def max_blocks(self) -> int:
        return cdiv(self.max_compact_slots, self.block_size)

    def live_compact_slots(self, active_text_tokens: int) -> int:
        """Slots Flex must scan for the accepted text plus one frame."""
        assert 0 <= active_text_tokens <= self.max_persistent_text_tokens
        return self.persistent_text_base + min(
            active_text_tokens + DUPLEXIO_NUM_TEXT_CELLS,
            self.max_persistent_text_tokens,
        )

    def live_compact_pages(
        self,
        *,
        live_audio_frames: int,
        active_text_tokens: int,
    ) -> tuple[int, ...]:
        """Compact pages containing live audio, transient text, or active text."""
        assert live_audio_frames >= 0
        assert 0 <= active_text_tokens <= self.max_persistent_text_tokens

        live_audio_slots = (
            min(live_audio_frames, self.audio_ring_frames)
            * self.num_audio_cells
        )
        audio_pages = range(cdiv(live_audio_slots, self.block_size))
        transient_page = self.transient_text_base // self.block_size
        live_persistent_tokens = min(
            active_text_tokens + DUPLEXIO_NUM_TEXT_CELLS,
            self.max_persistent_text_tokens,
        )
        persistent_page = self.persistent_text_base // self.block_size
        persistent_pages = range(
            persistent_page,
            persistent_page + cdiv(live_persistent_tokens, self.block_size),
        )
        return (*audio_pages, transient_page, *persistent_pages)

    def audio_slot(self, audio_position: int, audio_cell: int) -> int:
        assert 0 <= audio_cell < self.num_audio_cells
        return (
            audio_position % self.audio_ring_frames
        ) * self.num_audio_cells + audio_cell

    def transient_text_slot(self, text_cell: int) -> int:
        assert 0 <= text_cell < DUPLEXIO_NUM_TEXT_CELLS
        return self.transient_text_base + text_cell

    def persistent_text_slot(self, text_ordinal: int) -> int:
        assert 0 <= text_ordinal < self.max_persistent_text_tokens
        return self.persistent_text_base + text_ordinal


@dataclass(frozen=True, kw_only=True)
class DuplexIOKVCacheSpec(FullAttentionSpec):
    """Full-attention pages using DuplexIO's role-aware physical layout."""

    audio_window_frames: int
    max_model_len: int

    @property
    def layout(self) -> DuplexIOKVLayout:
        return DuplexIOKVLayout(
            block_size=self.block_size,
            audio_window_frames=self.audio_window_frames,
            max_model_len=self.max_model_len,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        del vllm_config
        return self.layout.max_blocks * self.page_size_bytes

    def max_num_blocks_per_req(
        self,
        vllm_config: VllmConfig,
        max_len: int,
    ) -> int:
        del vllm_config, max_len
        return self.layout.max_blocks

    @classmethod
    def merge(cls, specs: list[FullAttentionSpec]) -> Self:
        assert specs
        assert all(isinstance(spec, cls) for spec in specs), (
            "DuplexIO cache groups cannot contain another full-attention spec."
        )
        first = cast(Self, specs[0])
        assert all(spec == first for spec in specs[1:]), (
            "All DuplexIO full-attention layers must use the same cache layout."
        )
        return first


class DuplexIOKVCacheManager(FullAttentionManager):
    """Reserve the complete bounded layout when a live session is admitted.

    A DuplexIO request cannot be recomputed from scheduler token IDs because
    its prior inputs are PCM frames. Reserving the complete bounded layout up
    front means an admitted request never asks the scheduler for another KV
    block and therefore cannot be preempted by its own cache growth.
    """

    def __init__(self, kv_cache_spec: DuplexIOKVCacheSpec, **kwargs) -> None:
        if kwargs.get("enable_caching"):
            raise ValueError("DuplexIO role-aware KV does not support prefix caching")
        if not kwargs.get("needs_kv_cache_zeroing"):
            raise ValueError(
                "DuplexIO compact KV requires vLLM's hybrid-cache block zeroing"
            )
        super().__init__(kv_cache_spec, **kwargs)
        # vLLM 0.26 records only exact built-in full-attention spec types.
        # This registered full-attention subtype needs the same lifecycle.
        self._record_new_block_ids = True
        self.layout = kv_cache_spec.layout

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            num_tokens,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        if new_computed_blocks:
            raise ValueError("DuplexIO role-aware KV cannot consume prefix hits")
        return max(
            self.layout.max_blocks - len(self.req_to_blocks.get(request_id, ())),
            0,
        )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        del num_tokens, num_tokens_main_model
        request_blocks = self.req_to_blocks[request_id]
        num_new_blocks = self.layout.max_blocks - len(request_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        request_blocks.extend(new_blocks)
        self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks


# Model modules can be imported before a worker has asked the lazy registry for
# a built-in spec. Initialize the built-ins first so this out-of-tree entry does
# not become the registry's sole entry in that process.
register_all_kvcache_specs(get_current_vllm_config_or_none())
KVCacheSpecRegistry.register(
    DuplexIOKVCacheSpec,
    DuplexIOKVCacheManager,
    uniform_type_base_spec=DuplexIOKVCacheSpec,
)


def make_duplexio_kv_cache_spec(
    base: FullAttentionSpec,
    *,
    audio_window_frames: int,
    max_model_len: int,
) -> DuplexIOKVCacheSpec:
    """Convert an Attention-produced full spec to DuplexIO's compact spec."""
    if base.kv_quant_mode != KVQuantMode.NONE:
        raise ValueError("DuplexIO FlexAttention does not support quantized KV cache")
    if max_model_len < DUPLEXIO_NUM_CELLS:
        raise ValueError("DuplexIO max_model_len must fit at least one frame")
    if max_model_len > 2**32:
        raise ValueError("DuplexIO max_model_len exceeds its uint32 cache metadata")
    if audio_window_frames < 0:
        raise ValueError("DuplexIO audio attention window must be non-negative")
    return DuplexIOKVCacheSpec(
        block_size=base.block_size,
        num_kv_heads=base.num_kv_heads,
        head_size=base.head_size,
        head_size_v=base.head_size_v,
        dtype=base.dtype,
        kv_quant_mode=base.kv_quant_mode,
        page_size_padded=base.page_size_padded,
        indexes_kv_by_block_stride=base.indexes_kv_by_block_stride,
        sliding_window=base.sliding_window,
        attention_chunk_size=base.attention_chunk_size,
        non_causal=base.non_causal,
        audio_window_frames=audio_window_frames,
        max_model_len=max_model_len,
    )


__all__ = [
    "DuplexIOKVCacheManager",
    "DuplexIOKVCacheSpec",
    "DuplexIOKVLayout",
    "make_duplexio_kv_cache_spec",
]
