# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded paged-KV layout for DuplexIO full-attention layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
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

    The first region is a ring for active audio cells. The final region stores
    active text cells and is never reclaimed during the request. Query-local
    self-attention uses incoming K/V directly and needs no cache slots.
    """

    block_size: int
    audio_window_frames: int
    max_model_len: int

    @property
    def max_frames(self) -> int:
        return self.max_model_len // DUPLEXIO_NUM_CELLS

    @property
    def audio_ring_frames(self) -> int:
        # Training evicts whole 128-key (64-audio-frame) blocks. Retaining the
        # masked block tail preserves its reduction alignment after eviction.
        return min(self.audio_window_frames + 64, self.max_frames)

    @property
    def audio_slots(self) -> int:
        return self.audio_ring_frames * self.num_audio_cells

    @property
    def num_audio_cells(self) -> int:
        return DUPLEXIO_NUM_CELLS - DUPLEXIO_NUM_TEXT_CELLS

    @property
    def persistent_text_base(self) -> int:
        return cdiv(self.audio_slots, self.block_size) * self.block_size

    @property
    def text_base_page(self) -> int:
        return self.persistent_text_base // self.block_size

    @property
    def max_persistent_text_tokens(self) -> int:
        return self.max_frames * DUPLEXIO_NUM_TEXT_CELLS

    @property
    def max_compact_slots(self) -> int:
        return self.persistent_text_base + self.max_persistent_text_tokens

    @property
    def max_blocks(self) -> int:
        return cdiv(self.max_compact_slots, self.block_size)

    def audio_slot(self, audio_position: int, audio_cell: int) -> int:
        assert 0 <= audio_cell < self.num_audio_cells
        return (
            audio_position % self.audio_ring_frames
        ) * self.num_audio_cells + audio_cell

    def persistent_text_slot(self, text_ordinal: int) -> int:
        assert 0 <= text_ordinal < self.max_persistent_text_tokens
        return self.persistent_text_base + text_ordinal


class DuplexIOFrameMetadata:
    """Per-token cache addressing for one model step.

    A cell's slot decides its position, so nothing has to be stored alongside
    the key: an audio slot is the ring residue of its frame, a text slot is
    dense in emission order. One instance is shared by every full-attention
    layer, which keeps the addressing and the attention mask in one place.

    Buffers are sized for the largest batch and keep stable addresses for
    CUDA-graph replay. Whatever the current batch does not cover stays inert,
    so padded graph tokens neither write a slot nor see a key.
    """

    def __init__(
        self,
        layout: DuplexIOKVLayout,
        max_tokens: int,
        device: torch.device,
    ) -> None:
        self.layout = layout
        self.cell = torch.arange(max_tokens, device=device) % DUPLEXIO_NUM_CELLS
        self.key_active = torch.ones(max_tokens, dtype=torch.bool, device=device)
        self.text_ordinal = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.text_last = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.audio_first = torch.ones(max_tokens, dtype=torch.int32, device=device)
        self.audio_last = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
        self.filled = 0

    def update(
        self,
        *,
        key_active: Tensor,
        text_ordinal: Tensor,
        text_last: Tensor,
        audio_first: Tensor,
        audio_last: Tensor,
    ) -> None:
        """Install this step's cells, one row of each tensor per token."""
        tokens = key_active.shape[0]
        if tokens < self.filled:
            self.reset(tokens, self.filled)
        self.filled = tokens
        self.key_active[:tokens].copy_(key_active)
        self.text_ordinal[:tokens].copy_(text_ordinal)
        self.text_last[:tokens].copy_(text_last)
        self.audio_first[:tokens].copy_(audio_first)
        self.audio_last[:tokens].copy_(audio_last)

    def reset(self, start: int = 0, end: int | None = None) -> None:
        """Make tokens inert: no slot to write, no key in the window."""
        region = slice(start, end)
        self.key_active[region] = True
        self.text_ordinal[region] = 0
        self.text_last[region] = 0
        self.audio_first[region] = 1
        self.audio_last[region] = -1
        self.filled = min(self.filled, start)

    def write_slots(self, tokens: int) -> Tensor:
        """Compact slot per cell, -1 for cells that must not enter the cache.

        Audio cells land on their own frame, one past the last frame they see.
        Text cells land on their emission ordinal; an unemitted cell has ordinal
        zero and is skipped.
        """
        layout = self.layout
        cell = self.cell[:tokens]
        audio_last = self.audio_last[:tokens]
        text_ordinal = self.text_ordinal[:tokens]
        is_audio = cell >= DUPLEXIO_NUM_TEXT_CELLS
        audio_slot = (
            torch.remainder(audio_last + 1, layout.audio_ring_frames) * layout.num_audio_cells
            + cell
            - DUPLEXIO_NUM_TEXT_CELLS
        )
        stored = torch.where(
            is_audio,
            self.key_active[:tokens] & (audio_last >= 0),
            text_ordinal > 0,
        )
        text_slot = layout.persistent_text_base + text_ordinal - 1
        return torch.where(is_audio, audio_slot, text_slot).masked_fill(~stored, -1)

    def visible(self, batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        """Return whether a query cell sees a compact slot of a prior row.

        Its own cell is merged separately, so no same-row key is visible here.
        An audio slot's frame is the most recent one with its ring residue; slots
        that were never written, or that the window has passed, derive a frame
        below the query's first visible frame.
        """
        del batch, head
        layout = self.layout
        audio_last = self.audio_last[query]
        frame = audio_last - torch.remainder(
            audio_last - torch.div(key, layout.num_audio_cells, rounding_mode="floor"),
            layout.audio_ring_frames,
        )
        audio = (key < layout.audio_slots) & (frame >= self.audio_first[query])
        text = (key >= layout.persistent_text_base) & (
            key - layout.persistent_text_base < self.text_last[query]
        )
        return audio | text


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
    if base.block_size & (base.block_size - 1):
        raise ValueError(
            "DuplexIO paged FlexAttention needs a power-of-two KV block size; "
            f"got {base.block_size}. Pin `block_size` in the deployment config."
        )
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
    "DuplexIOFrameMetadata",
    "DuplexIOKVCacheManager",
    "DuplexIOKVCacheSpec",
    "DuplexIOKVLayout",
    "make_duplexio_kv_cache_spec",
]
