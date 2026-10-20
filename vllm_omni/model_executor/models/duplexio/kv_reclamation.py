# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_POSITIONS


@dataclass(frozen=True)
class DuplexIOKVLayout:
    """Physical slots owned by one admitted DuplexIO request, in three regions.

    Each region is addressed by a logical slot that only grows; the rings
    store it modulo their size, so what a position reads of a region is one
    contiguous logical range.

    - The audio ring holds live audio keys: live frame ``f`` (from one) puts
      its user cell at ``2 * (f - 1)`` and its agent cell one after.
    - The BOS ring holds BOS frames' audio keys: BOS frame ``b`` (from zero)
      puts its cells at ``2 * b`` and ``2 * b + 1``.
    - The persistent region holds the keys no window expires, in write order:
      every token and the pinned voice prompt.

    A key with audio clock ``k`` stays visible to a position with clock ``q``
    while ``q - k <= audio_window_frames``. A live frame writes its own frame
    while it reads the window before it, so each ring keeps the last
    ``audio_window_frames + 1`` frames; BOS frames decide distinct live rows,
    so no more of them fit a window either. Queries and no-ops write nothing.
    """

    block_size: int
    audio_window_frames: int
    max_model_len: int

    @property
    def ring_slots(self) -> int:
        return 2 * (self.audio_window_frames + 1)

    @property
    def bos_base(self) -> int:
        return cdiv(self.ring_slots, self.block_size) * self.block_size

    @property
    def persistent_base(self) -> int:
        return self.bos_base + cdiv(self.ring_slots, self.block_size) * self.block_size

    @property
    def max_persistent_keys(self) -> int:
        # The scheduler retains one slot per persistent key (see the model's
        # streaming_retained_tokens), so max_model_len bounds them.
        return self.max_model_len

    @property
    def max_blocks(self) -> int:
        return cdiv(self.persistent_base + self.max_persistent_keys, self.block_size)


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
            raise ValueError("DuplexIO compact KV requires vLLM's hybrid-cache block zeroing")
        super().__init__(kv_cache_spec, **kwargs)
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
        prefill_end: int = 0,
    ) -> int:
        del (
            num_tokens,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
            prefill_end,
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
        raise ValueError("DuplexIO attention does not support quantized KV cache")
    if max_model_len < FRAME_POSITIONS:
        raise ValueError("DuplexIO max_model_len must fit at least one frame")
    if base.block_size % 16:
        raise ValueError(
            "DuplexIO paged FlashAttention needs a KV block size divisible by 16; "
            f"got {base.block_size}. Pin `block_size` in the deployment config."
        )
    return DuplexIOKVCacheSpec(
        block_size=base.block_size,
        num_kv_heads=base.num_kv_heads,
        head_size=base.head_size,
        head_size_v=base.head_size_v,
        dtype=base.dtype,
        kv_quant_mode=base.kv_quant_mode,
        page_size_padded=base.page_size_padded,
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
