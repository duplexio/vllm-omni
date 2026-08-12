# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3.5 backbone layers with DuplexIO's six-cell semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from itertools import islice
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionImpl,
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
    create_block_mask_compiled,
    physical_to_logical_mapping,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    expand_stream_conv_weight,
    mask_inactive_gdn_gates,
)

_METADATA_BYTES = 4
_Q_EPOCH_OFFSET = 0
_K_EPOCH_OFFSET = _Q_EPOCH_OFFSET + _METADATA_BYTES
_K_POSITION_OFFSET = _K_EPOCH_OFFSET + _METADATA_BYTES
_K_TEXT_ORDINAL_OFFSET = _K_POSITION_OFFSET + _METADATA_BYTES
_K_ACTIVE_OFFSET = _K_TEXT_ORDINAL_OFFSET + _METADATA_BYTES
_CACHE_METADATA_SIZE = _K_ACTIVE_OFFSET + _METADATA_BYTES


def _encode_uint32(values: Tensor, dtype: torch.dtype) -> Tensor:
    return torch.stack(
        (
            values & 0xFF,
            (values >> 8) & 0xFF,
            (values >> 16) & 0xFF,
            (values >> 24) & 0xFF,
        ),
        dim=-1,
    ).to(dtype)


def _decode_uint32(values: Tensor) -> Tensor:
    encoded = values.to(torch.long)
    return (
        encoded[..., 0]
        | (encoded[..., 1] << 8)
        | (encoded[..., 2] << 16)
        | (encoded[..., 3] << 24)
    )


def duplexio_compact_key_visible(
    query_positions: Tensor,
    compact_key_positions: Tensor,
    query_epochs: Tensor,
    key_epochs: Tensor,
    key_positions: Tensor,
    text_ordinals: Tensor,
    key_active: Tensor,
    layout: DuplexIOKVLayout,
) -> Tensor:
    """Return exact visibility for keys stored in the compact physical layout."""
    same_request = query_epochs == key_epochs
    query_frames = torch.div(
        query_positions,
        DUPLEXIO_NUM_CELLS,
        rounding_mode="floor",
    )
    key_frames = torch.div(
        key_positions,
        DUPLEXIO_NUM_CELLS,
        rounding_mode="floor",
    )
    key_cells = torch.remainder(key_positions, DUPLEXIO_NUM_CELLS)
    prior_frame = query_frames > key_frames

    audio_cell = key_cells - DUPLEXIO_NUM_TEXT_CELLS
    expected_audio_slot = (
        torch.remainder(key_frames, layout.audio_ring_frames)
        * layout.num_audio_cells
        + audio_cell
    )
    audio_visible = (
        (compact_key_positions < layout.audio_slots)
        & (key_cells >= DUPLEXIO_NUM_TEXT_CELLS)
        & (compact_key_positions == expected_audio_slot)
        & (
            (query_positions == key_positions)
            | (
                key_active
                & prior_frame
                & (
                    query_frames - key_frames
                    <= layout.audio_window_frames
                )
            )
        )
    )

    transient_visible = (
        (compact_key_positions >= layout.transient_text_base)
        & (
            compact_key_positions
            < layout.transient_text_base + DUPLEXIO_NUM_TEXT_CELLS
        )
        & (key_cells < DUPLEXIO_NUM_TEXT_CELLS)
        & (
            compact_key_positions
            == layout.transient_text_base + key_cells
        )
        & (query_positions == key_positions)
    )

    persistent_visible = (
        (compact_key_positions >= layout.persistent_text_base)
        & (key_cells < DUPLEXIO_NUM_TEXT_CELLS)
        & (text_ordinals > 0)
        & key_active
        & (
            compact_key_positions
            == layout.persistent_text_base + text_ordinals - 1
        )
        & prior_frame
    )
    return same_request & (
        audio_visible | transient_visible | persistent_visible
    )


def duplexio_primary_compact_slots(
    positions: Tensor,
    layout: DuplexIOKVLayout,
) -> Tensor:
    """Map the six current frame cells to audio-ring or transient slots."""
    cells = torch.remainder(positions, DUPLEXIO_NUM_CELLS)
    frames = torch.div(positions, DUPLEXIO_NUM_CELLS, rounding_mode="floor")
    audio_slots = (
        torch.remainder(frames, layout.audio_ring_frames)
        * layout.num_audio_cells
        + cells
        - DUPLEXIO_NUM_TEXT_CELLS
    )
    return torch.where(
        cells < DUPLEXIO_NUM_TEXT_CELLS,
        layout.transient_text_base + cells,
        audio_slots,
    )


def duplexio_primary_write_slots(
    positions: Tensor,
    request_indices: Tensor,
    layout: DuplexIOKVLayout,
) -> Tensor:
    """Keep transient text writes only for each request's final scheduled frame."""
    slots = duplexio_primary_compact_slots(positions, layout)
    frames = torch.div(positions, DUPLEXIO_NUM_CELLS, rounding_mode="floor")
    final_frames = torch.full(
        (request_indices.shape[0],),
        -1,
        dtype=frames.dtype,
        device=frames.device,
    )
    final_frames.scatter_reduce_(
        0,
        request_indices,
        frames,
        reduce="amax",
        include_self=True,
    )
    cells = torch.remainder(positions, DUPLEXIO_NUM_CELLS)
    stale_text = (
        cells < DUPLEXIO_NUM_TEXT_CELLS
    ) & (frames != final_frames[request_indices])
    return slots.masked_fill(stale_text, -1)


def _physical_slots(
    block_table: Tensor,
    request_indices: Tensor,
    compact_slots: Tensor,
    block_size: int,
) -> Tensor:
    valid = compact_slots >= 0
    compact_slots = compact_slots.clamp_min(0)
    compact_blocks = torch.div(
        compact_slots,
        block_size,
        rounding_mode="floor",
    )
    physical_blocks = block_table[request_indices, compact_blocks].to(torch.long)
    physical_slots = physical_blocks * block_size + torch.remainder(
        compact_slots,
        block_size,
    )
    return physical_slots.masked_fill(~valid, -1)


class DuplexIOFlexAttentionMetadataBuilder(FlexAttentionMetadataBuilder):
    """Build Flex metadata over DuplexIO's compact per-request address space."""

    _cudagraph_support = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        del kv_cache_spec
        if vllm_config.scheduler_config.max_num_seqs == 1:
            return AttentionCGSupport.ALWAYS
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: DuplexIOKVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.layout = kv_cache_spec.layout
        max_pages = self.layout.max_blocks
        self.max_num_kv_indices = self.q_block_size * max_pages
        self.max_num_rswa_kv_indices = max_pages + 1
        self.compact_seq_lens = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )
        self.full_cudagraph_enabled = (
            vllm_config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL
        )
        blocks_per_page = (
            self.layout.block_size + 2 * self.kv_block_size - 2
        ) // self.kv_block_size
        self.graph_block_offsets = torch.arange(
            blocks_per_page,
            dtype=torch.int32,
            device=device,
        )
        self.graph_kv_indices = torch.full(
            (1, 1, 1, blocks_per_page * self.layout.max_blocks + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.graph_kv_num_blocks = torch.zeros(
            (1, 1, 1),
            dtype=torch.int32,
            device=device,
        )

    def build_graph_block_mask(
        self,
        metadata: FlexAttentionMetadata,
    ) -> BlockMask:
        block_mask = BlockMask(
            seq_lengths=(
                metadata.num_actual_tokens,
                metadata.total_cache_tokens,
            ),
            kv_num_blocks=self.graph_kv_num_blocks,
            kv_indices=self.graph_kv_indices,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            q_num_blocks=None,
            q_indices=None,
            full_q_num_blocks=None,
            full_q_indices=None,
            BLOCK_SIZE=(metadata.q_block_size, metadata.kv_block_size),
            mask_mod=metadata.mask_mod,
        )
        setattr(metadata, "duplexio_graph_block_offsets", self.graph_block_offsets)
        setattr(metadata, "duplexio_graph_block_mask", block_mask)
        update_duplexio_graph_block_mask(
            metadata,
            tuple(range(self.layout.max_blocks)),
        )
        return block_mask

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
        decode_offset = common_attn_metadata.compute_num_computed_tokens()
        compact_seq_lens = self.compact_seq_lens[
            : common_attn_metadata.seq_lens.shape[0]
        ]
        compact_seq_lens.fill_(self.layout.max_compact_slots)
        compact_metadata = common_attn_metadata.replace(
            seq_lens=compact_seq_lens,
            max_seq_len=self.layout.max_compact_slots,
            causal=False,
        )
        metadata = super().build(
            common_prefix_len,
            compact_metadata,
            fast_build=fast_build,
        )
        metadata.decode_offset.copy_(decode_offset)
        setattr(metadata, "duplexio_layout", self.layout)
        if (
            self.full_cudagraph_enabled
            and common_attn_metadata.num_reqs == 1
            and metadata.num_actual_tokens == DUPLEXIO_NUM_CELLS
        ):
            metadata.block_mask = self.build_graph_block_mask(metadata)
        return metadata


def update_duplexio_graph_block_mask(
    metadata: FlexAttentionMetadata,
    compact_page_indices: tuple[int, ...],
) -> None:
    """Update the stable six-cell graph's live physical block list."""
    block_mask = getattr(metadata, "duplexio_graph_block_mask", None)
    if not isinstance(block_mask, BlockMask):
        return
    block_offsets = getattr(metadata, "duplexio_graph_block_offsets")
    page_indices = metadata.block_table.new_tensor(compact_page_indices)
    physical_pages = metadata.block_table[0].index_select(
        0,
        page_indices,
    ).to(torch.int32)
    page_starts = physical_pages * metadata.block_size
    first_flex_blocks = page_starts // metadata.kv_block_size
    candidate_blocks = first_flex_blocks[:, None] + block_offsets[None, :]
    candidate_starts = candidate_blocks * metadata.kv_block_size
    overlaps_page = (
        candidate_starts < page_starts[:, None] + metadata.block_size
    ) & (
        candidate_starts + metadata.kv_block_size > page_starts[:, None]
    )
    flex_blocks = torch.unique(
        candidate_blocks[overlaps_page],
        sorted=True,
    )
    num_flex_blocks = flex_blocks.numel()
    block_mask.kv_indices.fill_(-1)
    block_mask.kv_indices[0, 0, 0, :num_flex_blocks].copy_(flex_blocks)
    block_mask.kv_num_blocks.fill_(num_flex_blocks)


def update_duplexio_attention_metadata(
    attn_metadata: object,
    active_text_tokens: list[int],
    live_audio_frames: list[int],
) -> None:
    """Expose each request's live compact address space to FlexAttention."""
    pending = [attn_metadata]
    seen: set[int] = set()
    while pending:
        metadata = pending.pop()
        if isinstance(metadata, dict):
            pending.extend(metadata.values())
            continue
        if isinstance(metadata, (list, tuple)):
            pending.extend(metadata)
            continue
        if not isinstance(metadata, FlexAttentionMetadata):
            continue
        if id(metadata) in seen:
            continue
        seen.add(id(metadata))
        layout = getattr(metadata, "duplexio_layout", None)
        if not isinstance(layout, DuplexIOKVLayout):
            continue

        num_reqs = metadata.seq_lens.shape[0]
        retained_tokens = active_text_tokens[:num_reqs]
        retained_tokens.extend([0] * (num_reqs - len(retained_tokens)))
        retained_audio_frames = live_audio_frames[:num_reqs]
        retained_audio_frames.extend(
            [0] * (num_reqs - len(retained_audio_frames))
        )
        compact_lengths = [
            layout.live_compact_slots(retained)
            for retained in retained_tokens
        ]
        lengths = metadata.seq_lens.new_tensor(compact_lengths)
        metadata.seq_lens.copy_(lengths)
        metadata.num_blocks_per_seq.copy_(
            torch.div(
                lengths + metadata.block_size - 1,
                metadata.block_size,
                rounding_mode="floor",
            )
        )
        inverse = physical_to_logical_mapping(
            metadata.block_table,
            lengths,
            metadata.block_size,
            metadata.physical_to_logical.shape[1],
        )
        metadata.physical_to_logical.copy_(inverse)
        graph_block_mask = getattr(metadata, "duplexio_graph_block_mask", None)
        if isinstance(graph_block_mask, BlockMask):
            assert metadata.num_reqs == 1
            compact_pages = layout.live_compact_pages(
                live_audio_frames=retained_audio_frames[0],
                active_text_tokens=retained_tokens[0],
            )
            update_duplexio_graph_block_mask(metadata, compact_pages)
            metadata.block_mask = graph_block_mask
        else:
            # The normal multi-request path rebuilds a request-aware block mask
            # in DuplexIOFlexAttentionImpl.forward. A single-request graph mask
            # must never be reused for a batched metadata object.
            metadata.block_mask = None


class DuplexIOFlexAttentionImpl(FlexAttentionImpl):
    """FlexAttention with bounded, role-aware paged-KV writes."""

    def forward(
        self,
        layer: nn.Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kv_cache: Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: Tensor,
        output_scale: Tensor | None = None,
        output_block_scale: Tensor | None = None,
    ) -> Tensor:
        if attn_metadata is not None:
            num_actual_tokens = attn_metadata.num_actual_tokens
            query = query[:num_actual_tokens]
            key = key[:num_actual_tokens]
            value = value[:num_actual_tokens]
            doc_ids = attn_metadata.doc_ids
            assert doc_ids is not None
            layout = getattr(attn_metadata, "duplexio_layout", None)
            assert isinstance(layout, DuplexIOKVLayout)

            key_positions = _decode_uint32(
                key[
                    :,
                    0,
                    self.head_size
                    - _CACHE_METADATA_SIZE
                    + _K_POSITION_OFFSET : self.head_size
                    - _CACHE_METADATA_SIZE
                    + _K_POSITION_OFFSET
                    + _METADATA_BYTES,
                ]
            )
            text_ordinals = _decode_uint32(
                key[
                    :,
                    0,
                    self.head_size
                    - _CACHE_METADATA_SIZE
                    + _K_TEXT_ORDINAL_OFFSET : self.head_size
                    - _CACHE_METADATA_SIZE
                    + _K_TEXT_ORDINAL_OFFSET
                    + _METADATA_BYTES,
                ]
            )
            request_indices = doc_ids[:num_actual_tokens].to(torch.long)
            primary_slots = _physical_slots(
                attn_metadata.block_table,
                request_indices,
                duplexio_primary_write_slots(
                    key_positions,
                    request_indices,
                    layout,
                ),
                attn_metadata.block_size,
            )
            super().do_kv_cache_update(
                layer,
                key,
                value,
                kv_cache,
                primary_slots,
            )

            persistent_compact_slots = (
                layout.persistent_text_base
                + torch.clamp(text_ordinals, min=1)
                - 1
            )
            persistent_slots = _physical_slots(
                attn_metadata.block_table,
                request_indices,
                persistent_compact_slots,
                attn_metadata.block_size,
            )
            persistent_slots = persistent_slots.masked_fill(
                text_ordinals == 0,
                -1,
            )
            super().do_kv_cache_update(
                layer,
                key,
                value,
                kv_cache,
                persistent_slots,
            )

            key_cache = kv_cache.transpose(1, 2)[..., : self.head_size]
            flat_key_cache = key_cache.reshape(
                -1,
                self.num_kv_heads,
                self.head_size,
            )
            metadata_base = self.head_size - _CACHE_METADATA_SIZE
            query_epochs = _decode_uint32(
                query[
                    :,
                    0,
                    metadata_base
                    + _Q_EPOCH_OFFSET : metadata_base
                    + _Q_EPOCH_OFFSET
                    + _METADATA_BYTES,
                ]
            )
            key_epochs = _decode_uint32(
                flat_key_cache[
                    :,
                    0,
                    metadata_base
                    + _K_EPOCH_OFFSET : metadata_base
                    + _K_EPOCH_OFFSET
                    + _METADATA_BYTES,
                ]
            )
            cached_key_positions = _decode_uint32(
                flat_key_cache[
                    :,
                    0,
                    metadata_base
                    + _K_POSITION_OFFSET : metadata_base
                    + _K_POSITION_OFFSET
                    + _METADATA_BYTES,
                ]
            )
            cached_text_ordinals = _decode_uint32(
                flat_key_cache[
                    :,
                    0,
                    metadata_base
                    + _K_TEXT_ORDINAL_OFFSET : metadata_base
                    + _K_TEXT_ORDINAL_OFFSET
                    + _METADATA_BYTES,
                ]
            )
            cached_key_active = _decode_uint32(
                flat_key_cache[
                    :,
                    0,
                    metadata_base
                    + _K_ACTIVE_OFFSET : metadata_base
                    + _K_ACTIVE_OFFSET
                    + _METADATA_BYTES,
                ]
            ).bool()

            def mask_mod(
                _batch: Tensor,
                _head: Tensor,
                query_index: Tensor,
                physical_key_index: Tensor,
            ) -> Tensor:
                is_valid, logical_query, logical_key = (
                    attn_metadata._convert_physical_to_logical(
                        doc_ids,
                        query_index,
                        physical_key_index,
                    )
                )
                visible = duplexio_compact_key_visible(
                    logical_query,
                    logical_key,
                    query_epochs[query_index],
                    key_epochs[physical_key_index],
                    cached_key_positions[physical_key_index],
                    cached_text_ordinals[physical_key_index],
                    cached_key_active[physical_key_index],
                    layout,
                )
                return is_valid & visible

            self.mm_prefix_range = attn_metadata.mm_prefix_range
            attn_metadata.sliding_window = self.sliding_window
            layer_mask_mod = getattr(layer, "logical_mask_mod", None)
            if layer_mask_mod is not None:
                attn_metadata.logical_mask_mod = layer_mask_mod
            layer_hint = getattr(layer, "block_sparsity_hint", None)
            if layer_hint is not None:
                attn_metadata.block_sparsity_hint = layer_hint
            attn_metadata.mask_mod = mask_mod
            graph_block_mask = getattr(
                attn_metadata,
                "duplexio_graph_block_mask",
                None,
            )
            if isinstance(graph_block_mask, BlockMask):
                graph_block_mask.mask_mod = mask_mod
                attn_metadata.block_mask = graph_block_mask
            else:
                attn_metadata.block_mask = create_block_mask_compiled(
                    mask_mod,
                    None,
                    None,
                    attn_metadata.num_actual_tokens,
                    attn_metadata.total_cache_tokens,
                    device=attn_metadata.block_table.device,
                    BLOCK_SIZE=(
                        attn_metadata.q_block_size,
                        attn_metadata.kv_block_size,
                    ),
                )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class DuplexIOFlexAttentionBackend(FlexAttentionBackend):
    """Model-selected FlexAttention backend for DuplexIO full-attention layers."""

    forward_includes_kv_cache_update = True

    @staticmethod
    def get_impl_cls() -> type[DuplexIOFlexAttentionImpl]:
        return DuplexIOFlexAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[DuplexIOFlexAttentionMetadataBuilder]:
        return DuplexIOFlexAttentionMetadataBuilder


class DuplexIOGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """Declare full-graph support for the single-request DuplexIO schedule."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.full_graph_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
        self.full_graph_metadata: GDNAttentionMetadata | None = None

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if vllm_config.scheduler_config.max_num_seqs == 1:
            return AttentionCGSupport.ALWAYS
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def is_full_graph_frame(
        self,
        metadata: CommonAttentionMetadata,
    ) -> bool:
        return (
            metadata.num_reqs == 1
            and metadata.num_actual_tokens == self.full_graph_tokens
        )

    def refresh_full_graph_metadata(
        self,
        common: CommonAttentionMetadata,
    ) -> GDNAttentionMetadata:
        metadata = self.full_graph_metadata
        assert metadata is not None
        state_indices = mamba_get_block_table_tensor(
            common.block_table_tensor,
            common.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )[:, 0]
        assert metadata.non_spec_state_indices_tensor is not None
        metadata.non_spec_state_indices_tensor.copy_(state_indices)
        assert metadata.has_initial_state is not None
        has_initial_state = common.compute_num_computed_tokens() > 0
        metadata.has_initial_state.copy_(has_initial_state)
        assert metadata.prefill_state_indices is not None
        metadata.prefill_state_indices.copy_(state_indices)
        assert metadata.prefill_has_initial_state is not None
        metadata.prefill_has_initial_state.copy_(has_initial_state)
        return metadata

    def retain_full_graph_metadata(
        self,
        metadata: GDNAttentionMetadata,
    ) -> GDNAttentionMetadata:
        assert metadata.non_spec_query_start_loc is not None
        assert metadata.non_spec_state_indices_tensor is not None
        assert metadata.has_initial_state is not None
        assert metadata.chunk_indices is not None
        assert metadata.chunk_offsets is not None
        query_start_loc = metadata.non_spec_query_start_loc.clone()
        state_indices = metadata.non_spec_state_indices_tensor.clone()
        has_initial_state = metadata.has_initial_state.clone()
        retained = replace(
            metadata,
            has_initial_state=has_initial_state,
            chunk_indices=metadata.chunk_indices.clone(),
            chunk_offsets=metadata.chunk_offsets.clone(),
            prefill_query_start_loc=query_start_loc,
            prefill_state_indices=state_indices,
            prefill_has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_state_indices_tensor=state_indices,
        )
        self.full_graph_metadata = retained
        return retained

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: Tensor | None = None,
        num_decode_draft_tokens_cpu: Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        if (
            self.full_graph_metadata is not None
            and self.is_full_graph_frame(common_attn_metadata)
        ):
            return self.refresh_full_graph_metadata(common_attn_metadata)
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        if self.is_full_graph_frame(common_attn_metadata):
            return self.retain_full_graph_metadata(metadata)
        return metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> GDNAttentionMetadata:
        assert self.is_full_graph_frame(common_attn_metadata)
        return self.build(0, common_attn_metadata)


class DuplexIOGDNAttentionBackend(GDNAttentionBackend):
    """GDN metadata backend specialized for DuplexIO's single request."""

    @staticmethod
    def get_name() -> str:
        return "DUPLEXIO_GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type[DuplexIOGDNAttentionMetadataBuilder]:
        return DuplexIOGDNAttentionMetadataBuilder


class DuplexIOPagedAttention(Attention):
    """Attention layer that declares DuplexIO's native cache contract."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        base = super().get_kv_cache_spec(vllm_config)
        if not isinstance(base, FullAttentionSpec):
            return base
        config = vllm_config.model_config.hf_config
        return make_duplexio_kv_cache_spec(
            base,
            audio_window_frames=config.audio_attention_window_frames,
            max_model_len=vllm_config.model_config.max_model_len,
        )


class DuplexIOQwenAttention(nn.Module):
    """Qwen3.5 attention with exact compact-cache metadata."""

    def __init__(
        self,
        config: Any,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (
            self.hidden_size // self.total_num_heads
        )
        self.cache_head_dim = self.head_dim + _CACHE_METADATA_SIZE
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=getattr(
                config,
                "dual_chunk_attention_config",
                None,
            ),
        )
        self.attn = DuplexIOPagedAttention(
            self.num_heads,
            self.cache_head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
            attn_backend=DuplexIOFlexAttentionBackend,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        logical_positions: Tensor,
        hidden_states: Tensor,
        key_active: Tensor,
        request_epochs: Tensor,
        text_ordinals: Tensor,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, key, value = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size],
                dim=-1,
            )
            query, gate = torch.chunk(
                q_gate.view(-1, self.num_heads, 2 * self.head_dim),
                2,
                dim=-1,
            )
            query = query.reshape(-1, self.q_size)
            gate = gate.reshape(-1, self.q_size)
        else:
            query, key, value = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            gate = None

        query = self.q_norm(
            query.view(-1, self.num_heads, self.head_dim)
        ).reshape(-1, self.q_size)
        key = self.k_norm(
            key.view(-1, self.num_kv_heads, self.head_dim)
        ).reshape(-1, self.kv_size)
        query, key = self.rotary_emb(positions, query, key)

        epoch_bytes = _encode_uint32(request_epochs, query.dtype)
        query = torch.cat(
            (
                query.view(-1, self.num_heads, self.head_dim),
                epoch_bytes[:, None, :].expand(-1, self.num_heads, -1),
                query.new_zeros(
                    query.shape[0],
                    self.num_heads,
                    _CACHE_METADATA_SIZE - _METADATA_BYTES,
                ),
            ),
            dim=-1,
        )
        key_metadata = torch.cat(
            (
                key.new_zeros(key.shape[0], _METADATA_BYTES),
                _encode_uint32(request_epochs, key.dtype),
                _encode_uint32(logical_positions, key.dtype),
                _encode_uint32(text_ordinals, key.dtype),
                _encode_uint32(key_active.to(torch.long), key.dtype),
            ),
            dim=-1,
        )
        key = torch.cat(
            (
                key.view(-1, self.num_kv_heads, self.head_dim),
                key_metadata[:, None, :].expand(-1, self.num_kv_heads, -1),
            ),
            dim=-1,
        )
        value = torch.cat(
            (
                value.view(-1, self.num_kv_heads, self.head_dim),
                value.new_zeros(
                    value.shape[0],
                    self.num_kv_heads,
                    _CACHE_METADATA_SIZE,
                ),
            ),
            dim=-1,
        )
        attended = self.attn(query, key, value)
        attended = attended.view(
            -1,
            self.num_heads,
            self.cache_head_dim,
        )[..., : self.head_dim].reshape(-1, self.q_size)
        if gate is not None:
            attended = attended * torch.sigmoid(gate)
        return self.o_proj(attended)[0]


class DuplexIOQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Qwen GDN with six independent causal-convolution histories."""

    def __init__(
        self,
        config: Any,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=False,
        )
        self.full_cudagraph_enabled = (
            vllm_config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL
            and not vllm_config.model_config.enforce_eager
        )
        original_weight = cast(Tensor, self.conv1d.weight)
        original_loader = cast(
            Callable[[Tensor, Tensor], None],
            cast(Any, original_weight).weight_loader,
        )
        expanded_weight = nn.Parameter(
            expand_stream_conv_weight(
                original_weight.detach(),
                num_cells=DUPLEXIO_NUM_CELLS,
            )
        )

        def weight_loader(param: Tensor, loaded_weight: Tensor) -> None:
            original = param.new_empty((*param.shape[:-1], self.conv_kernel_size))
            original_loader(original, loaded_weight)
            param.data.copy_(
                expand_stream_conv_weight(
                    original,
                    num_cells=DUPLEXIO_NUM_CELLS,
                )
            )

        set_weight_attrs(expanded_weight, {"weight_loader": weight_loader})
        self.conv1d.weight = expanded_weight

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:  # ty: ignore[invalid-method-override]
        # vLLM 0.26's base annotation says four shapes, but its calculator and
        # cache-manager contract both return the convolution and temporal pair.
        effective_kernel_size = (
            (self.conv_kernel_size - 1) * DUPLEXIO_NUM_CELLS + 1
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            effective_kernel_size,
            self.num_spec,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.full_cudagraph_enabled:
            return DuplexIOGDNAttentionBackend
        return super().get_attn_backend()

    def apply_stream_causal_conv(
        self,
        mixed_qkv: Tensor,
        conv_state: Tensor,
        state_indices: Tensor,
        query_start_loc: Tensor,
        has_initial_state: Tensor | None,
    ) -> Tensor:
        """Apply the expanded causal convolution and update its state cache.

        vLLM's Triton causal-convolution kernel only supports the original
        Qwen kernel widths. DuplexIO expands that kernel across six cells, so
        the effective width is 19. This grouped PyTorch operation preserves
        the same state contract for the expanded width.
        """
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        state_length = conv_weights.size(-1) - 1
        request_count = query_start_loc.shape[0] - 1
        if self.full_cudagraph_enabled and request_count == 1:
            state_indices = state_indices[:1]
            history = conv_state.index_select(0, state_indices)
            if has_initial_state is not None:
                history = history.masked_fill(
                    ~has_initial_state[:1, None, None],
                    0,
                )
            conv_input = torch.cat(
                (history, mixed_qkv.transpose(0, 1).unsqueeze(0)),
                dim=-1,
            )
            conv_output = F.conv1d(
                conv_input,
                conv_weights.unsqueeze(1),
                bias=self.conv1d.bias,
                groups=conv_weights.size(0),
            )
            if self.activation != "silu":
                raise ValueError(
                    "DuplexIO's native convolution path requires silu, "
                    f"got {self.activation!r}"
                )
            conv_state[state_indices] = conv_input[:, :, -state_length:]
            conv_output = conv_output.squeeze(0).transpose(0, 1).contiguous()
            return F.silu(conv_output)

        if request_count > 1:
            query_lengths = query_start_loc[1:] - query_start_loc[:-1]
            if bool(torch.all(query_lengths == query_lengths[0])):
                query_length = mixed_qkv.shape[0] // request_count
                assert mixed_qkv.shape[0] == request_count * query_length
                history = conv_state.index_select(
                    0,
                    state_indices[:request_count],
                )
                if has_initial_state is not None:
                    history = history.masked_fill(
                        ~has_initial_state[:request_count, None, None],
                        0,
                    )
                conv_input = torch.cat(
                    (
                        history,
                        mixed_qkv.view(
                            request_count,
                            query_length,
                            -1,
                        ).transpose(1, 2),
                    ),
                    dim=-1,
                )
                conv_output = F.conv1d(
                    conv_input,
                    conv_weights.unsqueeze(1),
                    bias=self.conv1d.bias,
                    groups=conv_weights.size(0),
                )
                if self.activation != "silu":
                    raise ValueError(
                        "DuplexIO's native convolution path requires silu, "
                        f"got {self.activation!r}"
                    )
                conv_state[state_indices[:request_count]] = conv_input[
                    :, :, -state_length:
                ]
                return F.silu(
                    conv_output.transpose(1, 2).contiguous().view_as(mixed_qkv)
                )

        output = torch.empty_like(mixed_qkv)

        for request_index in range(query_start_loc.numel() - 1):
            start = int(query_start_loc[request_index].item())
            end = int(query_start_loc[request_index + 1].item())
            if start == end:
                continue

            state_index = int(state_indices[request_index].item())
            history = conv_state[state_index]
            if (
                has_initial_state is not None
                and not bool(has_initial_state[request_index].item())
            ):
                history = torch.zeros_like(history)

            conv_input = torch.cat(
                (history, mixed_qkv[start:end].transpose(0, 1)),
                dim=-1,
            ).unsqueeze(0)
            conv_output = F.conv1d(
                conv_input,
                conv_weights.unsqueeze(1),
                bias=self.conv1d.bias,
                groups=conv_weights.size(0),
            )
            if self.activation != "silu":
                raise ValueError(
                    "DuplexIO's native convolution path requires silu, "
                    f"got {self.activation!r}"
                )
            output[start:end] = F.silu(
                conv_output.squeeze(0).transpose(0, 1)
            )
            conv_state[state_index].copy_(conv_input[0, :, -state_length:])

        return output

    def _forward_core(
        self,
        mixed_qkv: Tensor,
        b: Tensor,
        a: Tensor,
        core_attn_out: Tensor,
    ) -> None:
        """Run GDN with DuplexIO's expanded convolution state."""
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if attn_metadata.spec_sequence_masks is not None:
            raise RuntimeError(
                "Native DuplexIO does not support speculative GDN execution"
            )

        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata.non_spec_state_indices_tensor
        )
        assert non_spec_query_start_loc is not None
        assert non_spec_state_indices_tensor is not None

        self_kv_cache = self.kv_cache
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        mixed_qkv = self.apply_stream_causal_conv(
            mixed_qkv=mixed_qkv,
            conv_state=conv_state,
            state_indices=non_spec_state_indices_tensor,
            query_start_loc=non_spec_query_start_loc,
            has_initial_state=attn_metadata.has_initial_state,
        )

        split_non_spec = (
            attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            if split_non_spec:
                conv_output_prefill = mixed_qkv[num_decode_tokens:]
                a_prefill = a[num_decode_tokens:]
                b_prefill = b[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv
                a_prefill = a
                b_prefill = b

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=conv_output_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = (
                self.rearrange_mixed_qkv(mixed_qkv)
            )
            g_non_spec = None
            beta_non_spec = None

        if split_non_spec:
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv[:num_decode_tokens]
            )
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=self.dt_bias,
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        if attn_metadata.num_prefills > 0:
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            prefill_query_start_loc = attn_metadata.prefill_query_start_loc
            chunk_indices = attn_metadata.chunk_indices
            chunk_offsets = attn_metadata.chunk_offsets
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            assert prefill_query_start_loc is not None
            assert chunk_indices is not None
            assert chunk_offsets is not None

            initial_state = ssm_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0
            core_attn_out_non_spec, last_recurrent_state = (
                self.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=prefill_query_start_loc,
                    chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets,
                    use_qk_l2norm_in_kernel=False,
                )
            )
            ssm_state[prefill_state_indices] = last_recurrent_state.to(
                ssm_state.dtype
            )
            if split_non_spec:
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec],
                    dim=1,
                )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, _ = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            return

        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def forward_with_key_activity(
        self,
        hidden_states: Tensor,
        key_active: Tensor,
    ) -> Tensor:
        num_tokens = hidden_states.shape[0]
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_ba, _ = self.in_proj_ba(hidden_states)
        beta_logits, decay_logits = self.split_ba(projected_ba)
        beta_logits = beta_logits.contiguous()
        decay_logits = decay_logits.contiguous()
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, output_gate = mixed_qkvz.split(
            [qkv_size, z_size],
            dim=-1,
        )
        output_gate = output_gate.reshape(num_tokens, -1, self.head_v_dim)
        beta_logits, decay_logits = mask_inactive_gdn_gates(
            beta_logits,
            decay_logits,
            key_active,
        )
        core_output = hidden_states.new_zeros(
            num_tokens,
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
        )
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            beta_logits,
            decay_logits,
            core_output,
            layer_name=_encode_layer_name(self.prefix),
        )
        return self._output_projection(core_output, output_gate)


class DuplexIOQwenDecoderLayer(nn.Module):
    """Dense Qwen3.5 decoder layer with explicit DuplexIO metadata flow."""

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if config.model_type != "qwen3_5_text":
            raise ValueError(
                "Native DuplexIO requires the dense qwen3_5_text backbone"
            )
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.use_attn_reduce_scatter_for_moe = False
        if layer_type == "linear_attention":
            self.linear_attn = DuplexIOQwenGatedDeltaNetAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif layer_type == "full_attention":
            self.self_attn = DuplexIOQwenAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid Qwen3.5 layer type {layer_type!r}")
        self.mlp = Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )
            self.ffn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None,
        positions: Tensor,
        logical_positions: Tensor,
        key_active: Tensor,
        request_epochs: Tensor,
        text_ordinals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn.forward_with_key_activity(
                hidden_states,
                key_active,
            )
        else:
            hidden_states = self.self_attn(
                positions,
                logical_positions,
                hidden_states,
                key_active,
                request_epochs,
                text_ordinals,
            )
        if self.layer_scale:
            hidden_states = hidden_states * (
                self.attn_layer_scale[0].to(hidden_states.dtype) + 1
            )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        if self.layer_scale:
            hidden_states = hidden_states * (
                self.ffn_layer_scale[0].to(hidden_states.dtype) + 1
            )
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "positions": 0,
        "logical_positions": 0,
        "key_active": 0,
        "request_epochs": 0,
        "text_ordinals": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class DuplexIOQwenModel(nn.Module):
    """Inference-only dense Qwen3.5 model used by native DuplexIO."""

    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_suffix={".scale": ".weight"},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str) -> DuplexIOQwenDecoderLayer:
            layer_index = extract_layer_index(prefix)
            return DuplexIOQwenDecoderLayer(
                vllm_config,
                config.layer_types[layer_index],
                prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                config.hidden_size,
            )
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        positions: Tensor,
        logical_positions: Tensor,
        key_active: Tensor,
        request_epochs: Tensor,
        text_ordinals: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            assert inputs_embeds is not None
            hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                key_active=key_active,
                logical_positions=logical_positions,
                request_epochs=request_epochs,
                text_ordinals=text_ordinals,
            )
        if not get_pp_group().is_last_rank:
            assert residual is not None
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        return self.norm(hidden_states, residual)[0]

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )


__all__ = [
    "DuplexIOFlexAttentionBackend",
    "DuplexIOFlexAttentionMetadataBuilder",
    "DuplexIOGDNAttentionBackend",
    "DuplexIOGDNAttentionMetadataBuilder",
    "DuplexIOPagedAttention",
    "DuplexIOQwenAttention",
    "DuplexIOQwenDecoderLayer",
    "DuplexIOQwenGatedDeltaNetAttention",
    "DuplexIOQwenModel",
    "duplexio_compact_key_visible",
    "duplexio_primary_compact_slots",
]
