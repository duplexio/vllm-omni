# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3.5 backbone layers with DuplexIO's six-cell semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import islice
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
)
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
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionImpl,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

from vllm_omni.model_executor.models.duplexio.attention_semantics import (
    key_visible,
)
from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.numerics import call_compiled_function, fixed_linear
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    expand_stream_conv_weight,
    mask_inactive_gdn_gates,
)
from vllm_omni.model_executor.models.duplexio.stream_attention import (
    cached_rotary_pos_emb,
    gated_attention_output,
    gather_history,
    history_and_self_attention,
    history_block_mask,
    packed_history_indices,
)
from vllm_omni.model_executor.models.duplexio.stream_conv import stream_causal_conv
from vllm_omni.model_executor.models.duplexio.stream_gdn import (
    append_gdn,
    gdn_cache_dtypes,
    gdn_cache_shapes,
    prepare_gdn_inputs,
)

_METADATA_BYTES = 4
_Q_EPOCH_OFFSET = 0
_K_EPOCH_OFFSET = _Q_EPOCH_OFFSET + _METADATA_BYTES
_K_POSITION_OFFSET = _K_EPOCH_OFFSET + _METADATA_BYTES
_K_TEXT_ORDINAL_OFFSET = _K_POSITION_OFFSET + _METADATA_BYTES
_K_ACTIVE_OFFSET = _K_TEXT_ORDINAL_OFFSET + _METADATA_BYTES
_K_AUDIO_POS_OFFSET = _K_ACTIVE_OFFSET + _METADATA_BYTES
_CACHE_METADATA_SIZE = _K_AUDIO_POS_OFFSET + _METADATA_BYTES


class DuplexIORotaryEmbedding(nn.Module):
    """Cache phases from exported frequencies, never reconstruct their precision."""

    def __init__(self, rotary_dim: int, max_positions: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.register_buffer("inverse_frequencies", torch.empty(rotary_dim // 2, dtype=torch.float32))
        self.register_buffer("attention_scaling", torch.empty((), dtype=torch.float32))
        self.register_buffer("cos_sin_cache", torch.empty(0, rotary_dim, dtype=dtype), persistent=False)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights)
        if loaded != {"inverse_frequencies", "attention_scaling"}:
            raise ValueError(f"Incomplete exported RoPE constants: {sorted(loaded)}")
        positions = torch.arange(self.max_positions, device=self.inverse_frequencies.device, dtype=torch.float32)
        phases = torch.outer(positions, self.inverse_frequencies)
        self.cos_sin_cache = (
            torch.cat((phases.cos(), phases.sin()), -1) * self.attention_scaling
        ).to(self.cos_sin_cache.dtype)
        return loaded


class DuplexIORMSNorm(nn.Module):
    """Use the training kernel and preserve its FP32 direct scales."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(
        self, hidden: Tensor, residual: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        from quack import rmsnorm

        return rmsnorm(hidden, self.weight, residual=residual, eps=self.eps, prenorm=residual is not None)


class DuplexIOQwenMLP(Qwen3NextMLP):
    """Keep vLLM's sharded weights but use training's fused SwiGLU math."""

    def forward(self, hidden: Tensor) -> Tensor:
        from quack.mlp import mlp_func

        output = mlp_func(
            hidden,
            self.gate_up_proj.weight,
            self.down_proj.weight,
            activation="swiglu",
            recompute=False,
            concat_layout=True,
            tuned=False,
        )
        if self.down_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


@torch.compile(dynamic=True, fullgraph=True)
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


@torch.compile(dynamic=True, fullgraph=True)
def _decode_uint32(values: Tensor) -> Tensor:
    encoded = values.to(torch.long)
    return (
        encoded[..., 0]
        | (encoded[..., 1] << 8)
        | (encoded[..., 2] << 16)
        | (encoded[..., 3] << 24)
    )


@torch.compile(dynamic=True, fullgraph=True)
def owned_history_metadata(
    cache: Tensor, block_table: Tensor, block_size: int, blocks_per_request: int,
) -> tuple[Tensor, Tensor]:
    """Decode only admitted requests' pages, in one fused indexed read."""
    pages = block_table[:, :blocks_per_request].long()
    physical = (pages[:, :, None] * block_size + torch.arange(block_size, device=cache.device)).flatten(1)
    encoded = cache[physical, 0, -_CACHE_METADATA_SIZE:].view(*physical.shape, 6, _METADATA_BYTES)
    return physical, _decode_uint32(encoded)


def duplexio_compact_key_visible(
    query_positions: Tensor,
    compact_key_positions: Tensor,
    query_epochs: Tensor,
    key_epochs: Tensor,
    key_positions: Tensor,
    text_ordinals: Tensor,
    key_active: Tensor,
    query_audio_positions: Tensor,
    key_audio_positions: Tensor,
    layout: DuplexIOKVLayout,
) -> Tensor:
    """Return prior-frame visibility; query-local self K/V bypass the cache."""
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

    # Cross-frame visibility follows the training predicate: strict row
    # causality, active keys only, and the audio-time window (text keys are
    # never windowed; frozen-audio-time frames do not consume window budget).
    cross_frame_visible = key_visible(
        query_frames,
        key_frames,
        query_audio_positions,
        key_audio_positions,
        key_cells,
        key_active,
        layout.audio_window_frames,
        False,
    )

    audio_cell = key_cells - DUPLEXIO_NUM_TEXT_CELLS
    expected_audio_slot = (
        torch.remainder(key_audio_positions, layout.audio_ring_frames)
        * layout.num_audio_cells
        + audio_cell
    )
    audio_visible = (
        (compact_key_positions < layout.audio_slots)
        & (key_cells >= DUPLEXIO_NUM_TEXT_CELLS)
        & (compact_key_positions == expected_audio_slot)
        & cross_frame_visible
    )

    persistent_visible = (
        (compact_key_positions >= layout.persistent_text_base)
        & (key_cells < DUPLEXIO_NUM_TEXT_CELLS)
        & (text_ordinals > 0)
        & (
            compact_key_positions
            == layout.persistent_text_base + text_ordinals - 1
        )
        & cross_frame_visible
    )
    return same_request & (audio_visible | persistent_visible)


def duplexio_audio_write_slots(
    positions: Tensor,
    audio_positions: Tensor,
    key_active: Tensor,
    layout: DuplexIOKVLayout,
) -> Tensor:
    """Store only live audio; masked prefix cells must not overwrite history."""
    cells = torch.remainder(positions, DUPLEXIO_NUM_CELLS)
    audio_slots = (
        torch.remainder(audio_positions, layout.audio_ring_frames)
        * layout.num_audio_cells
        + cells
        - DUPLEXIO_NUM_TEXT_CELLS
    )
    return audio_slots.masked_fill((cells < DUPLEXIO_NUM_TEXT_CELLS) | ~key_active, -1)




@torch.compile(dynamic=True, fullgraph=True)
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


@dataclass
class DuplexIOAttentionMetadata(AttentionMetadata):
    """Only the request layout needed by training-ordered paged attention."""

    num_actual_tokens: int
    num_query_batches: int
    query_start_loc: Tensor
    block_table: Tensor
    doc_ids: Tensor
    block_size: int
    duplexio_layout: DuplexIOKVLayout
    duplexio_full_graph: bool
    duplexio_packed_capacity: int


class DuplexIOFlexAttentionMetadataBuilder(AttentionMetadataBuilder[DuplexIOAttentionMetadata]):
    """Build request ownership, not the generic backend's unused cache-pool mask."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

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
        self.doc_ids = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.long,
            device=device,
        )
        self.full_cudagraph_enabled = (
            vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DuplexIOAttentionMetadata:
        del common_prefix_len, fast_build
        common = common_attn_metadata
        doc_ids = self.doc_ids[:common.num_actual_tokens]
        torch.searchsorted(
            common.query_start_loc[1:],
            torch.arange(common.num_actual_tokens, device=self.device),
            right=True,
            out=doc_ids,
        )
        uniform_frames = (
            common.max_query_len == DUPLEXIO_NUM_CELLS
            and common.num_actual_tokens == DUPLEXIO_NUM_CELLS * common.num_reqs
        )
        return DuplexIOAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            num_query_batches=common.num_reqs if uniform_frames else 1,
            query_start_loc=common.query_start_loc,
            block_table=common.block_table_tensor,
            doc_ids=doc_ids,
            block_size=self.layout.block_size,
            duplexio_layout=self.layout,
            duplexio_full_graph=self.full_cudagraph_enabled and uniform_frames,
            duplexio_packed_capacity=(self.layout.max_compact_slots + 254) * common.num_reqs,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


def update_duplexio_attention_metadata(
    attn_metadata: object,
    active_text_tokens: list[int],
    live_audio_frames: list[int],
) -> None:
    """Bound packed history by live keys; graph-padding requests contribute zero."""
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
        if not isinstance(metadata, DuplexIOAttentionMetadata):
            continue
        if id(metadata) in seen:
            continue
        seen.add(id(metadata))
        layout = metadata.duplexio_layout
        num_reqs = metadata.block_table.shape[0]
        retained_tokens = active_text_tokens[:num_reqs]
        retained_tokens.extend([0] * (num_reqs - len(retained_tokens)))
        retained_audio_frames = live_audio_frames[:num_reqs]
        retained_audio_frames.extend(
            [0] * (num_reqs - len(retained_audio_frames))
        )
        if not metadata.duplexio_full_graph:
            metadata.duplexio_packed_capacity = sum(
                count + DUPLEXIO_NUM_TEXT_CELLS
                + min(frames, layout.audio_ring_frames) * layout.num_audio_cells + 254
                for count, frames in zip(retained_tokens, retained_audio_frames, strict=True)
            )


class DuplexIOFlexAttentionImpl(FlexAttentionImpl):
    """FlexAttention with bounded, role-aware paged-KV writes."""

    def forward(
        self,
        layer: nn.Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kv_cache: Tensor,
        attn_metadata: DuplexIOAttentionMetadata,
        output: Tensor,
        output_scale: Tensor | None = None,
        output_block_scale: Tensor | None = None,
    ) -> Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("DuplexIO attention does not support output quantization")
        if attn_metadata is None:
            return output.fill_(0)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        doc_ids = attn_metadata.doc_ids
        layout = attn_metadata.duplexio_layout

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
        token_key_active = _decode_uint32(
            key[
                :,
                0,
                self.head_size
                - _CACHE_METADATA_SIZE
                + _K_ACTIVE_OFFSET : self.head_size
                - _CACHE_METADATA_SIZE
                + _K_ACTIVE_OFFSET
                + _METADATA_BYTES,
            ]
        ).bool()
        # Query row i and key row i of an append are the same frame token,
        # so the incoming key metadata doubles as the query audio position.
        token_audio_positions = _decode_uint32(
            key[
                :,
                0,
                self.head_size
                - _CACHE_METADATA_SIZE
                + _K_AUDIO_POS_OFFSET : self.head_size
                - _CACHE_METADATA_SIZE
                + _K_AUDIO_POS_OFFSET
                + _METADATA_BYTES,
            ]
        )
        request_indices = doc_ids
        primary_slots = _physical_slots(
            attn_metadata.block_table,
            request_indices,
            duplexio_audio_write_slots(
                key_positions,
                token_audio_positions,
                token_key_active,
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
        physical, cached_metadata = owned_history_metadata(
            flat_key_cache, attn_metadata.block_table, layout.block_size, layout.max_blocks,
        )

        request_last = attn_metadata.query_start_loc[1:] - 1
        indices, valid, key_owners, packed_positions, packed_audio_positions = packed_history_indices(
            physical, query_epochs[request_last], token_audio_positions[request_last],
            cached_metadata[..., 1], cached_metadata[..., 2],
            cached_metadata[..., 5], cached_metadata[..., 4] != 0,
            layout.max_model_len, layout.audio_window_frames,
            attn_metadata.duplexio_packed_capacity,
        )

        batches = attn_metadata.num_query_batches
        block_mask = history_block_mask(
            request_indices.view(batches, -1),
            (key_positions // DUPLEXIO_NUM_CELLS).view(batches, -1),
            token_audio_positions.view(batches, -1),
            key_owners, packed_positions // DUPLEXIO_NUM_CELLS, packed_audio_positions,
            packed_positions % DUPLEXIO_NUM_CELLS, valid,
            layout.audio_window_frames, (128, 128),
        )
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        dim = self.head_size - _CACHE_METADATA_SIZE
        key_tensor = gather_history(key_cache.reshape(-1, self.num_kv_heads, self.head_size), indices, valid, dim)
        value_tensor = gather_history(value_cache.reshape(-1, self.num_kv_heads, self.head_size), indices, valid, dim)
        # Uniform six-cell requests get independent query tiles. Expanded KV
        # batches share the same packed storage; no keys are copied or padded.
        merged = history_and_self_attention(
            query[:, :, :dim].contiguous().view(batches, -1, self.num_heads, dim).transpose(1, 2),
            key_tensor.expand(batches, -1, -1, -1),
            value_tensor.expand(batches, -1, -1, -1),
            key[:, :, :dim].contiguous().view(batches, -1, self.num_kv_heads, dim).transpose(1, 2),
            value[:, :, :dim].contiguous().view(batches, -1, self.num_kv_heads, dim).transpose(1, 2),
            torch.zeros(indices.shape[0], device=query.device, dtype=torch.float32),
            block_mask=block_mask, scale=self.scale,
        ).transpose(1, 2).reshape(-1, self.num_heads, dim)
        output[:num_actual_tokens, :, :dim].copy_(merged)
        output[:num_actual_tokens, :, dim:].zero_()
        return output


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
    """Retain graph input buffers independently for each six-cell batch size."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.full_graph_metadata: dict[int, GDNAttentionMetadata] = {}

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if vllm_config.scheduler_config.max_num_seqs == 1:
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.UNIFORM_BATCH

    def is_full_graph_frame(
        self,
        metadata: CommonAttentionMetadata,
    ) -> bool:
        return (
            metadata.max_query_len == DUPLEXIO_NUM_CELLS
            and metadata.num_actual_tokens == DUPLEXIO_NUM_CELLS * metadata.num_reqs
        )

    def refresh_full_graph_metadata(
        self,
        common: CommonAttentionMetadata,
    ) -> GDNAttentionMetadata:
        metadata = self.full_graph_metadata[common.num_reqs]
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
        self.full_graph_metadata[state_indices.shape[0]] = retained
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
            common_attn_metadata.num_reqs in self.full_graph_metadata
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
    """GDN metadata backend for complete six-cell appends."""

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
        self.q_norm = DuplexIORMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DuplexIORMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        cos_sin_cache: Tensor,
        logical_positions: Tensor,
        hidden_states: Tensor,
        key_active: Tensor,
        request_epochs: Tensor,
        text_ordinals: Tensor,
        audio_positions: Tensor,
    ) -> Tensor:
        qkv = fixed_linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
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
        )
        key = self.k_norm(
            key.view(-1, self.num_kv_heads, self.head_dim)
        )
        query, key = call_compiled_function(
            cached_rotary_pos_emb, query, key, positions, cos_sin_cache
        )

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
                _encode_uint32(audio_positions, key.dtype),
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
            attended = call_compiled_function(gated_attention_output, attended, gate)
        output = fixed_linear(attended, self.o_proj.weight)
        if get_tensor_model_parallel_world_size() > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


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
        norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        norm.weight = self.norm.weight
        self.norm = norm
        self.norm.compile(dynamic=True, fullgraph=True)
        assert self.activation == "silu"
        self.full_cudagraph_enabled = (
            vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
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

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return gdn_cache_dtypes(self.model_config.dtype)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        assert self.num_spec == 0
        return gdn_cache_shapes(
            self.tp_size, self.num_k_heads, self.num_v_heads,
            self.head_k_dim, self.head_v_dim, self.conv_kernel_size,
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
        has_initial_state: Tensor,
        chunk_indices: Tensor,
    ) -> Tensor:
        return stream_causal_conv(
            mixed_qkv, self.conv1d.weight, self.conv1d.bias, conv_state,
            state_indices, query_start_loc, has_initial_state, chunk_indices,
        )

    def _forward_core(
        self,
        mixed_qkv: Tensor,
        b: Tensor,
        a: Tensor,
        core_attn_out: Tensor,
    ) -> None:
        """Whole six-cell appends always use vLLM's prefill metadata."""
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            # vLLM profiles before allocating request caches. Exercise the real
            # append with temporary state so its workspace is included as well.
            tokens = mixed_qkv.shape[0]
            boundaries = torch.tensor((0, tokens), device=mixed_qkv.device, dtype=torch.int32)
            state_indices = torch.zeros(1, device=mixed_qkv.device, dtype=torch.int32)
            has_initial_state = torch.zeros(1, device=mixed_qkv.device, dtype=torch.bool)
            blocks = torch.arange((tokens + 63) // 64, device=mixed_qkv.device, dtype=torch.int32)
            chunk_indices = torch.stack((torch.zeros_like(blocks), blocks), 1)
            cache = tuple(torch.empty((1, *shape), device=mixed_qkv.device, dtype=dtype)
                          for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype(), strict=True))
        else:
            assert isinstance(metadata, dict)
            metadata = metadata[self.prefix]
            assert isinstance(metadata, GDNAttentionMetadata)
            assert metadata.spec_sequence_masks is None
            assert metadata.num_decodes == 0
            state_indices = metadata.prefill_state_indices
            has_initial_state = metadata.prefill_has_initial_state
            boundaries = metadata.prefill_query_start_loc
            chunk_indices = metadata.chunk_indices
            tokens = metadata.num_actual_tokens
            cache = self.kv_cache
        assert state_indices is not None and has_initial_state is not None
        assert boundaries is not None and chunk_indices is not None
        conv_state, recurrent_state = cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        mixed_qkv = self.apply_stream_causal_conv(
            mixed_qkv[:tokens], conv_state, state_indices, boundaries,
            has_initial_state, chunk_indices,
        )
        q, k, v, g, beta = prepare_gdn_inputs(
            qkv=mixed_qkv,
            a=a[:tokens],
            b=b[:tokens],
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            key_heads=self.num_k_heads // self.tp_size,
            key_dim=self.head_k_dim,
            value_dim=self.head_v_dim,
        )
        core_attn_out[:tokens] = append_gdn(
            q, k, v, g, beta, recurrent_state, state_indices,
            boundaries, has_initial_state, chunk_indices,
        )

    def forward_with_key_activity(
        self,
        hidden_states: Tensor,
        key_active: Tensor,
    ) -> Tensor:
        num_tokens = hidden_states.shape[0]
        mixed_qkvz = fixed_linear(hidden_states, self.in_proj_qkvz.weight, self.in_proj_qkvz.bias)
        projected_ba = fixed_linear(hidden_states, self.in_proj_ba.weight, self.in_proj_ba.bias)
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
        normalized = call_compiled_function(
            self.norm, core_output.reshape(-1, self.head_v_dim), output_gate.reshape(-1, self.head_v_dim),
        ).view(num_tokens, -1)
        output = fixed_linear(normalized, self.out_proj.weight, self.out_proj.bias)
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


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
        self.mlp = DuplexIOQwenMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = DuplexIORMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = DuplexIORMSNorm(
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
        cos_sin_cache: Tensor,
        logical_positions: Tensor,
        key_active: Tensor,
        request_epochs: Tensor,
        text_ordinals: Tensor,
        audio_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if residual is not None:
            # Training rounds the previous MLP residual sum before the next norm.
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn.forward_with_key_activity(
                hidden_states,
                key_active,
            )
        else:
            hidden_states = self.self_attn(
                positions,
                cos_sin_cache,
                logical_positions,
                hidden_states,
                key_active,
                request_epochs,
                text_ordinals,
                audio_positions,
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
        "audio_positions": 0,
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
        self.rotary_emb = DuplexIORotaryEmbedding(
            int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)),
            config.max_position_embeddings,
            vllm_config.model_config.dtype,
        )
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
            DuplexIORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        audio_positions: Tensor,
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
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                hidden_states=hidden_states,
                residual=residual,
                key_active=key_active,
                logical_positions=logical_positions,
                request_epochs=request_epochs,
                text_ordinals=text_ordinals,
                audio_positions=audio_positions,
            )
        assert residual is not None
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        return self.norm(hidden_states + residual)

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
    "duplexio_audio_write_slots",
    "update_duplexio_attention_metadata",
]
