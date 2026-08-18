# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3.5 backbone layers with DuplexIO's six-cell semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, cast

import torch
from torch import Tensor, nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model, Qwen3_5RMSNorm
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionBackend,
    FlexAttentionImpl,
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
    physical_to_logical_mapping,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_omni.model_executor.models.duplexio.attention_semantics import (
    key_visible,
)
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
_K_AUDIO_POS_OFFSET = _K_TEXT_ORDINAL_OFFSET + _METADATA_BYTES
_CACHE_METADATA_SIZE = _K_AUDIO_POS_OFFSET + _METADATA_BYTES


def _encode_uint32(values: Tensor, dtype: torch.dtype) -> Tensor:
    shifts = values.new_tensor((0, 8, 16, 24))
    return torch.bitwise_and(values.unsqueeze(-1) >> shifts, 0xFF).to(dtype)


def _decode_uint32(values: Tensor) -> Tensor:
    shifts = values.new_tensor((0, 8, 16, 24), dtype=torch.long)
    return torch.sum(values.to(torch.long) << shifts, dim=-1)


def duplexio_compact_key_visible(
    query_positions: Tensor,
    compact_key_positions: Tensor,
    query_epochs: Tensor,
    key_epochs: Tensor,
    key_positions: Tensor,
    text_ordinals: Tensor,
    query_audio_positions: Tensor,
    key_audio_positions: Tensor,
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

    # The compact layout stores only active keys: inactive text is confined to
    # the self-visible transient region, and every cached audio key carries
    # real audio (the protocol appends one PCM frame per step).
    cross_frame_visible = key_visible(
        query_frames,
        key_frames,
        query_audio_positions,
        key_audio_positions,
        key_cells,
        torch.ones_like(key_cells, dtype=torch.bool),
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
        & ((query_positions == key_positions) | cross_frame_visible)
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
        & (
            compact_key_positions
            == layout.persistent_text_base + text_ordinals - 1
        )
        & cross_frame_visible
    )
    return same_request & (
        audio_visible | transient_visible | persistent_visible
    )


def duplexio_primary_compact_slots(
    positions: Tensor,
    audio_positions: Tensor,
    layout: DuplexIOKVLayout,
) -> Tensor:
    """Map the six current frame cells to audio-ring or transient slots.

    The audio ring is keyed by audio position, mirroring the training-side
    eviction bound: window positions never decrease, so a ring of
    ``audio_window_frames + 1`` distinct audio positions retains exactly the
    keys still visible to any future query.
    """
    cells = torch.remainder(positions, DUPLEXIO_NUM_CELLS)
    audio_slots = (
        torch.remainder(audio_positions, layout.audio_ring_frames)
        * layout.num_audio_cells
        + cells
        - DUPLEXIO_NUM_TEXT_CELLS
    )
    return torch.where(
        cells < DUPLEXIO_NUM_TEXT_CELLS,
        layout.transient_text_base + cells,
        audio_slots,
    )


def _physical_slots(
    block_table: Tensor,
    request_indices: Tensor,
    compact_slots: Tensor,
    block_size: int,
) -> Tensor:
    compact_blocks = torch.div(
        compact_slots,
        block_size,
        rounding_mode="floor",
    )
    physical_blocks = block_table[request_indices, compact_blocks].to(torch.long)
    return physical_blocks * block_size + torch.remainder(
        compact_slots,
        block_size,
    )


class DuplexIOFlexAttentionMetadataBuilder(FlexAttentionMetadataBuilder):
    """Build Flex metadata over DuplexIO's compact per-request address space."""

    _cudagraph_support = AttentionCGSupport.NEVER

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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
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
        setattr(metadata, "duplexio_layout", self.layout)
        return metadata


def update_duplexio_attention_metadata(
    attn_metadata: object,
    active_text_tokens: list[int],
) -> None:
    """Limit Flex's gathered pages to the retained prefix plus frame headroom."""
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
        metadata.block_mask = None


class DuplexIOFlexAttentionImpl(FlexAttentionImpl):
    """FlexAttention with bounded, role-aware paged-KV writes."""

    def __init__(
        self,
        *args: Any,
        audio_attention_window_frames: int,
        max_model_len: int,
        cache_block_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layout = DuplexIOKVLayout(
            block_size=cache_block_size,
            audio_window_frames=audio_attention_window_frames,
            max_model_len=max_model_len,
        )

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
            # Query row i and key row i of an append are the same frame token,
            # so the incoming key metadata doubles as the query audio position.
            token_audio_positions = _decode_uint32(
                key[
                    :,
                    0,
                    self.head_size
                    - _CACHE_METADATA_SIZE
                    + _K_AUDIO_POS_OFFSET :,
                ]
            )
            request_indices = doc_ids[:num_actual_tokens].to(torch.long)
            primary_slots = _physical_slots(
                attn_metadata.block_table,
                request_indices,
                duplexio_primary_compact_slots(
                    key_positions,
                    token_audio_positions,
                    self.layout,
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
                self.layout.persistent_text_base
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

            key_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)[0]
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
            cached_audio_positions = _decode_uint32(
                flat_key_cache[
                    :,
                    0,
                    metadata_base
                    + _K_AUDIO_POS_OFFSET : metadata_base
                    + _K_AUDIO_POS_OFFSET
                    + _METADATA_BYTES,
                ]
            )

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
                    token_audio_positions[query_index],
                    cached_audio_positions[physical_key_index],
                    self.layout,
                )
                return is_valid & visible

            attn_metadata.mask_mod = mask_mod
            attn_metadata.block_mask = None

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
            audio_attention_window_frames=(
                vllm_config.model_config.hf_config.audio_attention_window_frames
            ),
            max_model_len=vllm_config.model_config.max_model_len,
            cache_block_size=cache_config.block_size,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        logical_positions: Tensor,
        hidden_states: Tensor,
        request_epochs: Tensor,
        text_ordinals: Tensor,
        audio_positions: Tensor,
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
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
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
        audio_positions: Tensor,
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


class DuplexIOQwenModel(nn.Module):
    """Inference-only dense Qwen3.5 model used by native DuplexIO."""

    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

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
            Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
                hidden_states=hidden_states,
                residual=residual,
                key_active=key_active,
                logical_positions=logical_positions,
                request_epochs=request_epochs,
                text_ordinals=text_ordinals,
                audio_positions=audio_positions,
            )
        if not get_pp_group().is_last_rank:
            assert residual is not None
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )


__all__ = [
    "DuplexIOFlexAttentionBackend",
    "DuplexIOFlexAttentionMetadataBuilder",
    "DuplexIOPagedAttention",
    "DuplexIOQwenAttention",
    "DuplexIOQwenDecoderLayer",
    "DuplexIOQwenGatedDeltaNetAttention",
    "DuplexIOQwenModel",
    "duplexio_compact_key_visible",
    "duplexio_primary_compact_slots",
]
