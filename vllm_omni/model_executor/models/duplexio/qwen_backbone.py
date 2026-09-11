# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3.5 backbone layers with DuplexIO's six-cell semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from itertools import islice
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_layers_from_vllm_config
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
from vllm.utils.math_utils import cdiv
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
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOFrameMetadata,
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
    merge_self_attention,
    paged_history_attention,
)
from vllm_omni.model_executor.models.duplexio.stream_conv import stream_causal_conv
from vllm_omni.model_executor.models.duplexio.stream_gdn import (
    append_gdn,
    gdn_cache_dtypes,
    gdn_cache_shapes,
    prepare_gdn_inputs,
)

# Flex tiles the 1024-key cache pages; 16x64 keeps the six-cell query tile and
# the masked-out ring pages cheap.
KERNEL_OPTIONS: dict[str, int | bool] = {
    "FORCE_USE_FLEX_ATTENTION": True,
    "BLOCK_M": 16,
    "BLOCK_N": 64,
}


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
def step_page_bounds(
    seq_lens: Tensor,
    block_table: Tensor,
    page_ids: Tensor,
    compact_seq_lens: Tensor,
    touched_block_table: Tensor,
    layout: DuplexIOKVLayout,
) -> None:
    """Write this step's compact sequence bound and its page table.

    A frame owns four text slots at most, and only emitted cells claim one, so
    the bound over-scans into masked-out pages that the emitted count - unknown
    until preprocessing runs - would tighten.

    Pages the step can neither read nor write are blanked, which reads as "skip
    me" wherever a page list is built: a session's audio ring starts at the
    bottom of the compact space and its text region at a fixed base far above
    it, so scanning up to the bound alone would cover a wide band of pages that
    are never written. Both outputs keep their address for CUDA-graph replay.
    """
    rows = torch.div(seq_lens, DUPLEXIO_NUM_CELLS, rounding_mode="floor")
    text_slots = rows * DUPLEXIO_NUM_TEXT_CELLS
    audio_slots = rows.clamp(max=layout.audio_ring_frames) * layout.num_audio_cells
    text_pages = cdiv(text_slots, layout.block_size)[:, None]
    base = layout.text_base_page
    touched = (page_ids < cdiv(audio_slots, layout.block_size)[:, None]) | (
        (page_ids >= base) & (page_ids - base < text_pages)
    )
    compact_seq_lens.copy_(
        (layout.persistent_text_base + text_slots).clamp(max=layout.max_compact_slots)
    )
    touched_block_table.copy_(block_table * touched)


@torch.compile(dynamic=True, fullgraph=True)
def physical_slots(
    block_table: Tensor,
    request_indices: Tensor,
    compact_slots: Tensor,
    block_size: int,
) -> Tensor:
    """Resolve compact slots through the page table, -1 where there is no slot.

    The table is the touched-page table the mask was built from: a step writes
    the ring position and the text ordinals it also reads, so its pages are
    listed there.
    """
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
    """Point upstream's paged FlexAttention at DuplexIO's compact slot space.

    Slots are addressed by role - an audio ring, then a persistent text region -
    so the only backend-visible differences are the sequence bound (how many
    pages a step scans) and the mask (row causality plus the audio window,
    instead of token causality). Persistent index buffers, the direct block-mask
    build and CUDA-graph support are all upstream's.
    """

    def __init__(
        self,
        kv_cache_spec: DuplexIOKVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if not self.direct_build:
            raise ValueError(
                "DuplexIO needs FlexAttention's direct block-mask build, which "
                f"requires the kv block size ({self.kv_block_size}) to equal the "
                f"cache block size ({self.block_size})"
            )
        self.layout = kv_cache_spec.layout
        layers = get_layers_from_vllm_config(vllm_config, Attention, self.layer_names)
        frame = next(iter(layers.values())).frame
        if frame.layout != self.layout:
            raise ValueError(
                f"DuplexIO cache layout {self.layout} does not match the layout "
                f"{frame.layout} the model was built with"
            )
        # Every full-attention layer shares one frame, so one mask serves them all.
        self.duplexio_mask = self._maybe_get_custom_mask_mod(layers)
        self.max_num_kv_indices = self.q_block_size * self.layout.max_blocks
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.compact_seq_lens = torch.empty(max_num_seqs, dtype=torch.int32, device=device)
        # Written every step and read inside CUDA graphs, so the address is kept.
        self.touched_block_table = torch.empty(
            max_num_seqs,
            self.layout.max_blocks,
            dtype=torch.int32,
            device=device,
        )
        self.page_ids = torch.arange(self.layout.max_blocks, dtype=torch.int32, device=device)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlexAttentionMetadata:
        # The compact bound is a constant, so capture needs no sequence maximum
        # and never syncs on one.
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlexAttentionMetadata:
        common = common_attn_metadata
        layout = self.layout
        requests = common.seq_lens.shape[0]
        seq_lens = self.compact_seq_lens[:requests]
        block_table = self.touched_block_table[:requests]
        step_page_bounds(
            common.seq_lens,
            common.block_table_tensor,
            self.page_ids,
            seq_lens,
            block_table,
            layout,
        )
        metadata = super().build(
            common_prefix_len,
            common.replace(
                causal=False,
                seq_lens=seq_lens,
                max_seq_len=layout.max_compact_slots,
                block_table_tensor=block_table,
            ),
            fast_build,
        )
        # The paged mask closure reads both at kernel launch, so the block mask
        # super().build() already produced picks them up. Starting every query's
        # logical index at its batch offset makes it index this step's cells.
        metadata.logical_mask_mod = self.duplexio_mask
        metadata.decode_offset.copy_(common.query_start_loc[: seq_lens.shape[0]])
        return metadata


class DuplexIOFlexAttentionImpl(FlexAttentionImpl):
    """Paged FlexAttention over role-addressed slots, plus the query diagonal."""

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
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("DuplexIO attention does not support output quantization")
        if attn_metadata is None:
            return output.fill_(0)
        tokens = attn_metadata.num_actual_tokens
        query, key, value = query[:tokens], key[:tokens], value[:tokens]
        frame = layer.frame
        assert attn_metadata.doc_ids is not None
        self.do_kv_cache_update(
            layer,
            key,
            value,
            kv_cache,
            physical_slots(
                attn_metadata.block_table,
                attn_metadata.doc_ids[:tokens],
                frame.write_slots(tokens),
                attn_metadata.block_size,
            ),
        )

        # A cell's own K/V is not in the cache yet, so history and diagonal are
        # reduced separately and merged with the softmax normalizer.
        keys, values = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        history, lse = paged_history_attention(
            query,
            keys.view(-1, self.num_kv_heads, self.head_size),
            values.view(-1, self.num_kv_heads, self.head_size),
            attn_metadata.block_mask,
            self.scale,
            self.num_kv_heads != self.num_heads,
            KERNEL_OPTIONS,
        )
        output[:tokens].copy_(
            merge_self_attention(query, key, value, history, lse, self.scale)
        )
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
        frame: DuplexIOFrameMetadata,
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
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
            attn_backend=DuplexIOFlexAttentionBackend,
        )
        # The impl addresses cache slots from the frame, and the metadata builder
        # finds this step's mask on the layer the way upstream Flex expects.
        self.attn.frame = frame
        self.attn.logical_mask_mod = frame.visible
        self.q_norm = DuplexIORMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DuplexIORMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        cos_sin_cache: Tensor,
        hidden_states: Tensor,
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
        attended = self.attn(
            query,
            key,
            value.view(-1, self.num_kv_heads, self.head_dim),
        )
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

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        frame: DuplexIOFrameMetadata,
        prefix: str,
    ) -> None:
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
                frame=frame,
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
        key_active: Tensor,
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
                hidden_states,
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
        "key_active": 0,
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
        # One frame of cell addressing, shared by every full-attention layer and
        # filled by the runner before each step.
        self.frame = DuplexIOFrameMetadata(
            DuplexIOKVLayout(
                block_size=vllm_config.cache_config.block_size,
                audio_window_frames=vllm_config.model_config.hf_config.audio_attention_window_frames,
                max_model_len=vllm_config.model_config.max_model_len,
            ),
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.device_config.device,
        )

        def get_layer(prefix: str) -> DuplexIOQwenDecoderLayer:
            layer_index = extract_layer_index(prefix)
            return DuplexIOQwenDecoderLayer(
                vllm_config,
                config.layer_types[layer_index],
                self.frame,
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
        key_active: Tensor,
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
]
