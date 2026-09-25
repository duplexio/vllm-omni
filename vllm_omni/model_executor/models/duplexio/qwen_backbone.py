# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3.5 backbone layers with DuplexIO's six-cell semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import islice
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
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
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
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
from vllm_omni.model_executor.models.duplexio.numerics import call_compiled_function
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    expand_stream_conv_weight,
    mask_inactive_gdn_gates,
)
from vllm_omni.model_executor.models.duplexio.stream_attention import (
    cached_rotary_pos_emb,
    gated_attention_output,
    merge_row_attention,
)
from vllm_omni.model_executor.models.duplexio.stream_conv import stream_causal_conv
from vllm_omni.model_executor.models.duplexio.stream_gdn import (
    append_gdn,
    gdn_cache_dtypes,
    gdn_cache_shapes,
    prepare_gdn_inputs,
)

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
    """Use compute-dtype scales with FP32 normalization arithmetic."""

    def __init__(self, hidden_size: int, eps: float, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    @torch.compile(dynamic=True, fullgraph=True, options={"triton.cudagraphs": False})
    def forward(
        self, hidden: Tensor, residual: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        combined = hidden.float()
        if residual is not None:
            combined = combined + residual.float()
        output = F.rms_norm(combined, (combined.shape[-1],), self.weight.float(), self.eps).to(hidden.dtype)
        if residual is not None:
            return output, combined.to(residual.dtype)
        return output


@torch.compile(dynamic=True, fullgraph=True, options={"triton.cudagraphs": False})
def swiglu_mlp(hidden: Tensor, gate_up_weight: Tensor, down_weight: Tensor) -> Tensor:
    """Compute SwiGLU with packed projections and standard compiler optimizations."""
    gate, up = F.linear(hidden, gate_up_weight).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_weight)


class DuplexIOQwenMLP(Qwen3NextMLP):
    """Keep vLLM's sharded weights with compiled standard SwiGLU operations."""

    def forward(self, hidden: Tensor) -> Tensor:
        output = swiglu_mlp(
            hidden,
            self.gate_up_proj.weight,
            self.down_proj.weight,
        )
        if self.down_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class DuplexIORowReads:
    """Per-row FlashAttention tables for one step, at stable device addresses.

    Each row is one FlashAttention sequence of a single query position whose
    heads are the row's six cells times the grouped query heads, so a KV page is
    loaded once for all of them. A row reads two ranges (see
    ``DuplexIOFrameMetadata.row_reads``): its audio window, as a table of ring
    pages rotated to start at the window's first page, and the persistent
    region. The audio window has a fixed width, so a frozen row's one extra
    frame is read by slot and merged with the row's own cell.
    """

    def __init__(self, layout: DuplexIOKVLayout, max_tokens: int, device: torch.device) -> None:
        self.layout = layout
        rows = cdiv(max_tokens, DUPLEXIO_NUM_CELLS)
        self.window_keys = layout.audio_window_frames * layout.num_audio_cells
        self.audio_pages = cdiv(self.window_keys + layout.block_size - 1, layout.block_size)
        self.cu_seqlens = torch.arange(rows + 1, dtype=torch.int32, device=device)
        self.row_starts = self.cu_seqlens[:rows] * DUPLEXIO_NUM_CELLS
        self.page_ids = torch.arange(self.audio_pages, device=device)
        self.extra_ids = torch.arange(layout.num_audio_cells, dtype=torch.int32, device=device)
        self.tables = torch.zeros(rows, layout.max_blocks, dtype=torch.int32, device=device)
        self.audio_table = torch.zeros(rows, self.audio_pages, dtype=torch.int32, device=device)
        self.audio_seqused = torch.zeros(rows, dtype=torch.int32, device=device)
        self.persistent_seqused = torch.zeros(rows, dtype=torch.int32, device=device)
        # Per row, whether its audio and its persistent read are empty. An
        # empty read still hands FlashAttention one key, whose partial the
        # merge then drops: FA2's grouped decode path writes an empty
        # sequence's normalizer over other rows' normalizers.
        self.empty = torch.ones(rows * 2, dtype=torch.bool, device=device)
        self.extra_slots = torch.full((rows * layout.num_audio_cells,), -1, dtype=torch.long, device=device)
        self.write_slots = torch.full((rows * DUPLEXIO_NUM_CELLS,), -1, dtype=torch.long, device=device)

    def physical(self, tables: Tensor, logical: Tensor) -> Tensor:
        """Resolve per-row logical slots through the rows' page tables, -1 where none."""
        page_size = self.layout.block_size
        valid = logical >= 0
        logical = logical.clamp_min(0)
        pages = tables.gather(1, torch.div(logical, page_size, rounding_mode="floor").long())
        return (pages.long() * page_size + logical % page_size).masked_fill(~valid, -1)

    def plan(
        self, frame: DuplexIOFrameMetadata, query_start_loc: Tensor, block_table: Tensor, tokens: int,
    ) -> None:
        layout = self.layout
        rows = tokens // DUPLEXIO_NUM_CELLS
        request = torch.searchsorted(query_start_loc, self.row_starts[:rows], right=True, out_int32=True) - 1
        tables = self.tables[:rows]
        torch.index_select(block_table, 0, request.clamp_(0, block_table.shape[0] - 1), out=tables)
        start, end, persistent = frame.row_reads(tokens)
        # The window covers the last `window_keys` keys; whatever of the range
        # precedes it is at most one frame.
        windowed = torch.maximum(end - self.window_keys, start)
        first_page = torch.div(windowed, layout.block_size, rounding_mode="floor")
        audio = torch.where(windowed < end, end - first_page * layout.block_size, 0)
        torch.eq(torch.stack((audio, persistent), 1), 0, out=self.empty[: rows * 2].view(rows, 2))
        self.audio_seqused[:rows].copy_(audio.clamp_min(1))
        pages = torch.remainder(first_page[:, None] + self.page_ids, layout.audio_ring_pages)
        torch.gather(tables, 1, pages, out=self.audio_table[:rows])
        self.persistent_seqused[:rows].copy_(persistent.clamp_min(1))
        extra = start[:, None] + self.extra_ids
        self.extra_slots[: rows * layout.num_audio_cells].copy_(
            self.physical(tables, torch.remainder(extra, layout.persistent_base))
            .masked_fill(extra >= windowed[:, None], -1)
            .flatten()
        )
        self.write_slots[:tokens].copy_(
            self.physical(tables, frame.write_slots(tokens).view(rows, DUPLEXIO_NUM_CELLS)).flatten()
        )


@dataclass
class DuplexIOAttentionMetadata:
    """The step's request layout; row tables are planned by the first layer."""

    num_actual_tokens: int
    query_start_loc: Tensor
    block_table: Tensor
    reads: DuplexIORowReads
    reads_planned: bool = False


class DuplexIOFlashAttentionMetadataBuilder(AttentionMetadataBuilder[DuplexIOAttentionMetadata]):
    """Hand the step's request layout to the impl.

    Row tables need the step's frame, which the runner installs after metadata
    is built, so they are planned at forward time, inside the CUDA graph when
    one is captured.
    """

    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: DuplexIOKVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if kv_cache_spec.sliding_window is not None:
            raise ValueError("DuplexIO applies its own audio window, not a sliding window")
        layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
        frame = next(iter(layers.values())).frame
        if frame.layout != kv_cache_spec.layout:
            raise ValueError(
                f"DuplexIO cache layout {kv_cache_spec.layout} does not match the layout "
                f"{frame.layout} the model was built with"
            )
        self.reads = DuplexIORowReads(frame.layout, frame.cell.shape[0], device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DuplexIOAttentionMetadata:
        if common_prefix_len:
            raise NotImplementedError("DuplexIO attention does not support cascade prefixes")
        common = common_attn_metadata
        return DuplexIOAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            query_start_loc=common.query_start_loc,
            block_table=common.block_table_tensor,
            reads=self.reads,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> DuplexIOAttentionMetadata:
        return self.build(0, common_attn_metadata)


class DuplexIOFlashAttentionImpl(FlashAttentionImpl):
    """Paged FlashAttention over role-addressed slots, plus the query diagonal."""

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
        tokens = attn_metadata.num_actual_tokens
        rows = tokens // DUPLEXIO_NUM_CELLS
        query, key, value = query[:tokens], key[:tokens], value[:tokens]
        reads = attn_metadata.reads
        layout = reads.layout
        # Every layer shares the step's metadata, so the first to run plans
        # the rows for all of them.
        if not attn_metadata.reads_planned:
            reads.plan(layer.frame, attn_metadata.query_start_loc, attn_metadata.block_table, tokens)
            attn_metadata.reads_planned = True
        keys, values = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key, value, keys, values, reads.write_slots[:tokens],
            self.kv_cache_dtype, layer._k_scale, layer._v_scale,
        )
        # (tokens, heads) -> (rows, kv_heads * cells * groups): a KV head's
        # grouped query heads of all six cells are consecutive.
        packed = (
            query.view(rows, DUPLEXIO_NUM_CELLS, self.num_kv_heads, -1, self.head_size)
            .transpose(1, 2)
            .reshape(rows, -1, self.head_size)
        )

        def attend(
            seqused: Tensor, block_table: Tensor, max_keys: int, window: list[int] | None,
        ) -> tuple[Tensor, Tensor]:
            return flash_attn_varlen_func(
                q=packed,
                k=keys,
                v=values,
                max_seqlen_q=1,
                cu_seqlens_q=reads.cu_seqlens[: rows + 1],
                max_seqlen_k=max_keys,
                seqused_k=seqused,
                softmax_scale=self.scale,
                causal=True,
                window_size=window,
                block_table=block_table,
                return_softmax_lse=True,
                fa_version=self.vllm_flash_attn_version,
            )

        audio, audio_lse = attend(
            reads.audio_seqused[:rows],
            reads.audio_table[:rows],
            reads.audio_pages * layout.block_size,
            [reads.window_keys - 1, 0],
        )
        persistent, persistent_lse = attend(
            reads.persistent_seqused[:rows],
            reads.tables[:rows, layout.audio_ring_pages:],
            layout.max_persistent_keys,
            None,
        )
        output[:tokens].copy_(
            merge_row_attention(
                query, key, value,
                keys.flatten(0, 1), values.flatten(0, 1), reads.extra_slots[: rows * layout.num_audio_cells],
                audio, audio_lse, persistent, persistent_lse, reads.empty[: rows * 2], self.scale,
            )
        )
        return output


class DuplexIOFlashAttentionBackend(FlashAttentionBackend):
    """Model-selected FlashAttention backend for DuplexIO full-attention layers."""

    forward_includes_kv_cache_update = True

    @staticmethod
    def get_impl_cls() -> type[DuplexIOFlashAttentionImpl]:
        return DuplexIOFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[DuplexIOFlashAttentionMetadataBuilder]:
        return DuplexIOFlashAttentionMetadataBuilder


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
            attn_backend=DuplexIOFlashAttentionBackend,
        )
        # The impl addresses cache slots from the frame.
        self.attn.frame = frame
        self.q_norm = DuplexIORMSNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=vllm_config.model_config.dtype,
        )
        self.k_norm = DuplexIORMSNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=vllm_config.model_config.dtype,
        )

    def forward(
        self,
        positions: Tensor,
        cos_sin_cache: Tensor,
        hidden_states: Tensor,
    ) -> Tensor:
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
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
        output = F.linear(attended, self.o_proj.weight)
        if get_tensor_model_parallel_world_size() > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


@eager_break_during_capture
def gdn_attention_core(mixed_qkv: Tensor, b: Tensor, a: Tensor, output: Tensor, layer_name: str) -> None:
    """Variable-length GDN appends run eagerly between piecewise graph segments."""
    torch.ops.vllm.qwen_gdn_attention_core(mixed_qkv, b, a, output, layer_name=layer_name)


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
        mixed_qkvz = F.linear(hidden_states, self.in_proj_qkvz.weight, self.in_proj_qkvz.bias)
        projected_ba = F.linear(hidden_states, self.in_proj_ba.weight, self.in_proj_ba.bias)
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
        gdn_attention_core(mixed_qkv, beta_logits, decay_logits, core_output, _encode_layer_name(self.prefix))
        normalized = call_compiled_function(
            self.norm, core_output.reshape(-1, self.head_v_dim), output_gate.reshape(-1, self.head_v_dim),
        ).view(num_tokens, -1)
        output = F.linear(normalized, self.out_proj.weight, self.out_proj.bias)
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
            dtype=vllm_config.model_config.dtype,
        )
        self.post_attention_layernorm = DuplexIORMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=vllm_config.model_config.dtype,
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
            DuplexIORMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dtype=vllm_config.model_config.dtype,
            )
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
    "DuplexIOFlashAttentionBackend",
    "DuplexIOFlashAttentionMetadataBuilder",
    "DuplexIOGDNAttentionBackend",
    "DuplexIOGDNAttentionMetadataBuilder",
    "DuplexIOPagedAttention",
    "DuplexIOQwenAttention",
    "DuplexIOQwenDecoderLayer",
    "DuplexIOQwenGatedDeltaNetAttention",
    "DuplexIOQwenModel",
]
