# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3.5 backbone layers over DuplexIO's six-cell frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Any, cast

import torch
from torch import Tensor, nn
from vllm._custom_ops import reshape_and_cache_flash
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec
from vllm.vllm_flash_attn import flash_attn_varlen_func

from vllm_omni.model_executor.models.duplexio.frame_layout import NUM_CELLS
from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOFrameMetadata,
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.stream_attention import merge_row_attention
from vllm_omni.model_executor.models.duplexio.stream_conv import expand_stream_conv_weight, stream_causal_conv
from vllm_omni.model_executor.models.duplexio.stream_gdn import (
    append_gdn,
    gdn_cache_dtypes,
    gdn_cache_shapes,
    prepare_gdn_inputs,
)


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
        rows = cdiv(max_tokens, NUM_CELLS)
        self.window_keys = layout.audio_window_frames * layout.num_audio_cells
        self.audio_pages = cdiv(self.window_keys + layout.block_size - 1, layout.block_size)
        # Every step rewrites what it reads, so these buffers hold nothing across
        # steps: like the rest of the KV cache, sleep may discard them.
        self.cu_seqlens = torch.arange(rows + 1, dtype=torch.int32, device=device)
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
        self.write_slots = torch.full((rows * NUM_CELLS,), -1, dtype=torch.long, device=device)

    def physical(self, tables: Tensor, logical: Tensor) -> Tensor:
        """Resolve per-row logical slots through the rows' page tables, -1 where none."""
        page_size = self.layout.block_size
        valid = logical >= 0
        logical = logical.clamp_min(0)
        pages = tables.gather(1, torch.div(logical, page_size, rounding_mode="floor").long())
        return (pages.long() * page_size + logical % page_size).masked_fill(~valid, -1)

    def plan(
        self,
        frame: DuplexIOFrameMetadata,
        query_start_loc: Tensor,
        block_table: Tensor,
        tokens: int,
    ) -> None:
        layout = self.layout
        rows = tokens // NUM_CELLS
        device = block_table.device
        torch.arange(rows + 1, dtype=torch.int32, device=device, out=self.cu_seqlens[: rows + 1])
        row_starts = self.cu_seqlens[:rows] * NUM_CELLS
        request = torch.searchsorted(query_start_loc, row_starts, right=True, out_int32=True) - 1
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
        pages = torch.remainder(
            first_page[:, None] + torch.arange(self.audio_pages, device=device), layout.audio_ring_pages
        )
        torch.gather(tables, 1, pages, out=self.audio_table[:rows])
        self.persistent_seqused[:rows].copy_(persistent.clamp_min(1))
        extra = start[:, None] + torch.arange(layout.num_audio_cells, dtype=torch.int32, device=device)
        self.extra_slots[: rows * layout.num_audio_cells].copy_(
            self.physical(tables, torch.remainder(extra, layout.persistent_base))
            .masked_fill(extra >= windowed[:, None], -1)
            .flatten()
        )
        self.write_slots[:tokens].copy_(
            self.physical(tables, frame.write_slots(tokens, layout).view(rows, NUM_CELLS)).flatten()
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
        # vLLM settles a hybrid model's block size only after building it, so
        # the cache spec, not the model, owns the layout.
        self.reads = DuplexIORowReads(
            kv_cache_spec.layout,
            vllm_config.scheduler_config.max_num_batched_tokens,
            device,
        )

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

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> DuplexIOAttentionMetadata:
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
        rows = tokens // NUM_CELLS
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
            key,
            value,
            keys,
            values,
            reads.write_slots[:tokens],
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        # (tokens, heads) -> (rows, kv_heads * cells * groups): a KV head's
        # grouped query heads of all six cells are consecutive.
        packed = (
            query.view(rows, NUM_CELLS, self.num_kv_heads, -1, self.head_size)
            .transpose(1, 2)
            .reshape(rows, -1, self.head_size)
        )

        def attend(
            seqused: Tensor,
            block_table: Tensor,
            max_keys: int,
            window: list[int] | None,
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
            reads.tables[:rows, layout.audio_ring_pages :],
            layout.max_persistent_keys,
            None,
        )
        output[:tokens].copy_(
            merge_row_attention(
                query,
                key,
                value,
                keys.flatten(0, 1),
                values.flatten(0, 1),
                reads.extra_slots[: rows * layout.num_audio_cells],
                audio,
                audio_lse,
                persistent,
                persistent_lse,
                reads.empty[: rows * 2],
                self.scale,
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
        return metadata.max_query_len == NUM_CELLS and metadata.num_actual_tokens == NUM_CELLS * metadata.num_reqs

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
        if common_attn_metadata.num_reqs in self.full_graph_metadata and self.is_full_graph_frame(common_attn_metadata):
            return self.refresh_full_graph_metadata(common_attn_metadata)
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        token_boundaries = common_attn_metadata.query_start_loc_cpu.tolist()
        chunk_counts = [
            ((end - start) // NUM_CELLS + 63) // 64 for start, end in zip(token_boundaries, token_boundaries[1:])
        ]
        chunks = [(request, chunk) for request, count in enumerate(chunk_counts) for chunk in range(count)]
        metadata.chunk_indices = torch.tensor(
            chunks,
            dtype=torch.int32,
            device=common_attn_metadata.query_start_loc.device,
        )
        metadata.chunk_offsets = torch.tensor(
            list(accumulate(chunk_counts, initial=0)),
            dtype=torch.int32,
            device=common_attn_metadata.query_start_loc.device,
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
    """Attention layer that declares DuplexIO's cache layout; its impl addresses slots from ``frame``."""

    def __init__(self, *args: Any, frame: DuplexIOFrameMetadata, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.frame = frame

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        base = super().get_kv_cache_spec(vllm_config)
        assert isinstance(base, FullAttentionSpec)
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
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Qwen3.5 attention is gated and bias-free: queries carry their gates.
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * 2,
            self.total_num_kv_heads,
            bias=False,
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
            frame=frame,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, key, value = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        query, gate = torch.chunk(q_gate.view(-1, self.num_heads, 2 * self.head_dim), 2, dim=-1)
        query = query.reshape(-1, self.q_size)
        gate = gate.reshape(-1, self.q_size)

        query = self.q_norm(query.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.kv_size)
        query, key = self.rotary_emb(positions, query, key)
        attended = self.attn(query, key, value)
        output, _ = self.o_proj(attended * gate.sigmoid())
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
        assert self.activation == "silu"
        original_weight = cast(Tensor, self.conv1d.weight)
        original_loader = cast(
            Callable[[Tensor, Tensor], None],
            cast(Any, original_weight).weight_loader,
        )
        expanded_weight = nn.Parameter(expand_stream_conv_weight(original_weight.detach()))

        def weight_loader(param: Tensor, loaded_weight: Tensor) -> None:
            original = param.new_empty((*param.shape[:-1], self.conv_kernel_size))
            original_loader(original, loaded_weight)
            param.data.copy_(expand_stream_conv_weight(original))

        set_weight_attrs(expanded_weight, {"weight_loader": weight_loader})
        self.conv1d.weight = expanded_weight

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return gdn_cache_dtypes(self.model_config.dtype)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        assert self.num_spec == 0
        return gdn_cache_shapes(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        # Chunked by frames in every mode: the kernels read chunk indices against frame boundaries.
        return DuplexIOGDNAttentionBackend

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
            mixed_qkv,
            self.conv1d.weight,
            self.conv1d.bias,
            conv_state,
            state_indices,
            query_start_loc,
            has_initial_state,
            chunk_indices,
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
            blocks = torch.arange((tokens // NUM_CELLS + 63) // 64, device=mixed_qkv.device, dtype=torch.int32)
            chunk_indices = torch.stack((torch.zeros_like(blocks), blocks), 1)
            cache = tuple(
                torch.empty((1, *shape), device=mixed_qkv.device, dtype=dtype)
                for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype(), strict=True)
            )
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
            mixed_qkv[:tokens],
            conv_state,
            state_indices,
            boundaries,
            has_initial_state,
            chunk_indices,
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
            q,
            k,
            v,
            g,
            beta,
            recurrent_state,
            state_indices,
            boundaries,
            has_initial_state,
            chunk_indices,
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
        # Inactive cells must not touch the recurrent state: beta = sigmoid(-inf)
        # and the softplus-derived decay of -inf are both exactly zero.
        inactive = ~key_active.unsqueeze(-1)
        beta_logits = beta_logits.masked_fill(inactive, -torch.inf)
        decay_logits = decay_logits.masked_fill(inactive, -torch.inf)
        core_output = hidden_states.new_zeros(
            num_tokens,
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
        )
        gdn_attention_core(mixed_qkv, beta_logits, decay_logits, core_output, _encode_layer_name(self.prefix))
        normalized = self.norm(
            core_output.reshape(-1, self.head_v_dim),
            output_gate.reshape(-1, self.head_v_dim),
        ).view(num_tokens, -1)
        output, _ = self.out_proj(normalized)
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
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
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
        self.mlp = Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None,
        positions: Tensor,
        key_active: Tensor,
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
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "positions": 0,
        "key_active": 0,
        "inputs_embeds": 0,
    }
)
class DuplexIOQwenModel(nn.Module):
    """Inference-only dense Qwen3.5 backbone of DuplexIO."""

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
        # One frame of cell addressing, shared by every full-attention layer and
        # filled by the runner before each step.
        self.frame = DuplexIOFrameMetadata(
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

        self.layers = nn.ModuleList(get_layer(f"{prefix}.layers.{index}") for index in range(config.num_hidden_layers))
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        positions: Tensor,
        key_active: Tensor,
        inputs_embeds: Tensor,
    ) -> Tensor:
        hidden_states, residual = inputs_embeds, None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                key_active=key_active,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

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
