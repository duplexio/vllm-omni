# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3.5 backbone layers over DuplexIO's flat sequence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from vllm._custom_ops import reshape_and_cache_flash
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
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_POSITIONS
from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.stream_attention import (
    ATTENTION_COLUMNS,
    REQUEST,
    SELF_KEY,
    flat_paged_attention,
)
from vllm_omni.model_executor.models.duplexio.stream_gdn import (
    GDN_COLUMNS,
    GDN_REQUEST_COLUMNS,
    flat_gdn,
    gdn_cache_dtypes,
    gdn_cache_shapes,
    gdn_gates,
)

# Columns of a pass's packed per-position addressing: the RoPE row, the KV
# write (a slot within a region, -1 for none, and the region:
# ``flat_plan.Region``), then attention and GDN metadata.
ROPE, WRITE, WRITE_REGION = 0, 1, 2
ATTENTION = slice(3, 3 + ATTENTION_COLUMNS)
GDN = slice(3 + ATTENTION_COLUMNS, 3 + ATTENTION_COLUMNS + GDN_COLUMNS)
STEP_COLUMNS = 3 + ATTENTION_COLUMNS + GDN_COLUMNS


class FlatStep:
    """One backbone pass's flat addressing, at stable device addresses.

    The model installs a pass's positions before running it; every layer
    reads them. A request's positions in a pass are contiguous and whole
    frames. Uninstalled, the step is inert, for vLLM's warm-up and capture
    runs: no position writes a key or commits a GDN step, and each frame is its
    own request.
    """

    def __init__(self, max_tokens: int, device: torch.device) -> None:
        self.positions = torch.zeros(max_tokens, STEP_COLUMNS, dtype=torch.int32, device=device)
        self.rope = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.requests = torch.zeros(
            max_tokens // FRAME_POSITIONS + 1, GDN_REQUEST_COLUMNS, dtype=torch.int32, device=device
        )
        self.num_requests: int | None = None
        self.reset()

    def install(self, positions: Tensor, requests: Tensor) -> None:
        """``positions`` ``(tokens, STEP_COLUMNS)`` and ``requests`` ``(requests, GDN_REQUEST_COLUMNS)`` int32."""
        tokens = positions.shape[0]
        self.positions[:tokens].copy_(positions)
        self.rope[:tokens].copy_(positions[:, ROPE])
        self.requests[: requests.shape[0]].copy_(requests)
        self.num_requests = requests.shape[0]

    def reset(self) -> None:
        """Make every position inert: a no-op of its own frame's request."""
        tokens = self.positions.shape[0]
        self.positions.zero_()
        self.positions[:, WRITE] = -1
        self.positions[:, ATTENTION.start + SELF_KEY] = 1  # Each sees only its own key.
        self.rope.zero_()
        frames = torch.arange(self.requests.shape[0], device=self.requests.device, dtype=torch.int32)
        self.positions[:, ATTENTION.start + REQUEST] = torch.arange(
            tokens, device=frames.device, dtype=torch.int32
        ).div(FRAME_POSITIONS, rounding_mode="floor")
        self.requests.copy_(
            torch.stack(
                (
                    frames * FRAME_POSITIONS,
                    frames * FRAME_POSITIONS + FRAME_POSITIONS,
                    frames,
                    torch.zeros_like(frames),
                ),
                dim=1,
            )
        )
        self.num_requests = None

    def request_count(self, tokens: int) -> int:
        return tokens // FRAME_POSITIONS if self.num_requests is None else self.num_requests


@dataclass
class DuplexIOAttentionMetadata:
    block_table: Tensor
    layout: DuplexIOKVLayout


class DuplexIOFlashAttentionMetadataBuilder(AttentionMetadataBuilder[DuplexIOAttentionMetadata]):
    """Hand the step's block table to the impl; the model's ``FlatStep`` holds all addressing."""

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
        self.layout = kv_cache_spec.layout

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DuplexIOAttentionMetadata:
        if common_prefix_len:
            raise NotImplementedError("DuplexIO attention does not support cascade prefixes")
        return DuplexIOAttentionMetadata(common_attn_metadata.block_table_tensor, self.layout)

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> DuplexIOAttentionMetadata:
        return self.build(0, common_attn_metadata)


class DuplexIOFlashAttentionImpl(FlashAttentionImpl):
    """Paged attention of flat positions over their request's cache layout."""

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
        tokens = query.shape[0]
        layout = attn_metadata.layout
        step = layer.step.positions[:tokens]
        tables = attn_metadata.block_table
        keys, values = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key = key.view(tokens, self.num_kv_heads, self.head_size)
        value = value.view(tokens, self.num_kv_heads, self.head_size)
        # Region bases: the audio ring, the BOS ring, the persistent keys.
        logical = step[:, WRITE] + torch.where(
            step[:, WRITE_REGION] == 0,
            0,
            torch.where(step[:, WRITE_REGION] == 1, layout.bos_base, layout.persistent_base),
        )
        written = step[:, WRITE] >= 0
        logical = logical.masked_fill(~written, 0)
        request = step[:, ATTENTION.start + REQUEST].clamp(0, tables.shape[0] - 1)
        pages = tables[request.long(), torch.div(logical, layout.block_size, rounding_mode="floor").long()]
        slots = (pages.long() * layout.block_size + logical % layout.block_size).masked_fill(~written, -1)
        reshape_and_cache_flash(key, value, keys, values, slots, self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        attended = flat_paged_attention(
            query.view(tokens, self.num_heads, self.head_size),
            key,
            value,
            keys,
            values,
            tables,
            step[:, ATTENTION],
            scale=self.scale,
            ring_slots=layout.ring_slots,
            bos_base=layout.bos_base,
            persistent_base=layout.persistent_base,
            block_size=layout.block_size,
        )
        output[:tokens].copy_(attended.reshape(output[:tokens].shape))
        return output


class DuplexIOFlashAttentionBackend(FlashAttentionBackend):
    """Model-selected backend for DuplexIO full-attention layers; it writes its own keys."""

    forward_includes_kv_cache_update = True

    @staticmethod
    def get_impl_cls() -> type[DuplexIOFlashAttentionImpl]:
        return DuplexIOFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[DuplexIOFlashAttentionMetadataBuilder]:
        return DuplexIOFlashAttentionMetadataBuilder


@dataclass
class DuplexIOGDNMetadata:
    state_indices: Tensor  # The step's GDN cache slot per request, in batch order.


class DuplexIOGDNAttentionMetadataBuilder(AttentionMetadataBuilder[DuplexIOGDNMetadata]):
    """Each request's GDN cache slot, at a stable address; the model's ``FlatStep`` holds the rest."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.state_indices = torch.full(
            (vllm_config.scheduler_config.max_num_seqs,), -1, dtype=torch.int32, device=device
        )

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec) -> AttentionCGSupport:
        # A graph runs one program per request and stream, so it replays only
        # batches split into requests as at capture: one frame each.
        if vllm_config.scheduler_config.max_num_seqs == 1:
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DuplexIOGDNMetadata:
        common = common_attn_metadata
        indices = mamba_get_block_table_tensor(
            common.block_table_tensor,
            common.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )[:, 0]
        self.state_indices[: indices.shape[0]].copy_(indices)
        self.state_indices[indices.shape[0] :].fill_(-1)
        return DuplexIOGDNMetadata(self.state_indices)

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> DuplexIOGDNMetadata:
        return self.build(0, common_attn_metadata)


class DuplexIOGDNAttentionBackend(GDNAttentionBackend):
    """GDN cache slots for DuplexIO's per-stream recurrence."""

    @staticmethod
    def get_name() -> str:
        return "DUPLEXIO_GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type[DuplexIOGDNAttentionMetadataBuilder]:
        return DuplexIOGDNAttentionMetadataBuilder


class DuplexIOPagedAttention(Attention):
    """Attention layer that declares DuplexIO's cache layout; its impl addresses slots from ``step``."""

    def __init__(self, *args: Any, step: FlatStep, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.step = step

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
        step: FlatStep,
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
            step=step,
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


class DuplexIOQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Qwen GDN whose positions each step their own stream (``stream_gdn.flat_gdn``)."""

    def __init__(
        self,
        config: Any,
        vllm_config: VllmConfig,
        step: FlatStep,
        prefix: str,
    ) -> None:
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=False,
        )
        assert self.activation == "silu"
        self.step = step
        # The fused CUDA decode kernel knows one state per request, not per stream.
        self.enable_fused_gdn_decode = False

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
        return DuplexIOGDNAttentionBackend

    def _forward_core(
        self,
        mixed_qkv: Tensor,
        b: Tensor,
        a: Tensor,
        core_attn_out: Tensor,
    ) -> None:
        tokens = mixed_qkv.shape[0]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            # vLLM profiles before allocating request caches. Exercise the real
            # kernels with a temporary slot so their workspace is included.
            slots = torch.zeros(1, device=mixed_qkv.device, dtype=torch.int32)
            cache = tuple(
                torch.zeros((1, *shape), device=mixed_qkv.device, dtype=dtype)
                for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype(), strict=True)
            )
            requests = torch.tensor([[0, tokens, 0, 1]], device=mixed_qkv.device, dtype=torch.int32)
            num_requests = 1
        else:
            assert isinstance(metadata, dict)
            gdn_metadata = metadata[self.prefix]
            assert isinstance(gdn_metadata, DuplexIOGDNMetadata)
            slots = gdn_metadata.state_indices
            cache = self.kv_cache
            num_requests = self.step.request_count(tokens)
            requests = self.step.requests[:num_requests]
        conv_state, recurrent_state = cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        g, beta = gdn_gates(a[:tokens], b[:tokens], self.A_log, self.dt_bias)
        core_attn_out[:tokens] = flat_gdn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            g,
            beta,
            conv_state,
            recurrent_state,
            slots,
            self.step.positions[:tokens, GDN],
            requests,
            num_requests,
            key_heads=self.num_k_heads // self.tp_size,
            key_dim=self.head_k_dim,
            value_dim=self.head_v_dim,
        )


class DuplexIOQwenDecoderLayer(nn.Module):
    """Dense Qwen3.5 decoder layer with explicit DuplexIO metadata flow."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        step: FlatStep,
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
                step,
                prefix=f"{prefix}.linear_attn",
            )
        elif layer_type == "full_attention":
            self.self_attn = DuplexIOQwenAttention(
                config,
                vllm_config=vllm_config,
                step=step,
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
    ) -> tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
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
        # One pass's flat addressing, shared by every layer and installed by
        # the model before each backbone pass.
        self.step = FlatStep(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.device_config.device,
        )

        def get_layer(prefix: str) -> DuplexIOQwenDecoderLayer:
            layer_index = extract_layer_index(prefix)
            return DuplexIOQwenDecoderLayer(
                vllm_config,
                config.layer_types[layer_index],
                self.step,
                prefix,
            )

        self.layers = nn.ModuleList(get_layer(f"{prefix}.layers.{index}") for index in range(config.num_hidden_layers))
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        positions: Tensor,
        inputs_embeds: Tensor,
    ) -> Tensor:
        hidden_states, residual = inputs_embeds, None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )


__all__ = [
    "FlatStep",
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
