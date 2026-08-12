# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM implementation of the DuplexIO full-duplex model."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import xgrammar as xgr
from torch import Tensor, nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AgentAudioInputAdapter,
    AgentAudioOutputAdapter,
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_representation import (
    DelayedMimiRepresentation,
    DelayedMimiState,
    MimiEmbedding,
)
from vllm_omni.model_executor.models.duplexio.checkpoint import (
    load_voice_pools,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import (
    DuplexIOConfig,
)
from vllm_omni.model_executor.models.duplexio.depth_sampler import (
    DepthAutoregressiveSampler,
    DepthSamplerConfig,
    DepthSpeakerConditioning,
)
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerConfig,
    FastConformerStreamingState,
    FastConformerUserEncoder,
)
from vllm_omni.model_executor.models.duplexio.mimi import (
    MimiModel,
    MimiStreamingState,
)
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOQwenModel,
    update_duplexio_attention_metadata,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    duplexio_frame_positions,
    duplexio_logical_positions,
)
from vllm_omni.model_executor.models.duplexio.tool_calling import (
    ToolCallConstraintCompiler,
    ToolCallConstraintState,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

TEXT_STREAM_NAMES = ("system", "user", "agent", "tool_call")
TARGET_STREAM_NAMES = ("user", "agent", "tool_call")
USER_CELL = 1
AGENT_CELL = 2
TOOL_CALL_CELL = 3
AGENT_AUDIO_CELL = 5


@dataclass
class DuplexIORequestState:
    """Small request-owned state not represented by Qwen's KV caches."""

    text_input_ids: Tensor
    agent_audio_codes: Tensor
    agent_input_delay: DelayedMimiState
    agent_mimi: MimiStreamingState
    user_delay: DelayedMimiState
    agent_delay: DelayedMimiState
    user_mimi: MimiStreamingState
    output_mimi: MimiStreamingState
    user_asr: FastConformerStreamingState
    user_asr_prefill_features: Tensor
    speaker_embedding: Tensor
    depth_speaker_conditioning: DepthSpeakerConditioning
    system_token_ids: tuple[int, ...]
    sampling_generator: torch.Generator
    tool_call_constraint: ToolCallConstraintState | None = None
    system_token_offset: int = 0
    frames_seen: int = 0
    active_text_tokens: int = 0
    tool_call_sequence: int = 0
    cache_epoch: int = 0

    def fork(self) -> DuplexIORequestState:
        """Create an append-local state whose updates commit on success."""
        return DuplexIORequestState(
            text_input_ids=self.text_input_ids,
            agent_audio_codes=self.agent_audio_codes,
            agent_input_delay=DelayedMimiState(
                previous_acoustic_codes=self.agent_input_delay.previous_acoustic_codes,
                pending_semantic_code=self.agent_input_delay.pending_semantic_code,
            ),
            agent_mimi=self.agent_mimi.fork(),
            user_delay=DelayedMimiState(
                previous_acoustic_codes=self.user_delay.previous_acoustic_codes,
                pending_semantic_code=self.user_delay.pending_semantic_code,
            ),
            agent_delay=DelayedMimiState(
                previous_acoustic_codes=self.agent_delay.previous_acoustic_codes,
                pending_semantic_code=self.agent_delay.pending_semantic_code,
            ),
            user_mimi=self.user_mimi.fork(),
            output_mimi=self.output_mimi.fork(),
            user_asr=self.user_asr,
            user_asr_prefill_features=self.user_asr_prefill_features,
            speaker_embedding=self.speaker_embedding,
            depth_speaker_conditioning=self.depth_speaker_conditioning,
            system_token_ids=self.system_token_ids,
            sampling_generator=_fork_generator(self.sampling_generator),
            tool_call_constraint=(
                self.tool_call_constraint.fork()
                if self.tool_call_constraint is not None
                else None
            ),
            system_token_offset=self.system_token_offset,
            frames_seen=self.frames_seen,
            active_text_tokens=self.active_text_tokens,
            tool_call_sequence=self.tool_call_sequence,
            cache_epoch=self.cache_epoch,
        )


@dataclass(frozen=True, slots=True)
class TokenSamplingOptions:
    mode: str
    temperature: float
    top_k: int
    top_p: float
    suppressed_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EmitSamplingTemperatures:
    user: float
    agent: float
    tool_call: float


class _DuplexIOBaseModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.model = DuplexIOQwenModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model",
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.lm_head",
            )


class _DuplexIOMultiStreamQwen(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        text_config = vllm_config.model_config.hf_text_config
        self.base_model = _DuplexIOBaseModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.base_model",
        )
        self.channel_emb = nn.Parameter(
            torch.zeros(len(TEXT_STREAM_NAMES), text_config.hidden_size)
        )
        self.output_head_proj = nn.ModuleDict(
            {
                name: nn.Linear(text_config.hidden_size, text_config.hidden_size)
                for name in TARGET_STREAM_NAMES
            }
        )


class DuplexIOForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    CustomProcessMixin,
):
    """Serve framed DuplexIO streams through resumable vLLM requests."""

    packed_modules_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True
    postprocess_uses_hidden_states = False
    postprocess_uses_multimodal_outputs = False
    requires_request_sample_eligibility = True
    # The training export includes a system-stream projection. Serving feeds
    # the system stream as context and predicts only user, agent, and tool_call.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"llm.output_head_proj.system.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        if not isinstance(config, DuplexIOConfig):
            raise TypeError(
                "DuplexIOForConditionalGeneration requires DuplexIOConfig"
            )
        _validate_vllm_runtime_contract(vllm_config)
        self.vllm_config = vllm_config
        self.config = config
        self.text_config = vllm_config.model_config.hf_text_config
        self.full_cudagraph_enabled = (
            vllm_config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL
            and not vllm_config.model_config.enforce_eager
        )
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("duplexio", "key_active"),
            ("duplexio", "request_epochs"),
            ("duplexio", "text_ordinals"),
            ("duplexio", "agent_audio_skip"),
        }
        graph_device = vllm_config.device_config.device
        graph_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "graph_key_active",
            torch.ones(graph_tokens, dtype=torch.bool, device=graph_device),
            persistent=False,
        )
        self.register_buffer(
            "graph_request_epochs",
            torch.zeros(graph_tokens, dtype=torch.long, device=graph_device),
            persistent=False,
        )
        self.register_buffer(
            "graph_text_ordinals",
            torch.zeros(graph_tokens, dtype=torch.long, device=graph_device),
            persistent=False,
        )
        self.pad_token_id = config.pad_token_id
        self.silence_token_id = config.silence_token_id
        if self.pad_token_id is None or self.silence_token_id is None:
            raise ValueError("DuplexIO requires exported pad and silence token IDs")

        self.llm = _DuplexIOMultiStreamQwen(
            vllm_config=vllm_config,
            prefix=f"{prefix}.llm" if prefix else "llm",
        )
        quantized_config = config.quantized_audio_config
        adapter_config = config.audio_adapter_config
        num_codebooks = quantized_config["num_codebooks"]
        codebook_size = quantized_config["codebook_size"]
        representation_dim = quantized_config["embedding_dim"]
        hidden_size = self.text_config.hidden_size
        adapter_hidden_size = adapter_config.get("hidden_size") or hidden_size

        self.audio_codec = MimiModel(config.audio_codec_config)
        self.audio_representation = DelayedMimiRepresentation(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            acoustic_delay_frames=quantized_config["acoustic_delay_frames"],
        )
        self.user_audio_embedding = MimiEmbedding(
            num_codebooks,
            codebook_size,
            representation_dim,
        )
        self.agent_audio_embedding = MimiEmbedding(
            num_codebooks,
            codebook_size,
            representation_dim,
        )
        self.user_audio_input_adapter = AudioInputAdapter(
            representation_dim,
            adapter_hidden_size,
            hidden_size,
        )
        self.agent_audio_input_adapter = AgentAudioInputAdapter(
            representation_dim,
            config.speaker_embed_dim,
            adapter_hidden_size,
            hidden_size,
        )
        self.agent_audio_output_adapter = AgentAudioOutputAdapter(
            hidden_size,
            adapter_hidden_size,
            config.speaker_embed_dim,
            adapter_hidden_size,
            adapter_config.get("agent_audio_skip_dropout", 0.0),
        )
        asr_config = FastConformerConfig(
            **{
                key: value
                for key, value in config.user_asr_encoder_config.items()
                if key != "implementation"
                and key != "attention_right_context"
                and key != "model_id"
            }
        )
        self.user_asr_encoder = FastConformerUserEncoder(
            asr_config,
            compute_dtype=vllm_config.model_config.dtype,
        )
        compilation = vllm_config.compilation_config
        self.compiled_user_asr_step: Callable[..., tuple[Tensor, ...]] | None = (
            torch.compile(
                self.user_asr_encoder.steady_step,
                fullgraph=True,
            )
            if (
                graph_device.type == "cuda"
                and not vllm_config.model_config.enforce_eager
                and compilation.backend == "inductor"
                and compilation.mode != CompilationMode.NONE
            )
            else None
        )
        self.user_asr_proj = nn.Linear(
            asr_config.dim,
            representation_dim,
            bias=False,
        )
        depth_config = config.depth_transformer_config
        self.audio_sampler = DepthAutoregressiveSampler(
            DepthSamplerConfig(
                conditioning_dim=hidden_size,
                speaker_embedding_dim=config.speaker_embed_dim,
                text_vocab_size=self.text_config.vocab_size,
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                low_rank_embeddings=depth_config.get("low_rank_embeddings", 128),
                dim=depth_config.get("dim", 1_024),
                num_layers=depth_config.get("num_layers", 6),
                num_heads=depth_config.get("num_heads", 16),
                feedforward_dim=depth_config.get("mlp_dim", 4_224),
                sampling_temperature=depth_config.get(
                    "sampling_temperature", 0.8
                ),
                sampling_top_k=depth_config.get("sampling_top_k", 250),
                semantic_sampling_top_k=depth_config.get(
                    "semantic_sampling_top_k"
                ),
            )
        )
        self.emit_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_size, 1)
                for name in TARGET_STREAM_NAMES
            }
        )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.llm.base_model.model.make_empty_intermediate_tensors
        )
        self._voice_pools = load_voice_pools(
            vllm_config.model_config.model,
            speaker_embed_dim=config.speaker_embed_dim,
            default_voice=config.default_voice,
            revision=vllm_config.model_config.revision,
        )
        self._forced_next_token_ids: list[int] | None = None
        self._next_cache_epoch = 1
        self.tool_call_compiler = ToolCallConstraintCompiler(
            cached_tokenizer_from_config(vllm_config.model_config),
            self.text_config.vocab_size,
        )
        self.set_custom_preprocess(self.preprocess)
        self.set_custom_postprocess(self.postprocess)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.llm.base_model.model.embed_input_ids(input_ids)

    def update_attention_metadata(
        self,
        attn_metadata: object,
        request_infos: list[dict[str, Any]],
    ) -> None:
        """Expose accepted audio and active-text counts to Flex metadata."""
        active_text_tokens = []
        live_audio_frames = []
        for info in request_infos:
            state = info.get("duplexio_model_state")
            accepted_text_tokens = (
                state.active_text_tokens
                if isinstance(state, DuplexIORequestState)
                else 0
            )
            accepted_frames = (
                state.frames_seen
                if isinstance(state, DuplexIORequestState)
                else 0
            )
            duplex = info.get("duplex")
            frame_count = (
                duplex.get("frame_count", 1)
                if isinstance(duplex, Mapping)
                else 1
            )
            assert isinstance(frame_count, int) and frame_count > 0
            active_text_tokens.append(
                accepted_text_tokens
                + (frame_count - 1) * len(TEXT_STREAM_NAMES)
            )
            live_audio_frames.append(accepted_frames + frame_count)
        update_duplexio_attention_metadata(
            attn_metadata,
            active_text_tokens,
            live_audio_frames,
        )

    def update_graph_inputs(
        self,
        request_infos: list[dict[str, Any]],
    ) -> None:
        """Copy current preprocessed metadata into stable CUDA-graph inputs."""
        if not request_infos:
            self.graph_key_active.fill_(True)
            self.graph_request_epochs.zero_()
            self.graph_text_ordinals.zero_()
            return

        duplex_infos = [info["duplexio"] for info in request_infos]
        key_active = torch.cat([info["key_active"] for info in duplex_infos])
        request_epochs = torch.cat(
            [info["request_epochs"] for info in duplex_infos]
        )
        text_ordinals = torch.cat(
            [info["text_ordinals"] for info in duplex_infos]
        )
        token_count = key_active.shape[0]
        self.graph_key_active[:token_count].copy_(key_active)
        self.graph_request_epochs[:token_count].copy_(request_epochs)
        self.graph_text_ordinals[:token_count].copy_(text_ordinals)

    def supports_cudagraph_replay(
        self,
        request_infos: list[dict[str, Any]],
    ) -> bool:
        """Replay only ordinary live frames, never context-prefill frames."""
        return all(
            not bool(info["duplex"].get("duplexio_prefill", False))
            and not bool(info["duplex"].get("duplexio_system_input", False))
            for info in request_infos
        )

    @torch.inference_mode()
    def preprocess(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        **info: Any,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        del input_embeds
        duplex = info.get("duplex")
        if not isinstance(duplex, Mapping):
            raise ValueError("Native DuplexIO accepts only framed duplex appends")
        frame_count = duplex.get("frame_count")
        if (
            not isinstance(frame_count, int)
            or frame_count < 1
            or input_ids.numel() != frame_count * DUPLEXIO_NUM_CELLS
        ):
            raise ValueError(
                "Native DuplexIO requires complete six-cell frames; "
                f"got frame_count={frame_count}, tokens={input_ids.numel()}"
            )
        runtime_config = duplex.get("runtime_config")
        if not isinstance(runtime_config, Mapping):
            raise ValueError("DuplexIO append is missing runtime_config")
        state = info.get("duplexio_model_state")
        if not isinstance(state, DuplexIORequestState):
            state = self._new_request_state(runtime_config, input_ids.device)
        else:
            state = state.fork()

        is_prefill = duplex.get("duplexio_prefill", False)
        prefill_final = duplex.get("duplexio_prefill_final", False)
        is_system_input = duplex.get("duplexio_system_input", False)
        system_input_final = duplex.get("duplexio_system_input_final", False)
        if not all(
            isinstance(value, bool)
            for value in (
                is_prefill,
                prefill_final,
                is_system_input,
                system_input_final,
            )
        ):
            raise ValueError(
                "DuplexIO text-input flags must be boolean when present"
            )
        if prefill_final and not is_prefill:
            raise ValueError("DuplexIO prefill_final requires duplexio_prefill")
        if system_input_final and not is_system_input:
            raise ValueError(
                "DuplexIO system_input_final requires duplexio_system_input"
            )
        if is_prefill and is_system_input:
            raise ValueError("DuplexIO prefill and system input are mutually exclusive")
        if frame_count != 1 and not (is_prefill or is_system_input):
            raise ValueError(
                "Native DuplexIO batches only silent text-input frames"
            )

        text_ids = state.text_input_ids.expand(frame_count, -1).clone()
        if is_system_input:
            system_token_ids = duplex.get("duplexio_system_token_ids")
            if (
                not isinstance(system_token_ids, list)
                or len(system_token_ids) != frame_count
                or not all(
                    isinstance(token_id, int) and token_id >= 0
                    for token_id in system_token_ids
                )
            ):
                raise ValueError(
                    "DuplexIO system input requires one token ID per frame"
                )
            text_ids.fill_(self.silence_token_id)
            text_ids[:, 0] = torch.tensor(
                system_token_ids,
                dtype=torch.long,
                device=input_ids.device,
            )
            state.text_input_ids = torch.full_like(
                state.text_input_ids,
                self.silence_token_id,
            )
        else:
            system_token_start = state.system_token_offset
            system_token_end = system_token_start + frame_count
            if is_prefill and system_token_end > len(state.system_token_ids):
                raise ValueError(
                    "DuplexIO prefill exceeds the remaining system tokens"
                )
            system_tokens = state.system_token_ids[
                system_token_start:system_token_end
            ]
            if system_tokens:
                text_ids[: len(system_tokens), 0] = torch.tensor(
                    system_tokens,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                state.system_token_offset += len(system_tokens)
            if len(system_tokens) < frame_count:
                text_ids[len(system_tokens) :, 0] = self.silence_token_id

        codec_parameter = next(self.audio_codec.parameters())
        if is_prefill or is_system_input:
            waveform = torch.zeros(
                1,
                1,
                frame_count * self.config.frame_size,
                device=input_ids.device,
                dtype=codec_parameter.dtype,
            )
        else:
            waveform = _decode_frame(
                duplex["payload"],
                input_ids.device,
                codec_parameter.dtype,
            )
        expected_samples = frame_count * self.config.frame_size
        if waveform.shape[-1] != expected_samples:
            raise ValueError(
                "DuplexIO append PCM does not match its frame count: "
                f"{waveform.shape[-1]} != {expected_samples}"
            )
        with torch.profiler.record_function("duplexio.user_codec_encode"):
            raw_user_codes = self.audio_codec.encode(
                waveform,
                self.audio_representation.num_codebooks,
                state.user_mimi,
            )[0].T
        user_codes = self.audio_representation.encode_sequence(
            raw_user_codes,
            state.user_delay,
        )
        with torch.profiler.record_function("duplexio.user_fastconformer"):
            if is_prefill:
                asr_features = state.user_asr_prefill_features[
                    system_token_start:system_token_end
                ]
            else:
                asr_features, state.user_asr = self.step_user_asr(
                    waveform,
                    state.user_asr,
                )
        user_features = self.user_audio_embedding(user_codes)
        user_features = user_features + self.user_asr_proj(
            asr_features.to(user_features.dtype)
        )
        if is_system_input:
            agent_codes = self.encode_silent_agent_frames(
                frame_count,
                state.agent_mimi,
                state.agent_input_delay,
                input_ids.device,
            )
            state.agent_audio_codes = agent_codes[-1]
        elif is_prefill:
            has_remaining_system_tokens = (
                state.system_token_offset < len(state.system_token_ids)
            )
            frames_to_encode = frame_count if has_remaining_system_tokens else frame_count - 1
            following_codes = self.encode_silent_agent_frames(
                frames_to_encode,
                state.agent_mimi,
                state.agent_input_delay,
                input_ids.device,
            )
            agent_codes = torch.cat(
                (
                    state.agent_audio_codes.unsqueeze(0),
                    following_codes[: frame_count - 1],
                )
            )
            if following_codes.shape[0] > 0:
                state.agent_audio_codes = following_codes[-1]
        else:
            agent_codes = state.agent_audio_codes.unsqueeze(0)
        agent_features = self.agent_audio_embedding(agent_codes)
        speaker = state.speaker_embedding.unsqueeze(0)
        request_index = torch.zeros(
            frame_count,
            dtype=torch.long,
            device=input_ids.device,
        )
        user_hidden, _ = self.user_audio_input_adapter(user_features)
        agent_hidden, agent_skip = self.agent_audio_input_adapter(
            agent_features,
            speaker,
            request_index,
        )

        if is_prefill:
            has_remaining_system_tokens = (
                state.system_token_offset < len(state.system_token_ids)
            )
            if prefill_final and has_remaining_system_tokens:
                raise ValueError(
                    "DuplexIO final prefill frame left system tokens unconsumed"
                )
            if not prefill_final and not has_remaining_system_tokens:
                raise ValueError(
                    "DuplexIO non-final prefill frame consumed all system tokens"
                )
            if prefill_final:
                # Validation encodes continuation user audio as a fresh stream;
                # prompt audio only conditions the prompt's cached activations.
                state.user_mimi = self.audio_codec.new_streaming_state()
                state.user_delay = self.audio_representation.new_state(
                    device=input_ids.device
                )
        text_hidden = self.llm.base_model.model.embed_input_ids(
            text_ids.flatten()
        ).view(frame_count, len(TEXT_STREAM_NAMES), -1)
        text_hidden = text_hidden.masked_fill(
            (text_ids == self.silence_token_id).unsqueeze(-1),
            0,
        )
        text_hidden = text_hidden + self.llm.channel_emb
        embeddings = torch.cat(
            (
                text_hidden,
                user_hidden.unsqueeze(1),
                agent_hidden.unsqueeze(1),
            ),
            dim=1,
        ).flatten(0, 1)
        text_active = (
            (text_ids != self.pad_token_id)
            & (text_ids != self.silence_token_id)
        )
        key_active = torch.cat(
            (
                text_active,
                torch.full(
                    (frame_count, 2),
                    not is_prefill and not is_system_input,
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
            ),
            dim=1,
        ).flatten()
        text_ordinals = torch.zeros(
            frame_count,
            DUPLEXIO_NUM_CELLS,
            dtype=torch.long,
            device=input_ids.device,
        )
        frame_ordinals = (
            torch.cumsum(text_active.flatten().to(torch.long), dim=0)
            + state.active_text_tokens
        ).view_as(text_active)
        text_ordinals[:, : len(TEXT_STREAM_NAMES)] = torch.where(
            text_active,
            frame_ordinals,
            0,
        )
        state.active_text_tokens += int(text_active.sum().item())
        state.frames_seen += frame_count
        return input_ids, embeddings, {
            "duplexio_working_state": state,
            "duplexio": {
                "key_active": key_active,
                "request_epochs": torch.full(
                    (frame_count * DUPLEXIO_NUM_CELLS,),
                    state.cache_epoch,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "text_ordinals": text_ordinals.flatten(),
                "agent_audio_skip": agent_skip[-1],
            },
        }

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        request_sample_eligible: list[bool] | None = None,
        **kwargs: object,
    ) -> Tensor | IntermediateTensors:
        del input_ids, request_token_spans, request_sample_eligible, kwargs
        if inputs_embeds is None:
            raise ValueError("Native DuplexIO requires precomputed frame embeddings")
        token_count = inputs_embeds.shape[0]
        if self.full_cudagraph_enabled or not model_intermediate_buffer:
            key_active = self.graph_key_active[:token_count]
            request_epochs = self.graph_request_epochs[:token_count]
            text_ordinals = self.graph_text_ordinals[:token_count]
        else:
            infos = model_intermediate_buffer or []
            duplex_infos = [info["duplexio"] for info in infos]
            key_active = torch.cat(
                [info["key_active"] for info in duplex_infos]
            )
            request_epochs = torch.cat(
                [info["request_epochs"] for info in duplex_infos]
            )
            text_ordinals = torch.cat(
                [info["text_ordinals"] for info in duplex_infos]
            )
        with torch.profiler.record_function("duplexio.backbone"):
            return self.llm.base_model.model(
                positions=duplexio_frame_positions(positions),
                logical_positions=duplexio_logical_positions(positions),
                key_active=key_active,
                request_epochs=request_epochs,
                text_ordinals=text_ordinals,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

    def make_omni_output(
        self,
        model_output: Tensor | IntermediateTensors,
        **kwargs: Any,
    ) -> OmniOutput | IntermediateTensors:
        """Sample the three text streams and audio outside the model graph."""
        if isinstance(model_output, IntermediateTensors):
            return model_output

        hidden_states = model_output
        raw_infos = kwargs.get("model_intermediate_buffer")
        if raw_infos is None:
            infos: list[dict[str, Any]] = []
        else:
            assert isinstance(raw_infos, list)
            infos = cast(list[dict[str, Any]], raw_infos)
        if not infos:
            self._forced_next_token_ids = None
            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs={},
            )

        raw_request_token_spans = kwargs.get("request_token_spans")
        if not isinstance(raw_request_token_spans, list):
            raise RuntimeError("DuplexIO live execution requires request_token_spans")
        request_token_spans = cast(
            list[tuple[int, int]],
            raw_request_token_spans,
        )
        if len(request_token_spans) != len(infos):
            raise RuntimeError(
                "DuplexIO received "
                f"{len(request_token_spans)} spans for {len(infos)} requests"
            )
        raw_request_sample_eligible = kwargs.get("request_sample_eligible")
        if not isinstance(raw_request_sample_eligible, list):
            raise RuntimeError(
                "DuplexIO live execution requires request_sample_eligible"
            )
        request_sample_eligible = cast(
            list[bool],
            raw_request_sample_eligible,
        )
        if len(request_sample_eligible) != len(infos):
            raise RuntimeError(
                "DuplexIO received "
                f"{len(request_sample_eligible)} sampling flags for "
                f"{len(infos)} requests"
            )
        if not all(request_sample_eligible):
            raise RuntimeError(
                "DuplexIO frames must be scheduled atomically; disable chunked "
                "prefill and speculative decoding"
            )

        forced_ids: list[int] = []
        audio_outputs: list[Tensor] = []
        audio_token_ids: list[Tensor] = []
        user_ids: list[Tensor] = []
        agent_ids: list[Tensor] = []
        tool_ids: list[Tensor] = []
        listen_flags: list[Tensor] = []
        end_flags: list[Tensor] = []
        epochs: list[Tensor] = []
        turn_ids: list[Tensor] = []
        prefill_flags: list[Tensor] = []
        prefill_complete_flags: list[Tensor] = []
        system_input_flags: list[Tensor] = []
        system_input_complete_flags: list[Tensor] = []
        tool_call_complete_flags: list[Tensor] = []
        tool_call_payloads: list[Tensor] = []
        for request_index, ((start, end), info) in enumerate(
            zip(request_token_spans, infos, strict=True)
        ):
            state = info.get("duplexio_working_state")
            if not isinstance(state, DuplexIORequestState):
                raise RuntimeError(
                    f"DuplexIO request {request_index} is missing working state"
                )
            span_length = end - start
            if span_length < DUPLEXIO_NUM_CELLS or span_length % DUPLEXIO_NUM_CELLS:
                raise ValueError(
                    "DuplexIO request span must contain complete frames, got "
                    f"({start}, {end})"
                )
            duplex_info = info.get("duplexio")
            if not isinstance(duplex_info, Mapping):
                raise RuntimeError(
                    f"DuplexIO request {request_index} is missing frame metadata"
                )
            agent_audio_skip = duplex_info.get("agent_audio_skip")
            if not isinstance(agent_audio_skip, Tensor):
                raise RuntimeError(
                    f"DuplexIO request {request_index} is missing audio skip metadata"
                )
            duplex = info.get("duplex", {})
            if not isinstance(duplex, Mapping):
                raise RuntimeError(
                    f"DuplexIO request {request_index} is missing duplex metadata"
                )
            is_prefill = duplex.get("duplexio_prefill", False)
            prefill_final = duplex.get("duplexio_prefill_final", False)
            is_system_input = duplex.get("duplexio_system_input", False)
            system_input_final = duplex.get(
                "duplexio_system_input_final",
                False,
            )
            if not all(
                isinstance(value, bool)
                for value in (
                    is_prefill,
                    prefill_final,
                    is_system_input,
                    system_input_final,
                )
            ):
                raise RuntimeError(
                    f"DuplexIO request {request_index} has invalid text-input flags"
                )
            if prefill_final and not is_prefill:
                raise RuntimeError(
                    f"DuplexIO request {request_index} has invalid final prefill flag"
                )
            if system_input_final and not is_system_input:
                raise RuntimeError(
                    f"DuplexIO request {request_index} has invalid final system-input flag"
                )
            frame_count = duplex.get("frame_count")
            if frame_count != span_length // DUPLEXIO_NUM_CELLS:
                raise RuntimeError(
                    f"DuplexIO request {request_index} frame span does not match metadata"
                )
            row_hidden = hidden_states[end - DUPLEXIO_NUM_CELLS : end]
            suppress_output = (is_prefill and not prefill_final) or (
                is_system_input and not system_input_final
            )
            if suppress_output:
                silence_ids = row_hidden.new_full(
                    (1,),
                    self.silence_token_id,
                    dtype=torch.long,
                )
                forced_ids.append(self.silence_token_id)
                user_ids.append(silence_ids)
                agent_ids.append(silence_ids)
                tool_ids.append(silence_ids)
                audio_outputs.append(row_hidden.new_empty(0, dtype=torch.float32))
                audio_token_ids.append(row_hidden.new_empty(0, dtype=torch.long))
                listen_flags.append(torch.tensor([False]))
                end_flags.append(torch.tensor([bool(duplex.get("final", False))]))
                epochs.append(torch.tensor([int(duplex.get("epoch", 0))]))
                turn_ids.append(torch.tensor([int(duplex.get("turn_id", 0))]))
                prefill_flags.append(torch.tensor([is_prefill]))
                prefill_complete_flags.append(torch.tensor([False]))
                system_input_flags.append(torch.tensor([is_system_input]))
                system_input_complete_flags.append(torch.tensor([False]))
                tool_call_complete_flags.append(torch.tensor([False]))
                tool_call_payloads.append(
                    row_hidden.new_empty(0, dtype=torch.uint8)
                )
                continue
            with torch.profiler.record_function("duplexio.text_sampling"):
                predicted_text, tool_call = self._sample_text(
                    row_hidden,
                    info,
                    state.tool_call_constraint,
                    state.sampling_generator,
                )
            audio_condition = self.agent_audio_output_adapter(
                row_hidden[AGENT_AUDIO_CELL : AGENT_AUDIO_CELL + 1],
                agent_audio_skip.unsqueeze(0),
                state.speaker_embedding.unsqueeze(0),
                torch.zeros(1, dtype=torch.long, device=row_hidden.device),
            )
            depth_sampling = _depth_sampling(info)
            with torch.profiler.record_function("duplexio.depth_sampling"):
                with torch.autocast(
                    device_type=row_hidden.device.type,
                    dtype=self.vllm_config.model_config.dtype,
                    enabled=(
                        row_hidden.is_cuda
                        and self.vllm_config.model_config.dtype != torch.float32
                    ),
                ):
                    predicted_audio = self.audio_sampler.sample(
                        audio_condition,
                        predicted_text[1:2],
                        state.depth_speaker_conditioning,
                        temperature=depth_sampling[0],
                        top_k=depth_sampling[1],
                        generator=state.sampling_generator,
                    )[0]
            state.text_input_ids = torch.cat(
                (
                    predicted_text.new_tensor([self.silence_token_id]),
                    predicted_text,
                )
            )
            state.agent_audio_codes = predicted_audio
            raw_audio_codes = self.audio_representation.decode_column(
                predicted_audio,
                state.agent_delay,
            )
            with torch.profiler.record_function("duplexio.output_codec_decode"):
                waveform = (
                    row_hidden.new_empty(0, dtype=torch.float32)
                    if raw_audio_codes is None
                    else self.audio_codec.decode(
                        raw_audio_codes.view(1, -1, 1),
                        state.output_mimi,
                    )[0, 0]
                )
            forced_ids.append(int(predicted_text[1].item()))

            user_ids.append(predicted_text[0:1].detach())
            agent_ids.append(predicted_text[1:2].detach())
            tool_ids.append(predicted_text[2:3].detach())
            audio_outputs.append(waveform.detach())
            audio_token_ids.append(predicted_audio.detach())
            model_listen = bool(
                predicted_text[1].item() == self.silence_token_id
                and predicted_text[2].item() == self.silence_token_id
            )
            listen_flags.append(torch.tensor([model_listen]))
            end_flags.append(torch.tensor([bool(duplex.get("final", False))]))
            epochs.append(torch.tensor([int(duplex.get("epoch", 0))]))
            turn_ids.append(torch.tensor([int(duplex.get("turn_id", 0))]))
            prefill_flags.append(
                torch.tensor([is_prefill and not prefill_final])
            )
            prefill_complete_flags.append(
                torch.tensor([is_prefill and prefill_final])
            )
            system_input_flags.append(
                torch.tensor([is_system_input and not system_input_final])
            )
            system_input_complete_flags.append(
                torch.tensor([is_system_input and system_input_final])
            )
            tool_call_complete_flags.append(
                torch.tensor([tool_call is not None])
            )
            if tool_call is not None:
                state.tool_call_sequence += 1
            tool_call_payloads.append(
                serialize_tool_call(
                    tool_call,
                    state.tool_call_sequence,
                    row_hidden.device,
                )
            )

        self._forced_next_token_ids = forced_ids
        multimodal_outputs = cast(
            Any,
            {
                "audio": audio_outputs,
                "agent_audio_token_ids": audio_token_ids,
                "sample_rate_hz": [torch.tensor([self.config.sample_rate])]
                * len(audio_outputs),
                "user_token_id": user_ids,
                "agent_token_id": agent_ids,
                "tool_call_token_id": tool_ids,
                "model_listen": listen_flags,
                "end_of_turn": end_flags,
                "duplex_epoch": epochs,
                "duplex_turn_id": turn_ids,
                "duplex_prefill": prefill_flags,
                "duplex_prefill_complete": prefill_complete_flags,
                "duplex_system_input": system_input_flags,
                "duplex_system_input_complete": system_input_complete_flags,
                "tool_call_complete": tool_call_complete_flags,
                "tool_call_json": tool_call_payloads,
            },
        )
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def _sample_text(
        self,
        row_hidden: Tensor,
        info: Mapping[str, object],
        tool_call_constraint: ToolCallConstraintState | None,
        generator: torch.Generator,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        cells = row_hidden[[USER_CELL, AGENT_CELL, TOOL_CALL_CELL]]
        projected = torch.stack(
            [
                self.llm.output_head_proj[name](cells[index])
                for index, name in enumerate(TARGET_STREAM_NAMES)
            ]
        )
        logits = self.logits_processor(self.llm.base_model.lm_head, projected)
        emit_logits = torch.stack(
            [
                self.emit_heads[name](projected[index]).squeeze(-1)
                for index, name in enumerate(TARGET_STREAM_NAMES)
            ]
        )
        sampling = _text_sampling(info)
        emit_temperatures = _emit_temperatures(info)
        user_sampling = TokenSamplingOptions(
            mode="argmax",
            temperature=1.0,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            suppressed_token_ids=sampling.suppressed_token_ids,
        )
        user_id = _sample_factorized_text_ids(
            logits[:1],
            emit_logits[:1],
            silence_token_id=self.silence_token_id,
            sampling=user_sampling,
            emit_temperature=emit_temperatures.user,
            generator=generator,
        )
        agent_sampling = TokenSamplingOptions(
            mode=sampling.mode,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            suppressed_token_ids=(
                *sampling.suppressed_token_ids,
                *_agent_suppressed_token_ids(info),
            ),
        )
        agent_id = _sample_factorized_text_ids(
            logits[1:2],
            emit_logits[1:2],
            silence_token_id=self.silence_token_id,
            sampling=agent_sampling,
            emit_temperature=emit_temperatures.agent,
            generator=generator,
        )
        tool_id, tool_call_complete = _sample_tool_call_token_id(
            logits[2:3],
            emit_logits[2:3],
            constraint=tool_call_constraint,
            silence_token_id=self.silence_token_id,
            sampling=sampling,
            emit_temperature=emit_temperatures.tool_call,
            generator=generator,
        )
        tool_call = (
            self.tool_call_compiler.take_completed_call(tool_call_constraint)
            if tool_call_complete and tool_call_constraint is not None
            else None
        )
        return torch.cat((user_id, agent_id, tool_id)), tool_call

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> Tensor | None:
        del sampling_metadata
        if hidden_states.numel() == 0:
            return None
        token_ids = self._forced_next_token_ids
        self._forced_next_token_ids = None
        if token_ids is None:
            token_ids = [self.silence_token_id] * hidden_states.shape[0]
        if len(token_ids) != hidden_states.shape[0]:
            raise RuntimeError(
                "DuplexIO forced-token batch mismatch: "
                f"{len(token_ids)} ids for {hidden_states.shape[0]} rows"
            )
        logits = torch.full(
            (hidden_states.shape[0], self.text_config.vocab_size),
            -torch.inf,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        logits[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            torch.tensor(token_ids, device=hidden_states.device),
        ] = 0
        return logits

    def postprocess(
        self,
        model_output: object,
        **info_dict: object,
    ) -> dict[str, object]:
        del model_output
        state = info_dict.get("duplexio_working_state")
        return {"duplexio_model_state": state} if state is not None else {}

    def _new_request_state(
        self,
        runtime_config: Mapping[str, object],
        device: torch.device,
    ) -> DuplexIORequestState:
        voice = runtime_config.get("duplexio_voice")
        if not isinstance(voice, str) or voice not in self._voice_pools:
            raise ValueError(f"DuplexIO request selected unknown voice {voice!r}")
        pool = self._voice_pools[voice]
        embedding_index = runtime_config.get("duplexio_voice_embedding_index", 0)
        if not isinstance(embedding_index, int) or not 0 <= embedding_index < len(pool):
            raise ValueError(
                f"Voice embedding index {embedding_index!r} is invalid for {voice!r}"
            )
        system_tokens = runtime_config.get("duplexio_system_token_ids", ())
        if not isinstance(system_tokens, (list, tuple)) or not all(
            isinstance(token, int) for token in system_tokens
        ):
            raise ValueError("duplexio_system_token_ids must be integer token IDs")
        sampling_seed = runtime_config.get("duplexio_sampling_seed")
        if not isinstance(sampling_seed, int) or sampling_seed < 0:
            raise ValueError("duplexio_sampling_seed must be a non-negative integer")
        sampling_generator = torch.Generator(device=device)
        sampling_generator.manual_seed(sampling_seed)
        tools = runtime_config.get("duplexio_tools", [])
        tool_choice = runtime_config.get(
            "duplexio_tool_choice",
            {"mode": "none"},
        )
        if not isinstance(tools, list) or not all(
            isinstance(tool, Mapping) for tool in tools
        ):
            raise ValueError("duplexio_tools must be a list of tool definitions")
        if not isinstance(tool_choice, Mapping):
            raise ValueError("duplexio_tool_choice must be an object")
        cache_epoch = self._next_cache_epoch
        if cache_epoch >= 2**32:
            raise RuntimeError("DuplexIO cache epoch space is exhausted")
        self._next_cache_epoch += 1
        agent_input_delay = self.audio_representation.new_state(device=device)
        agent_mimi = self.audio_codec.new_streaming_state()
        user_asr_prefill_features, user_asr = (
            self.user_asr_encoder.encode_silent_prefix(
                len(system_tokens),
                device=device,
            )
        )
        speaker_embedding = pool[embedding_index].to(
            device=device,
            dtype=self.llm.channel_emb.dtype,
        )
        return DuplexIORequestState(
            text_input_ids=torch.full(
                (len(TEXT_STREAM_NAMES),),
                self.silence_token_id,
                dtype=torch.long,
                device=device,
            ),
            agent_audio_codes=self.encode_silent_agent_frame(
                agent_mimi,
                agent_input_delay,
                device,
            ),
            agent_input_delay=agent_input_delay,
            agent_mimi=agent_mimi,
            user_delay=self.audio_representation.new_state(device=device),
            agent_delay=self.audio_representation.new_state(device=device),
            user_mimi=self.audio_codec.new_streaming_state(),
            output_mimi=self.audio_codec.new_streaming_state(),
            user_asr=user_asr,
            user_asr_prefill_features=user_asr_prefill_features,
            speaker_embedding=speaker_embedding,
            depth_speaker_conditioning=self.audio_sampler.prepare_speaker(
                speaker_embedding.unsqueeze(0)
            ),
            system_token_ids=cast(tuple[int, ...], tuple(system_tokens)),
            sampling_generator=sampling_generator,
            tool_call_constraint=self.tool_call_compiler.new_state(
                cast(list[Mapping[str, Any]], tools),
                tool_choice,
            ),
            cache_epoch=cache_epoch,
        )

    @torch.inference_mode()
    def encode_silent_agent_frame(
        self,
        mimi: MimiStreamingState,
        delay: DelayedMimiState,
        device: torch.device,
    ) -> Tensor:
        """Encode one zero waveform frame for the agent prompt stream."""
        return self.encode_silent_agent_frames(1, mimi, delay, device)[0]

    @torch.inference_mode()
    def encode_silent_agent_frames(
        self,
        frame_count: int,
        mimi: MimiStreamingState,
        delay: DelayedMimiState,
        device: torch.device,
    ) -> Tensor:
        """Encode silent agent prompt frames in one Mimi sequence."""
        if frame_count == 0:
            return delay.previous_acoustic_codes.new_empty(
                0,
                self.audio_representation.num_codebooks,
            )
        codec_parameter = next(self.audio_codec.parameters())
        waveform = torch.zeros(
            1,
            1,
            frame_count * self.config.frame_size,
            device=device,
            dtype=codec_parameter.dtype,
        )
        raw_codes = self.audio_codec.encode(
            waveform,
            self.audio_representation.num_codebooks,
            mimi,
        )[0].T
        return self.audio_representation.encode_sequence(raw_codes, delay)

    def step_user_asr(
        self,
        waveform: Tensor,
        state: FastConformerStreamingState,
    ) -> tuple[Tensor, FastConformerStreamingState]:
        """Use the compiled fixed-shape path after the ASR window is full."""
        compiled_step = self.compiled_user_asr_step
        if compiled_step is None:
            return self.user_asr_encoder.step_sequence(waveform, state)

        config = self.user_asr_encoder.config
        if (
            waveform.shape[-1] != config.frame_size
            or state.frames_seen < config.attention_left_context
        ):
            return self.user_asr_encoder.step_sequence(waveform, state)

        outputs = compiled_step(
            waveform,
            state.sample_buffer,
            state.feature_buffer,
            *state.attention_caches,
            *state.convolution_caches,
        )
        layer_count = config.num_layers
        return outputs[0], FastConformerStreamingState(
            sample_buffer=outputs[1],
            feature_buffer=outputs[2],
            attention_caches=tuple(outputs[3 : 3 + layer_count]),
            convolution_caches=tuple(outputs[3 + layer_count :]),
            frames_seen=state.frames_seen + 1,
        )

    @torch.inference_mode()
    def warmup_compiled_user_asr_step(self) -> None:
        """Compile the steady one-frame ASR path before accepting sessions."""
        compiled_step = self.compiled_user_asr_step
        if compiled_step is None:
            return
        config = self.user_asr_encoder.config
        device = next(self.user_asr_encoder.parameters()).device
        waveform = torch.zeros(
            1,
            1,
            config.frame_size,
            device=device,
            dtype=next(self.audio_codec.parameters()).dtype,
        )
        state = self.user_asr_encoder.new_state(device=device)
        for _ in range(config.attention_left_context):
            _, state = self.user_asr_encoder.step(waveform, state)
        with set_default_torch_dtype(torch.float32):
            outputs = compiled_step(
                waveform,
                state.sample_buffer,
                state.feature_buffer,
                *state.attention_caches,
                *state.convolution_caches,
            )
        assert outputs[0].shape == (1, config.dim)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
        self.warmup_compiled_user_asr_step()
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        text_config = vllm_config.model_config.hf_text_config
        effective_kernel_size = (
            (text_config.linear_conv_kernel_dim - 1) * DUPLEXIO_NUM_CELLS + 1
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            text_config.linear_num_key_heads,
            text_config.linear_num_value_heads,
            text_config.linear_key_head_dim,
            text_config.linear_value_head_dim,
            effective_kernel_size,
            0,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


def _decode_frame(
    payload: object,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not isinstance(payload, Mapping):
        raise ValueError("DuplexIO frame payload must be a mapping")
    encoded = payload.get("audio")
    if not isinstance(encoded, str):
        raise ValueError("DuplexIO frame payload is missing base64 audio")
    samples = np.frombuffer(base64.b64decode(encoded, validate=True), dtype="<f4")
    waveform = torch.from_numpy(samples.copy()).to(device=device, dtype=dtype)
    return waveform.view(1, 1, -1)


def _text_sampling(info: Mapping[str, object]) -> TokenSamplingOptions:
    duplex = info.get("duplex")
    if not isinstance(duplex, Mapping):
        raise ValueError("DuplexIO frame is missing runtime metadata")
    runtime = duplex.get("runtime_config")
    if not isinstance(runtime, Mapping):
        raise ValueError("DuplexIO frame is missing runtime configuration")
    sampling = runtime.get("duplexio_text_sampling")
    suppressed = runtime.get("duplexio_suppressed_token_ids")
    if not isinstance(sampling, Mapping) or not isinstance(suppressed, list):
        raise ValueError("DuplexIO frame is missing text sampling configuration")
    mode = sampling.get("mode")
    temperature = sampling.get("temperature")
    top_k = sampling.get("top_k")
    top_p = sampling.get("top_p")
    if (
        not isinstance(mode, str)
        or not isinstance(temperature, (int, float))
        or not isinstance(top_k, int)
        or not isinstance(top_p, (int, float))
        or not all(isinstance(token_id, int) for token_id in suppressed)
    ):
        raise ValueError("Invalid DuplexIO text sampling configuration")
    return TokenSamplingOptions(
        mode=mode,
        temperature=float(temperature),
        top_k=top_k,
        top_p=float(top_p),
        suppressed_token_ids=tuple(suppressed),
    )


def _emit_temperatures(info: Mapping[str, object]) -> EmitSamplingTemperatures:
    duplex = info.get("duplex")
    runtime = duplex.get("runtime_config") if isinstance(duplex, Mapping) else None
    temperatures = (
        runtime.get("duplexio_emit_temperatures")
        if isinstance(runtime, Mapping)
        else None
    )
    if not isinstance(temperatures, Mapping):
        raise ValueError("DuplexIO frame is missing emit temperatures")
    values = tuple(temperatures.get(name) for name in TARGET_STREAM_NAMES)
    if not all(
        isinstance(value, (int, float)) and value >= 0
        for value in values
    ):
        raise ValueError("Invalid DuplexIO emit temperatures")
    return EmitSamplingTemperatures(*(float(value) for value in values))


def _agent_suppressed_token_ids(
    info: Mapping[str, object],
) -> tuple[int, ...]:
    duplex = info.get("duplex")
    runtime = duplex.get("runtime_config") if isinstance(duplex, Mapping) else None
    suppressed = (
        runtime.get("duplexio_agent_suppressed_token_ids")
        if isinstance(runtime, Mapping)
        else None
    )
    if not isinstance(suppressed, list) or not all(
        isinstance(token_id, int) for token_id in suppressed
    ):
        raise ValueError("DuplexIO frame is missing agent token suppression")
    return tuple(suppressed)


def _sample_content_token_ids(
    logits: Tensor,
    sampling: TokenSamplingOptions,
    *,
    generator: torch.Generator,
) -> Tensor:
    if sampling.suppressed_token_ids:
        logits = logits.clone()
        token_ids = torch.tensor(
            sampling.suppressed_token_ids,
            device=logits.device,
        )
        logits.index_fill_(-1, token_ids, torch.finfo(logits.dtype).min)
    if sampling.mode in {"argmax", "max"}:
        return logits.argmax(dim=-1)

    scaled = logits.float() / sampling.temperature
    top_k = min(sampling.top_k, scaled.shape[-1])
    top_values, top_indices = torch.topk(scaled, k=top_k, dim=-1)
    if sampling.mode == "top_k":
        probabilities = torch.softmax(top_values, dim=-1)
    elif sampling.mode == "top_p":
        sorted_probabilities = torch.softmax(top_values, dim=-1)
        remove = sorted_probabilities.cumsum(dim=-1) > sampling.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        top_values = top_values.masked_fill(
            remove,
            torch.finfo(top_values.dtype).min,
        )
        probabilities = torch.softmax(top_values, dim=-1)
    else:
        raise ValueError(f"Unsupported DuplexIO sampling mode: {sampling.mode!r}")
    sampled = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
    return top_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


def _sample_factorized_text_ids(
    logits: Tensor,
    emit_logits: Tensor,
    *,
    silence_token_id: int,
    sampling: TokenSamplingOptions,
    emit_temperature: float,
    generator: torch.Generator,
) -> Tensor:
    """Sample emit/silence independently from the conditional content ID."""
    emit = _sample_emit(
        emit_logits,
        emit_temperature,
        generator=generator,
    )

    content_ids = _sample_content_token_ids(
        logits,
        TokenSamplingOptions(
            mode=sampling.mode,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            suppressed_token_ids=(
                *sampling.suppressed_token_ids,
                silence_token_id,
            ),
        ),
        generator=generator,
    )
    return torch.where(
        emit,
        content_ids,
        torch.full_like(content_ids, silence_token_id),
    )


def _sample_emit(
    emit_logits: Tensor,
    temperature: float,
    *,
    generator: torch.Generator,
) -> Tensor:
    if temperature == 0:
        return emit_logits >= 0
    return torch.bernoulli(
        torch.sigmoid(emit_logits.float() / temperature),
        generator=generator,
    ).bool()


def _sample_tool_call_token_id(
    logits: Tensor,
    emit_logits: Tensor,
    *,
    constraint: ToolCallConstraintState | None,
    silence_token_id: int,
    sampling: TokenSamplingOptions,
    emit_temperature: float,
    generator: torch.Generator,
) -> tuple[Tensor, bool]:
    if constraint is None or not constraint.enabled:
        return logits.new_full((1,), silence_token_id, dtype=torch.long), False

    if not constraint.active:
        emit = constraint.force_next_call or bool(
            _sample_emit(
                emit_logits,
                emit_temperature,
                generator=generator,
            ).item()
        )
        if not emit:
            return logits.new_full((1,), silence_token_id, dtype=torch.long), False
        constraint.begin()

    constrained_logits = logits.clone()
    bitmask = constraint.next_token_bitmask(
        constrained_logits.shape[-1],
        constrained_logits.device,
    )
    xgr.apply_token_bitmask_inplace(
        constrained_logits,
        bitmask,
        vocab_size=constrained_logits.shape[-1],
    )
    token_id = _sample_content_token_ids(
        constrained_logits,
        TokenSamplingOptions(
            mode=sampling.mode,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            suppressed_token_ids=(
                *sampling.suppressed_token_ids,
                silence_token_id,
            ),
        ),
        generator=generator,
    )
    completed = constraint.accept(int(token_id.item()))
    return token_id, completed


def serialize_tool_call(
    tool_call: Mapping[str, Any] | None,
    sequence: int,
    device: torch.device,
) -> Tensor:
    if tool_call is None:
        return torch.empty(0, dtype=torch.uint8, device=device)
    payload = json.dumps(
        {
            "sequence": sequence,
            "name": tool_call["name"],
            "arguments": tool_call["arguments"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _depth_sampling(info: Mapping[str, object]) -> tuple[float, int]:
    duplex = info.get("duplex")
    runtime = duplex.get("runtime_config") if isinstance(duplex, Mapping) else None
    sampling = (
        runtime.get("duplexio_depth_sampling")
        if isinstance(runtime, Mapping)
        else None
    )
    if not isinstance(sampling, Mapping):
        raise ValueError("DuplexIO append is missing depth sampling configuration")
    temperature = sampling.get("temperature")
    top_k = sampling.get("top_k")
    if not isinstance(temperature, (int, float)) or not isinstance(top_k, int):
        raise ValueError("Invalid DuplexIO depth sampling configuration")
    return float(temperature), top_k


def _validate_vllm_runtime_contract(vllm_config: VllmConfig) -> None:
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Native DuplexIO does not support pipeline parallelism")
    if vllm_config.speculative_config is not None:
        raise ValueError("Native DuplexIO does not support speculative decoding")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("Native DuplexIO requires prefix caching to be disabled")
    if (
        vllm_config.parallel_config.decode_context_parallel_size != 1
        or vllm_config.parallel_config.prefill_context_parallel_size != 1
    ):
        raise ValueError("Native DuplexIO does not support context parallelism")
    if vllm_config.parallel_config.use_ubatching:
        raise ValueError("Native DuplexIO does not support microbatching")
    compilation = vllm_config.compilation_config
    if compilation.cudagraph_mode == CUDAGraphMode.FULL:
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("DuplexIO full CUDA graphs require max_num_seqs=1")
        if compilation.cudagraph_capture_sizes != [DUPLEXIO_NUM_CELLS]:
            raise ValueError(
                "DuplexIO full CUDA graphs require the six-cell capture size"
            )


def _fork_generator(generator: torch.Generator) -> torch.Generator:
    fork = torch.Generator(device=generator.device)
    fork.set_state(generator.get_state())
    return fork


__all__ = ["DuplexIOForConditionalGeneration", "DuplexIORequestState"]
