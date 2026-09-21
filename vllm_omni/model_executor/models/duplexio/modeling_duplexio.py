# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM implementation of the DuplexIO full-duplex model."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
import xgrammar as xgr
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torchaudio.transforms import Resample
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_representation import (
    ContinuousAudioRepresentation,
    DelayedMimiRepresentation,
    DelayedMimiState,
    MimiEmbedding,
)
from vllm_omni.model_executor.models.duplexio.checkpoint import (
    resolve_checkpoint_directory,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import (
    DuplexIOConfig,
)
from vllm_omni.model_executor.models.duplexio.depth_sampler import (
    DepthAutoregressiveSampler,
    DepthSamplerConfig,
)
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
    FastConformerRNNT,
    streaming_resample_batch,
)
from vllm_omni.model_executor.models.duplexio.flowmap import FlowMapSampler
from vllm_omni.model_executor.models.duplexio.mimi import (
    MimiModel,
    MimiStreamingState,
)
from vllm_omni.model_executor.models.duplexio.pocket_mimi import (
    ContinuousMimiState,
    PocketMimi,
)
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOQwenModel,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    duplexio_frame_positions,
)
from vllm_omni.model_executor.models.duplexio.sampling_config import ContentPolicy
from vllm_omni.model_executor.models.duplexio.stream_gdn import gdn_cache_dtypes, gdn_cache_shapes
from vllm_omni.model_executor.models.duplexio.text_sampling import TokenSamplingOptions, content_distribution
from vllm_omni.model_executor.models.duplexio.tool_calling import (
    ToolCallConstraintCompiler,
    ToolCallConstraintState,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

TEXT_STREAM_NAMES = ("system", "user", "agent", "tool_call")
OUTPUT_STREAM_NAMES = ("agent", "tool_call")
USER_CELL = 1
AGENT_CELL = 2
TOOL_CALL_CELL = 3
AGENT_AUDIO_CELL = 5


@dataclass
class TextSamplingResult:
    """Sampled text and log probabilities in request order.

    Each text_ids entry contains the user, agent, and tool token IDs.
    """

    text_ids: list[Tensor]
    tool_calls: list[dict[str, Any] | None]
    agent_emit_logprobs: list[Tensor]
    agent_token_logprobs: list[Tensor]
    user_emit_logprobs: list[Tensor]
    user_token_logprobs: list[Tensor]
    tool_emit_logprobs: list[Tensor]  # Raw emit-head scores, including forced decisions.
    tool_token_logprobs: list[Tensor]


@dataclass
class ToolTokenSample:
    token_id: Tensor
    logprob: Tensor


@dataclass
class FramePrediction:
    """One sampled output, before request feedback and optional codec decoding."""

    text_ids: Tensor
    audio: Tensor
    tool_call: dict[str, Any] | None
    # Log probabilities of the agent stream's sampled emit decision and, where it
    # emitted, of its sampled content token, both under the truncated distribution
    # actually drawn from.
    agent_emit_logprob: Tensor
    agent_token_logprob: Tensor
    user_emit: Tensor
    user_emit_logprob: Tensor
    user_token_logprob: Tensor
    tool_emit_logprob: Tensor
    tool_token_logprob: Tensor


@dataclass
class DuplexIORequestState:
    """Request-owned state; Qwen KV/GDN caches belong to the native runner."""

    text_input_ids: Tensor  # CPU feedback, also used for scheduler text counts.
    agent_audio_codes: Tensor
    user_asr: FastConformerAudioStreamState
    input_mimi: ContinuousMimiState | MimiStreamingState
    output_mimi: ContinuousMimiState | MimiStreamingState
    agent_delay: DelayedMimiState | None
    # Raw reference audio; text-only context never advances either encoder.
    voice_prompt: Tensor
    system_token_ids: tuple[int, ...]
    sampling_generator: torch.Generator
    tool_call_constraint: ToolCallConstraintState | None = None
    frames_seen: int = 0
    audio_position: int = 0
    active_text_tokens: int = 0
    tool_call_sequence: int = 0
    sampling: RequestSampling | None = None

    def fork(self) -> DuplexIORequestState:
        """Commit an append only after its model step succeeds."""
        result = copy.copy(self)
        result.sampling_generator = _fork_generator(self.sampling_generator)
        if self.agent_delay is not None:
            result.agent_delay = copy.copy(self.agent_delay)
        if isinstance(self.output_mimi, MimiStreamingState):
            result.output_mimi = self.output_mimi.fork()
            result.input_mimi = self.input_mimi.fork()
        if self.tool_call_constraint is not None:
            result.tool_call_constraint = self.tool_call_constraint.fork()
        # Codec updates are functional; ASR batches own their mutable HF caches.
        return result


@dataclass
class PreparedAudio:
    """One scheduled slice, with audio encoded before per-request framing."""

    state: DuplexIORequestState
    frame_start: int
    frame_count: int
    prompt_count: int
    prompt_chunk_frames: int
    user_features: Tensor
    agent_codes: Tensor


class EmitSamplingTemperatures(BaseModel):
    """Validated emission temperatures received at the engine boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    agent: float = Field(ge=0)
    tool_call: float = Field(ge=0)
    user: float = Field(ge=0)


class DepthSamplingOptions(BaseModel):
    """Categorical audio settings; continuous audio has no depth sampler."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    temperature: float = Field(ge=0)
    top_k: int = Field(ge=0)


@dataclass(frozen=True)
class RequestSampling:
    """Resolved request policy; suppression tensors remain model-owned."""

    source: tuple[object, ...]
    agent: TokenSamplingOptions
    tool: TokenSamplingOptions
    user: TokenSamplingOptions
    emission: EmitSamplingTemperatures
    depth: DepthSamplingOptions | None


class DuplexIOLogitsProcessor(LogitsProcessor):
    """Use the learner's fixed BF16 projection before vLLM's vocabulary gather."""

    def _apply_head(
        self,
        lm_head: ParallelLMHead,
        hidden_states: Tensor,
        embedding_bias: Tensor | None,
    ) -> Tensor:
        return F.linear(hidden_states, lm_head.weight, embedding_bias)


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
        self.channel_emb = nn.Parameter(torch.zeros(len(TEXT_STREAM_NAMES), text_config.hidden_size))
        self.output_head_proj = nn.ModuleDict(
            {name: nn.Linear(text_config.hidden_size, text_config.hidden_size) for name in OUTPUT_STREAM_NAMES}
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
    # The per-frame multimodal metadata carries everything the client needs;
    # skip the per-append hidden-states D2H payload entirely.
    omni_pooler_payload_include_hidden = False
    has_preprocess = True
    decode_query_len = DUPLEXIO_NUM_CELLS
    has_postprocess = True
    postprocess_uses_hidden_states = False
    postprocess_uses_multimodal_outputs = False
    requires_request_sample_eligibility = True
    # The per-cell system/user projections are training auxiliaries. User
    # decisions instead use the trained full-frame projection and emit head.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm.output_head_proj.system.": None,
            "llm.output_head_proj.user.": None,
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        if not isinstance(config, DuplexIOConfig):
            raise TypeError("DuplexIOForConditionalGeneration requires DuplexIOConfig")
        _validate_vllm_runtime_contract(vllm_config)
        self.vllm_config = vllm_config
        self.config = config
        self.policy_version = 0
        self.text_config = vllm_config.model_config.hf_text_config
        self.full_cudagraph_enabled = (
            vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and not vllm_config.model_config.enforce_eager
        )
        # Automatic graph-tree warmup clears cuBLAS workspaces that vLLM's
        # already captured backbone graphs may still reference.
        self.content_distribution = (
            torch.compile(
                content_distribution, fullgraph=True, dynamic=True,
                options={"emulate_precision_casts": True, "triton.cudagraphs": False},
            )
            if self.full_cudagraph_enabled else content_distribution
        )
        self.frame_inputs = (
            torch.compile(frame_inputs, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
            if self.full_cudagraph_enabled else frame_inputs
        )
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("duplexio", "positions"),
            ("duplexio", "key_active"),
            ("duplexio", "text_ordinals"),
            ("duplexio", "text_last"),
            ("duplexio", "audio_first"),
            ("duplexio", "audio_last"),
            ("duplexio", "prompt_ordinal"),
            ("duplexio", "prompt_last"),
            ("duplexio_replay", "text_ids"),
            ("duplexio_replay", "user_features"),
            ("duplexio_replay", "agent_audio"),
            ("duplexio_replay", "audio_mask"),
            ("duplexio_replay", "prompt_frames"),
        }
        self.pad_token_id = config.pad_token_id
        self.silence_token_id = config.silence_token_id
        if self.pad_token_id is None or self.silence_token_id is None:
            raise ValueError("DuplexIO requires exported pad and silence token IDs")

        self.llm = _DuplexIOMultiStreamQwen(
            vllm_config=vllm_config,
            prefix=f"{prefix}.llm" if prefix else "llm",
        )
        # Cell addressing for the backbone's cache, filled once per step and
        # read by every full-attention layer.
        self.frame = self.llm.base_model.model.frame
        hidden_size = self.text_config.hidden_size
        adapter_hidden_size = config.audio_adapter_config.get("hidden_size") or hidden_size
        root = resolve_checkpoint_directory(
            vllm_config.model_config.model,
            revision=vllm_config.model_config.revision,
        )
        self.user_asr = FastConformerRNNT.from_export(
            config.user_asr_config, root, use_cuda_graph=self.full_cudagraph_enabled,
        )
        self.user_audio_resampler = Resample(config.sample_rate, 16_000, dtype=torch.float32).to(
            device=self.llm.channel_emb.device,
        )
        if config.audio_representation == "continuous":
            representation_dim = config.continuous_audio_config["embedding_dim"]
            self.audio_codec = PocketMimi()
            self.audio_representation = ContinuousAudioRepresentation(representation_dim)
            self.agent_audio_embedding = nn.Identity()
            self.audio_sampler = FlowMapSampler(
                representation_dim,
                hidden_size,
                config.flowmap_config["mlp_dim"],
                config.flowmap_config["mlp_depth"],
                inference_steps=config.flowmap_config["inference_steps"],
                sampling_temperature=config.flowmap_config["sampling_temperature"],
                use_cuda_graph=self.full_cudagraph_enabled,
            )
        else:
            quantized = config.quantized_audio_config
            depth = config.depth_transformer_config
            representation_dim = quantized["embedding_dim"]
            self.audio_codec = MimiModel(config.audio_codec_config)
            self.audio_representation = DelayedMimiRepresentation(
                num_codebooks=quantized["num_codebooks"],
                codebook_size=quantized["codebook_size"],
                acoustic_delay_frames=quantized["acoustic_delay_frames"],
            )
            self.agent_audio_embedding = MimiEmbedding(
                quantized["num_codebooks"],
                quantized["codebook_size"],
                representation_dim,
            )
            self.audio_sampler = DepthAutoregressiveSampler(
                DepthSamplerConfig(
                    conditioning_dim=hidden_size,
                    text_vocab_size=self.text_config.vocab_size,
                    codebook_size=quantized["codebook_size"],
                    num_codebooks=quantized["num_codebooks"],
                    low_rank_embeddings=depth["low_rank_embeddings"],
                    dim=depth["dim"],
                    num_layers=depth["num_layers"],
                    num_heads=depth["num_heads"],
                    feedforward_dim=depth["mlp_dim"],
                    sampling_temperature=depth["sampling_temperature"],
                    sampling_top_k=depth["sampling_top_k"],
                    semantic_sampling_top_k=depth.get("semantic_sampling_top_k"),
                )
            )
        self.user_audio_input_adapter = AudioInputAdapter(
            self.user_asr.output_dim,
            adapter_hidden_size,
            hidden_size,
        )
        self.agent_audio_input_adapter = AudioInputAdapter(
            representation_dim,
            adapter_hidden_size,
            hidden_size,
        )
        self.user_token_projection = nn.Linear(DUPLEXIO_NUM_CELLS * hidden_size, hidden_size)
        self.user_emit_head = nn.Linear(DUPLEXIO_NUM_CELLS * hidden_size, 1)
        self.agent_emit_head = nn.Linear(DUPLEXIO_NUM_CELLS * hidden_size, 1)
        self.tool_call_emit_head = nn.Linear(DUPLEXIO_NUM_CELLS * hidden_size, 1)
        self.logits_processor = DuplexIOLogitsProcessor(self.text_config.vocab_size)
        self.make_empty_intermediate_tensors = self.llm.base_model.model.make_empty_intermediate_tensors
        self._forced_next_token_ids: list[int] | None = None
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        agent_suppressed, tool_suppressed = text_suppression_ids(self.tokenizer, self.silence_token_id)
        self.register_buffer(
            "agent_suppressed_token_ids",
            torch.tensor(agent_suppressed, dtype=torch.long, device=self.llm.channel_emb.device),
            persistent=False,
        )
        self.register_buffer(
            "tool_suppressed_token_ids",
            torch.tensor(tool_suppressed, dtype=torch.long, device=self.llm.channel_emb.device),
            persistent=False,
        )
        # Training's conditional user CE excludes only the silence token.
        self.register_buffer(
            "user_suppressed_token_ids",
            torch.tensor([self.silence_token_id], dtype=torch.long, device=self.llm.channel_emb.device),
            persistent=False,
        )
        self.tool_call_compiler = ToolCallConstraintCompiler(self.tokenizer, self.text_config.vocab_size)
        self.set_custom_preprocess(self.preprocess)
        self.set_custom_postprocess(self.postprocess)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.llm.base_model.model.embed_input_ids(input_ids)

    def update_graph_inputs(
        self,
        request_infos: list[dict[str, Any]],
    ) -> None:
        """Install this step's cell addressing at stable device addresses."""
        if not request_infos:
            self.frame.reset()
            return
        duplex_infos = [info["duplexio"] for info in request_infos]
        self.frame.update(
            key_active=torch.cat([info["key_active"] for info in duplex_infos]),
            text_ordinal=torch.cat([info["text_ordinals"] for info in duplex_infos]),
            text_last=torch.cat([info["text_last"] for info in duplex_infos]),
            audio_first=torch.cat([info["audio_first"] for info in duplex_infos]),
            audio_last=torch.cat([info["audio_last"] for info in duplex_infos]),
            prompt_ordinal=torch.cat([info["prompt_ordinal"] for info in duplex_infos]),
            prompt_last=torch.cat([info["prompt_last"] for info in duplex_infos]),
        )
        positions = torch.cat([info["positions"] for info in duplex_infos])
        self.frame.positions[:positions.numel()].copy_(positions)

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

    def prepare_audio_request(self, tokens: int, info: dict[str, Any], device: torch.device) -> PreparedAudio:
        """Validate framing and fork state at the model's input boundary."""
        duplex = info.get("duplex")
        if not isinstance(duplex, Mapping):
            raise ValueError("Native DuplexIO accepts only framed duplex appends")
        append_frames = duplex.get("frame_count")
        token_offset = info["duplex_token_offset"]
        prompt_len = info["duplex_prompt_len"]
        if (
            not isinstance(append_frames, int) or append_frames < 1
            or tokens == 0 or tokens % DUPLEXIO_NUM_CELLS
            or token_offset % DUPLEXIO_NUM_CELLS or prompt_len % DUPLEXIO_NUM_CELLS
        ):
            raise ValueError(
                "Native DuplexIO requires complete six-cell frames; "
                f"got frame_count={append_frames}, tokens={tokens}, offset={token_offset}, end={prompt_len}"
            )
        frame_count = tokens // DUPLEXIO_NUM_CELLS
        frame_start = append_frames - (prompt_len - token_offset) // DUPLEXIO_NUM_CELLS
        frame_end = frame_start + frame_count
        assert 0 <= frame_start < frame_end <= append_frames
        runtime_config = duplex.get("runtime_config")
        if not isinstance(runtime_config, Mapping):
            raise ValueError("DuplexIO append is missing runtime_config")
        is_prefill = duplex.get("duplexio_prefill", False)
        is_system_input = duplex.get("duplexio_system_input", False)
        state = info.get("duplexio_model_state")
        if not isinstance(state, DuplexIORequestState):
            if not is_prefill:
                raise ValueError("DuplexIO requires a complete prefix before live input")
            state = self._new_request_state(runtime_config, device)
        else:
            state = state.fork()

        is_live = not (is_prefill or is_system_input)
        if append_frames != 1 and is_live:
            raise ValueError("Native DuplexIO batches only prefix or tool-context frames")
        prompt_frames = state.voice_prompt.numel() // self.config.frame_size
        prompt_count = prompt_frames if is_prefill else 0
        if is_prefill and (
            state.frames_seen != frame_start or append_frames != prompt_count + len(state.system_token_ids)
        ):
            raise ValueError("DuplexIO prefill must contain the complete speaker and system prefix exactly once")
        prompt_chunk_frames = max(0, min(frame_end, prompt_count) - frame_start)
        return PreparedAudio(
            state, frame_start, frame_count, prompt_count, prompt_chunk_frames,
            self.llm.channel_emb.new_zeros(frame_count, self.user_asr.output_dim),
            state.agent_audio_codes.unsqueeze(0) if is_live else self.initial_agent_audio(frame_count),
        )

    @torch.inference_mode()
    def prepare_audio_requests(
        self, requests: list[tuple[int, dict[str, Any]]], device: torch.device,
    ) -> list[PreparedAudio]:
        prepared = [self.prepare_audio_request(tokens, info, device) for tokens, info in requests]
        acoustic = []
        user_waveforms = []
        prompt_indices = []
        prompt_waveforms = []
        for index, (item, (_, info)) in enumerate(zip(prepared, requests, strict=True)):
            duplex = info["duplex"]
            if not (duplex.get("duplexio_prefill", False) or duplex.get("duplexio_system_input", False)):
                waveform = torch.frombuffer(bytearray(duplex["pcm"]), dtype=torch.float32).to(device)
            elif item.prompt_chunk_frames:
                start = item.frame_start * self.config.frame_size
                count = item.prompt_chunk_frames * self.config.frame_size
                prompt_waveform = item.state.voice_prompt[start:start + count]
                prompt_indices.append(index)
                prompt_waveforms.append(prompt_waveform)
                waveform = torch.zeros_like(prompt_waveform)
            else:
                continue
            acoustic.append(index)
            user_waveforms.append(waveform)
        if acoustic:
            user_features = self.encode_user_audio_batch(
                user_waveforms, [prepared[index].state for index in acoustic],
            )
            for index, features in zip(acoustic, user_features, strict=True):
                prepared[index].user_features = F.pad(
                    features, (0, 0, 0, prepared[index].frame_count - features.shape[0]),
                )
        if prompt_indices:
            states = [prepared[index].state for index in prompt_indices]
            codes = self.encode_agent_audio_batch(prompt_waveforms, states)
            for index, audio in zip(prompt_indices, codes, strict=True):
                prepared[index].agent_codes = torch.cat((audio, prepared[index].agent_codes[audio.shape[0]:]))
        return prepared

    @torch.inference_mode()
    def preprocess_batch(
        self, *, req_ids: list[str], model_intermediate_buffer: dict[str, dict[str, Any]], device: torch.device,
    ) -> dict[str, dict[str, PreparedAudio]]:
        requests = [model_intermediate_buffer[req_id] for req_id in req_ids]
        prepared = self.prepare_audio_requests(
            [(info["_omni_num_scheduled_tokens"], info) for info in requests], device,
        )
        return {req_id: {"prepared_audio": audio} for req_id, audio in zip(req_ids, prepared, strict=True)}

    @torch.inference_mode()
    def preprocess(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        prepared_audio: PreparedAudio | None = None,
        **info: Any,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        del input_embeds
        input_ids = input_ids.to(self.llm.channel_emb.device)
        if prepared_audio is None:
            prepared_audio, = self.prepare_audio_requests([(input_ids.numel(), info)], input_ids.device)
        state = prepared_audio.state
        frame_start, frame_count = prepared_audio.frame_start, prepared_audio.frame_count
        frame_end = frame_start + frame_count
        prompt_count, prompt_chunk_frames = prepared_audio.prompt_count, prepared_audio.prompt_chunk_frames
        user_features, agent_codes = prepared_audio.user_features, prepared_audio.agent_codes
        duplex = info["duplex"]
        runtime_config = duplex["runtime_config"]
        is_prefill = duplex.get("duplexio_prefill", False)
        is_system_input = duplex.get("duplexio_system_input", False)
        is_live = not (is_prefill or is_system_input)
        prompt_written = min(state.frames_seen, state.voice_prompt.numel() // self.config.frame_size)
        prompt_mask = torch.arange(frame_start, frame_end, device=input_ids.device) < prompt_count
        text_ids = state.text_input_ids.expand(frame_count, -1).clone()
        if is_prefill:
            text_ids.fill_(self.silence_token_id)
            text_ids[prompt_chunk_frames:, 0] = torch.tensor(
                state.system_token_ids[max(0, frame_start - prompt_count):max(0, frame_end - prompt_count)],
                dtype=torch.long, device="cpu",
            )
        elif is_system_input:
            system_token_ids = duplex["duplexio_system_token_ids"]
            text_ids.fill_(self.silence_token_id)
            text_ids[:, 0] = torch.tensor(
                system_token_ids[frame_start:frame_end],
                dtype=torch.long,
                device="cpu",
            )
            state.text_input_ids = torch.full_like(
                state.text_input_ids,
                self.silence_token_id,
            )
        else:
            text_ids[:, 0] = self.silence_token_id

        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=self.vllm_config.model_config.dtype,
            enabled=input_ids.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
        ):
            user_hidden = self.user_audio_input_adapter(user_features)
            agent_hidden = self.agent_audio_input_adapter(self.agent_audio_embedding(agent_codes))
            if not is_live:
                user_hidden = user_hidden.masked_fill(~prompt_mask[:, None], 0)
                agent_hidden = agent_hidden.masked_fill(~prompt_mask[:, None], 0)
        active_text_count = ((text_ids != self.pad_token_id) & (text_ids != self.silence_token_id)).sum().item()
        text_ids = text_ids.to(input_ids.device, non_blocking=True)
        text_hidden = self.llm.base_model.model.embed_input_ids(text_ids.flatten()).view(
            frame_count, len(TEXT_STREAM_NAMES), -1
        )
        (
            embeddings,
            key_active,
            text_ordinals,
            text_last,
            audio_first,
            audio_last,
            prompt_ordinal,
            prompt_last,
        ) = self.frame_inputs(
            text_ids, text_hidden, self.llm.channel_emb, user_hidden, agent_hidden,
            self.pad_token_id, self.silence_token_id, state.active_text_tokens,
            state.audio_position, self.config.audio_attention_window_frames,
            is_live, prompt_written, prompt_mask,
        )
        state.active_text_tokens += active_text_count
        positions = torch.arange(
            state.frames_seen * DUPLEXIO_NUM_CELLS,
            (state.frames_seen + frame_count) * DUPLEXIO_NUM_CELLS,
            device=input_ids.device,
        )
        state.frames_seen += frame_count
        # Audio time advances only on frames that carry real audio: text-only
        # prefill and system-token bursts leave the counter frozen so they do
        # not consume the audio attention window.
        if is_live:
            state.audio_position += frame_count
        replay = {}
        if runtime_config.get("duplexio_record_inputs", False):
            replay = {
                "text_ids": text_ids,
                "user_features": user_features,
                "agent_audio": agent_codes,
                "audio_mask": torch.full((frame_count,), is_live, dtype=torch.bool, device=input_ids.device),
                "prompt_frames": prompt_mask,
            }
        return (
            input_ids,
            embeddings,
            {
                "duplexio_working_state": state,
                "duplexio_replay": replay,
                "duplexio": {
                    "positions": positions,
                    "key_active": key_active,
                    "text_ordinals": text_ordinals,
                    "text_last": text_last,
                    "audio_first": audio_first,
                    "audio_last": audio_last,
                    "prompt_ordinal": prompt_ordinal,
                    "prompt_last": prompt_last,
                },
            },
        )

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
        del model_intermediate_buffer
        with torch.profiler.record_function("duplexio.backbone"):
            return self.llm.base_model.model(
                positions=duplexio_frame_positions(self.frame.positions[:inputs_embeds.shape[0]]),
                key_active=self.frame.key_active[: inputs_embeds.shape[0]],
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
            raise RuntimeError(f"DuplexIO received {len(request_token_spans)} spans for {len(infos)} requests")
        raw_request_sample_eligible = kwargs.get("request_sample_eligible")
        if not isinstance(raw_request_sample_eligible, list):
            raise RuntimeError("DuplexIO live execution requires request_sample_eligible")
        request_sample_eligible = cast(
            list[bool],
            raw_request_sample_eligible,
        )
        if len(request_sample_eligible) != len(infos):
            raise RuntimeError(
                f"DuplexIO received {len(request_sample_eligible)} sampling flags for {len(infos)} requests"
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
        retained_tokens: list[int] = []
        position_budgets: list[int] = []
        predictor_hiddens: list[Tensor] = []
        agent_emit_logprobs: list[Tensor] = []
        agent_token_logprobs: list[Tensor] = []
        tool_emit_logprobs: list[Tensor] = []
        tool_token_logprobs: list[Tensor] = []
        user_outputs: dict[str, list[Tensor]] = {
            name: [] for name in (
                "user_emit", "user_emit_logprob", "user_token_logprob",
                "policy_version",
            )
        }
        replay_outputs: dict[str, list[Tensor]] = {
            f"replay_{name}": [] for name in ("text_ids", "user_features", "agent_audio", "audio_mask", "prompt_frames")
        }
        predictions = self.sample_frames(hidden_states, request_token_spans, infos, request_sample_eligible)
        decode_indices = [index for index in predictions if infos[index]["duplex"].get("decode_audio", True)]
        waveforms = self.decode_agent_audio_batch(
            [predictions[index].audio for index in decode_indices],
            [infos[index]["duplexio_working_state"] for index in decode_indices],
        )
        decoded = dict(zip(decode_indices, waveforms, strict=True))
        sampled_ids = dict(zip(
            predictions,
            torch.stack([prediction.text_ids for prediction in predictions.values()]).tolist() if predictions else [],
            strict=True,
        ))
        for request_index, ((start, end), info) in enumerate(zip(request_token_spans, infos, strict=True)):
            state = info.get("duplexio_working_state")
            if not isinstance(state, DuplexIORequestState):
                raise RuntimeError(f"DuplexIO request {request_index} is missing working state")
            # Four text slots per scheduler row; audio and voice KV have fixed
            # reserved regions. Keep one row to preserve the recurrent-state marker.
            retained_tokens.append(max(1, (state.active_text_tokens + 3) // 4) * DUPLEXIO_NUM_CELLS)
            position_budgets.append(
                (self.text_config.max_position_embeddings - state.frames_seen) * DUPLEXIO_NUM_CELLS
            )
            span_length = end - start
            if span_length < DUPLEXIO_NUM_CELLS or span_length % DUPLEXIO_NUM_CELLS:
                raise ValueError(f"DuplexIO request span must contain complete frames, got ({start}, {end})")
            duplex = info.get("duplex", {})
            if not isinstance(duplex, Mapping):
                raise RuntimeError(f"DuplexIO request {request_index} is missing duplex metadata")
            is_prefill = duplex.get("duplexio_prefill", False)
            is_system_input = duplex.get("duplexio_system_input", False)
            row_hidden = hidden_states[end - DUPLEXIO_NUM_CELLS : end]
            prediction = predictions.get(request_index)
            predicting = prediction is not None
            end_flags.append(torch.tensor([duplex.get("final", False)]))
            epochs.append(torch.tensor([duplex.get("epoch", 0)]))
            turn_ids.append(torch.tensor([duplex.get("turn_id", 0)]))
            prefill_flags.append(torch.tensor([is_prefill and not predicting]))
            prefill_complete_flags.append(torch.tensor([is_prefill and predicting]))
            system_input_flags.append(torch.tensor([is_system_input and not predicting]))
            system_input_complete_flags.append(torch.tensor([is_system_input and predicting]))
            record_hiddens = duplex["runtime_config"].get("duplexio_record_hiddens", False)
            predictor_hiddens.append(
                row_hidden.detach() if record_hiddens and prediction is not None else row_hidden.new_empty(0)
            )
            # Return sampling probabilities alongside each prediction.
            agent_emit_logprobs.append(
                prediction.agent_emit_logprob if prediction is not None else row_hidden.new_empty(0)
            )
            agent_token_logprobs.append(
                prediction.agent_token_logprob if prediction is not None else row_hidden.new_empty(0)
            )
            tool_emit_logprobs.append(
                prediction.tool_emit_logprob if prediction is not None else row_hidden.new_empty(0)
            )
            tool_token_logprobs.append(
                prediction.tool_token_logprob if prediction is not None else row_hidden.new_empty(0)
            )
            user_values = {}
            if prediction is not None:
                user_values = {
                    "user_emit": prediction.user_emit,
                    "user_emit_logprob": prediction.user_emit_logprob,
                    "user_token_logprob": prediction.user_token_logprob,
                    "policy_version": torch.tensor([self.policy_version], dtype=torch.long, device=row_hidden.device),
                }
            for name, values in user_outputs.items():
                values.append(user_values.get(name, row_hidden.new_empty(0)))
            replay = info.get("duplexio_replay", {})
            for name, values in replay_outputs.items():
                values.append(replay.get(name.removeprefix("replay_"), row_hidden.new_empty(0)))
            if prediction is None:
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
                tool_call_complete_flags.append(torch.tensor([False]))
                tool_call_payloads.append(row_hidden.new_empty(0, dtype=torch.uint8))
                continue
            predicted_audio, tool_call = prediction.audio, prediction.tool_call
            _, agent_token_id, tool_token_id = sampled_ids[request_index]
            state.text_input_ids = torch.tensor(
                [self.silence_token_id, *sampled_ids[request_index]], dtype=torch.long, device="cpu",
            )
            state.agent_audio_codes = predicted_audio
            waveform = decoded.get(request_index)
            if waveform is None:
                waveform = row_hidden.new_empty(0, dtype=torch.float32)
            forced_ids.append(agent_token_id)

            user_ids.append(state.text_input_ids[1:2])
            agent_ids.append(state.text_input_ids[2:3])
            tool_ids.append(state.text_input_ids[3:4])
            audio_outputs.append(waveform.detach())
            audio_token_ids.append(predicted_audio.detach())
            model_listen = agent_token_id == self.silence_token_id and tool_token_id == self.silence_token_id
            listen_flags.append(torch.tensor([model_listen]))
            tool_call_complete_flags.append(torch.tensor([tool_call is not None]))
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
                "chunk": {
                    "agent_audio_token_ids": audio_token_ids,
                    "sample_rate_hz": [torch.tensor([self.config.sample_rate])] * len(audio_outputs),
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
                    "predictor_hiddens": predictor_hiddens,
                    "agent_emit_logprob": agent_emit_logprobs,
                    "agent_token_logprob": agent_token_logprobs,
                    "tool_emit_logprob": tool_emit_logprobs,
                    "tool_token_logprob": tool_token_logprobs,
                    **user_outputs,
                    **replay_outputs,
                },
            },
        )
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
            streaming_retained_tokens=retained_tokens,
            streaming_position_budget=position_budgets,
        )

    def sample_frames(
        self,
        hidden_states: Tensor,
        request_token_spans: list[tuple[int, int]],
        infos: list[dict[str, Any]],
        request_sample_eligible: list[bool],
    ) -> dict[int, FramePrediction]:
        """Batch deterministic heads while preserving each request's RNG order."""
        indices = [index for index, eligible in enumerate(request_sample_eligible) if eligible]
        if not indices:
            return {}
        ends = [request_token_spans[index][1] for index in indices]
        rows = torch.stack([hidden_states[end - DUPLEXIO_NUM_CELLS : end] for end in ends])
        with torch.profiler.record_function("duplexio.text_projection"):
            text_logits, emit_logits = self.project_text(rows)
        with torch.profiler.record_function("duplexio.text_sampling"):
            sample_infos = [infos[index] for index in indices]
            text_samples = self.sample_text_batch(text_logits, emit_logits, sample_infos)
        noises: list[Tensor] = []
        depth_audio: list[Tensor] = []
        continuous = isinstance(self.audio_sampler, FlowMapSampler)
        for row, index in enumerate(indices):
            info = infos[index]
            state = info["duplexio_working_state"]
            if continuous:
                noises.append(torch.randn(
                    1, self.audio_representation.embedding_dim, device=rows.device,
                    dtype=torch.float32, generator=state.sampling_generator,
                ))
            else:
                depth = state.sampling.depth
                assert depth is not None
                with torch.autocast(
                    rows.device.type, dtype=self.vllm_config.model_config.dtype,
                    enabled=rows.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
                ):
                    depth_audio.append(self.audio_sampler.sample(
                        rows[row, AGENT_AUDIO_CELL : AGENT_AUDIO_CELL + 1], text_samples.text_ids[row][1:2],
                        temperature=depth.temperature, top_k=depth.top_k,
                        generator=state.sampling_generator,
                    )[0])
        if continuous:
            with torch.profiler.record_function("duplexio.audio_sampling"), torch.autocast(
                rows.device.type, dtype=self.vllm_config.model_config.dtype,
                enabled=rows.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
            ):
                audio = self.audio_sampler.sample(rows[:, AGENT_AUDIO_CELL].float(), torch.cat(noises)).unbind(0)
        else:
            audio = depth_audio
        return {
            index: FramePrediction(
                text_ids=text_samples.text_ids[row],
                audio=audio[row],
                tool_call=text_samples.tool_calls[row],
                agent_emit_logprob=text_samples.agent_emit_logprobs[row],
                agent_token_logprob=text_samples.agent_token_logprobs[row],
                user_emit=text_samples.text_ids[row][:1] != self.silence_token_id,
                user_emit_logprob=text_samples.user_emit_logprobs[row],
                user_token_logprob=text_samples.user_token_logprobs[row],
                tool_emit_logprob=text_samples.tool_emit_logprobs[row],
                tool_token_logprob=text_samples.tool_token_logprobs[row],
            )
            for row, index in enumerate(indices)
        }

    def project_text(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Project the entire request batch before any CPU-side tool decisions."""
        projected = torch.cat(
            (
                self.llm.output_head_proj["agent"](rows[:, AGENT_CELL]),
                self.llm.output_head_proj["tool_call"](rows[:, TOOL_CALL_CELL]),
                self.user_token_projection(rows.flatten(1)),
            )
        )
        logits = self.logits_processor(self.llm.base_model.lm_head, projected)
        logits = logits.view(3, rows.shape[0], -1).transpose(0, 1)
        full_frames = rows.flatten(1)
        emit_logits = torch.cat(
            (
                self.agent_emit_head(full_frames), self.tool_call_emit_head(full_frames),
                self.user_emit_head(full_frames),
            ),
            dim=-1,
        )
        return logits, emit_logits

    def sample_text_batch(
        self,
        logits: Tensor,
        emit_logits: Tensor,
        infos: list[dict[str, Any]],
    ) -> TextSamplingResult:
        """Sample agent/tool/user decisions with request-owned random generators."""
        for info in infos:
            state = info["duplexio_working_state"]
            runtime = info["duplex"]["runtime_config"]
            if state.sampling is None or state.sampling.source != sampling_source(runtime):
                state.sampling = self.resolve_sampling(runtime)
        agent_ids, agent_emit_logprobs, agent_token_logprobs = self.sample_agent_tokens(
            logits[:, 0], emit_logits[:, 0], infos
        )
        samplings: list[TokenSamplingOptions] = []
        tool_starts = [False] * len(infos)
        tool_token_logprobs = [emit_logits.new_zeros(1, dtype=torch.float32) for _ in infos]
        pending_indices: list[int] = []
        pending_starts: list[Tensor] = []
        for row, info in enumerate(infos):
            state = info["duplexio_working_state"]
            samplings.append(state.sampling.tool)
            temperatures = state.sampling.emission
            constraint = state.tool_call_constraint
            if constraint is not None and constraint.enabled and not constraint.active:
                if constraint.force_next_call:
                    tool_starts[row] = True
                else:
                    pending_indices.append(row)
                    pending_starts.append(_sample_emit(
                        emit_logits[row, 1:2],
                        temperatures.tool_call,
                        generator=state.sampling_generator,
                    ))
        if pending_starts:
            for row, start in zip(pending_indices, torch.cat(pending_starts).tolist(), strict=True):
                tool_starts[row] = start

        texts: list[Tensor] = []
        calls: list[dict[str, Any] | None] = [None] * len(infos)
        token_indices: list[int] = []
        tool_tokens: list[Tensor] = []
        for row, info in enumerate(infos):
            state = info["duplexio_working_state"]
            tool_sample = sample_tool_token(
                logits[row, 1:2], constraint=state.tool_call_constraint,
                emit=tool_starts[row],
                sampling=samplings[row], generator=state.sampling_generator,
                distribution=self.content_distribution,
            )
            if tool_sample is None:
                tool_id = logits.new_full((1,), self.silence_token_id, dtype=torch.long)
            else:
                tool_id = tool_sample.token_id
                tool_token_logprobs[row] = tool_sample.logprob
                token_indices.append(row)
                tool_tokens.append(tool_id)
            texts.append(torch.cat((agent_ids[row], tool_id)))
        # Score the chosen decision with the raw head, even when serving forced it.
        tool_emitted = torch.stack([text[-1] for text in texts]) != self.silence_token_id
        tool_logits = emit_logits[:, 1].float()
        tool_emit_logprobs = list(F.logsigmoid(torch.where(tool_emitted, tool_logits, -tool_logits)).split(1))
        if tool_tokens:
            for row, token_id in zip(token_indices, torch.cat(tool_tokens).tolist(), strict=True):
                constraint = infos[row]["duplexio_working_state"].tool_call_constraint
                if constraint.accept(token_id):
                    calls[row] = self.tool_call_compiler.take_completed_call(constraint)
        user_ids, user_emit_logprobs, user_token_logprobs = self.sample_stream_tokens(
            logits[:, 2], emit_logits[:, 2], infos,
            stream="user",
        )
        texts = [torch.cat((user, text)) for user, text in zip(user_ids, texts, strict=True)]
        return TextSamplingResult(
            text_ids=texts,
            tool_calls=calls,
            agent_emit_logprobs=agent_emit_logprobs,
            agent_token_logprobs=agent_token_logprobs,
            user_emit_logprobs=user_emit_logprobs,
            user_token_logprobs=user_token_logprobs,
            tool_emit_logprobs=tool_emit_logprobs,
            tool_token_logprobs=tool_token_logprobs,
        )

    def sample_agent_tokens(
        self, logits: Tensor, emit_logits: Tensor, infos: list[dict[str, Any]]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        return self.sample_stream_tokens(
            logits, emit_logits, infos, stream="agent",
        )

    def sample_stream_tokens(
        self, logits: Tensor, emit_logits: Tensor, infos: list[dict[str, Any]],
        *, stream: str,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Filter equal-policy requests together; keep their random draws independent.

        Returns the sampled ids and, per row, the log probability of the emit decision
        and of the content draw. Both come from the distributions already materialized
        for sampling, so nothing is recomputed and no draw order changes.
        """
        policies = [info["duplexio_working_state"].sampling for info in infos]
        samplings = [policy.user if stream == "user" else policy.agent for policy in policies]
        groups: dict[tuple[float, int | None, float | None, float], list[int]] = {}
        for row, sampling in enumerate(samplings):
            emission = policies[row].emission
            emit_temperature = emission.user if stream == "user" else emission.agent
            key = sampling.temperature, sampling.top_k, sampling.top_p, emit_temperature
            groups.setdefault(key, []).append(row)
        tokens: dict[int, Tensor] = {}
        emit_logprobs: dict[int, Tensor] = {}
        token_logprobs: dict[int, Tensor] = {}
        for key, rows in groups.items():
            emit_temperature = key[3]
            sampling = samplings[rows[0]]
            group_logits = torch.stack([logits[row] for row in rows])
            group_emit_logits = emit_logits[rows].float()
            generators = {
                index: infos[row]["duplexio_working_state"].sampling_generator
                for index, row in enumerate(rows)
            }
            if emit_temperature == 0:
                emitted = group_emit_logits >= 0
                emission_logprobs = torch.zeros_like(group_emit_logits)
            else:
                probability = torch.sigmoid(group_emit_logits / emit_temperature)
                emitted = torch.cat([
                    torch.bernoulli(probability[index:index + 1], generator=generator)
                    for index, generator in generators.items()
                ]).bool()
                emission_logprobs = torch.where(emitted, probability, 1 - probability).log()
            greedy = sampling.temperature == 0
            if greedy:
                content = _sample_content_token_ids(
                    group_logits, sampling, generator=generators[0],
                )
                content_logprobs = torch.zeros_like(group_emit_logits)
            else:
                indices, probabilities = self.content_distribution(group_logits, sampling)
                # vLLM's sampler avoids multinomial's validation/synchronization.
                # It mutates probabilities; retain the distribution for logprobs.
                selected = random_sample(probabilities.clone(), generators).unsqueeze(-1)
                content = indices.gather(-1, selected).squeeze(-1)
                content_logprobs = probabilities.gather(-1, selected).squeeze(-1).log()
            content = torch.where(emitted, content, self.silence_token_id)
            # A discarded content draw on a wait frame is not an action.
            content_logprobs = torch.where(emitted, content_logprobs, 0)
            for index, row in enumerate(rows):
                tokens[row] = content[index:index + 1]
                emit_logprobs[row] = emission_logprobs[index:index + 1]
                token_logprobs[row] = content_logprobs[index:index + 1]
        order = range(len(infos))
        return (
            [tokens[row] for row in order],
            [emit_logprobs[row] for row in order],
            [token_logprobs[row] for row in order],
        )

    def resolve_sampling(self, runtime: Mapping[str, Any]) -> RequestSampling:
        """Validate new or updated wire settings once before sampling a request."""
        source = sampling_source(runtime)
        agent_config, user_config, emission_config, depth_config = source
        agent = ContentPolicy.model_validate(agent_config)
        user = ContentPolicy.model_validate(user_config)
        return RequestSampling(
            source=copy.deepcopy(source),
            agent=TokenSamplingOptions(agent.temperature, agent.top_k, agent.top_p, self.agent_suppressed_token_ids),
            tool=TokenSamplingOptions(agent.temperature, agent.top_k, agent.top_p, self.tool_suppressed_token_ids),
            user=TokenSamplingOptions(user.temperature, user.top_k, user.top_p, self.user_suppressed_token_ids),
            emission=EmitSamplingTemperatures.model_validate(emission_config),
            depth=DepthSamplingOptions.model_validate(depth_config) if depth_config is not None else None,
        )

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
                f"DuplexIO forced-token batch mismatch: {len(token_ids)} ids for {hidden_states.shape[0]} rows"
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
        reference = runtime_config["duplexio_voice_prompt_pcm"]
        prompt_frames = runtime_config["duplexio_voice_prompt_frames"]
        assert isinstance(reference, bytes)
        assert isinstance(prompt_frames, int) and 1 <= prompt_frames <= self.config.voice_prompt_max_frames
        prompt_bytes = prompt_frames * self.config.frame_size * 4
        assert len(reference) >= prompt_bytes and len(reference) % 4 == 0
        samples = torch.frombuffer(bytearray(reference[:prompt_bytes]), dtype=torch.float32).to(device)
        system_tokens = runtime_config.get("duplexio_system_token_ids", ())
        if not isinstance(system_tokens, (list, tuple)) or not all(isinstance(token, int) for token in system_tokens):
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
        if not isinstance(tools, list) or not all(isinstance(tool, Mapping) for tool in tools):
            raise ValueError("duplexio_tools must be a list of tool definitions")
        if not isinstance(tool_choice, Mapping):
            raise ValueError("duplexio_tool_choice must be an object")
        continuous = isinstance(self.audio_codec, PocketMimi)
        agent_delay = (
            None if continuous else self.audio_representation.new_state(device=device)
        )
        return DuplexIORequestState(
            text_input_ids=torch.full(
                (len(TEXT_STREAM_NAMES),),
                self.silence_token_id,
                dtype=torch.long,
                device="cpu",
            ),
            agent_audio_codes=self.initial_agent_audio(1)[0],
            user_asr=FastConformerAudioStreamState(),
            input_mimi=self.audio_codec.new_state(1) if continuous else self.audio_codec.new_streaming_state(),
            agent_delay=agent_delay,
            output_mimi=(self.audio_codec.new_state(1) if continuous else self.audio_codec.new_streaming_state()),
            voice_prompt=samples,
            system_token_ids=cast(tuple[int, ...], tuple(system_tokens)),
            sampling_generator=sampling_generator,
            tool_call_constraint=self.tool_call_compiler.new_state(
                cast(list[Mapping[str, Any]], tools),
                tool_choice,
            ),
        )

    @torch.inference_mode()
    def encode_user_audio(self, waveform: Tensor, state: DuplexIORequestState) -> Tensor:
        """Consume acoustic time only, retaining state across text-only bursts."""
        return self.encode_user_audio_batch([waveform], [state])[0]

    def encode_user_audio_batch(self, waveforms: list[Tensor], states: list[DuplexIORequestState]) -> list[Tensor]:
        with torch.profiler.record_function("duplexio.asr_resample"):
            resampled, tails = streaming_resample_batch(
                waveforms, [state.user_asr.resample_tail for state in states], self.user_audio_resampler,
            )
        encoded, caches = self.user_asr.encode_audio_batch(
            resampled, [state.user_asr for state in states],
        )
        for state, cache, tail in zip(states, caches, tails, strict=True):
            state.user_asr = cache
            cache.resample_tail = tail
        return [value[0] for value in encoded]

    def encode_agent_audio_batch(self, waveforms: list[Tensor], states: list[DuplexIORequestState]) -> list[Tensor]:
        if isinstance(self.audio_codec, PocketMimi):
            with torch.autocast(
                device_type=waveforms[0].device.type, dtype=self.vllm_config.model_config.dtype,
                enabled=waveforms[0].is_cuda and self.vllm_config.model_config.dtype != torch.float32,
            ):
                encoded, caches = self.audio_codec.encode_batch(
                    [waveform[None, None] for waveform in waveforms], [state.input_mimi for state in states],
                )
            for state, cache in zip(states, caches, strict=True):
                state.input_mimi = cache
            return [self.audio_representation.normalize(value[0].T) for value in encoded]
        outputs = []
        for waveform, state in zip(waveforms, states, strict=True):
            codes, state.input_mimi = self.encode_agent_audio(waveform, state.input_mimi, state.agent_delay)
            outputs.append(codes)
        return outputs

    def decode_agent_audio_batch(self, audio: list[Tensor], states: list[DuplexIORequestState]) -> list[Tensor]:
        if not audio:
            return []
        if isinstance(self.audio_codec, PocketMimi):
            with torch.profiler.record_function("duplexio.output_codec_decode"), torch.autocast(
                device_type=audio[0].device.type, dtype=self.vllm_config.model_config.dtype,
                enabled=audio[0].is_cuda and self.vllm_config.model_config.dtype != torch.float32,
            ):
                latents = self.audio_representation.denormalize(torch.stack(audio))
                decoded, caches = self.audio_codec.decode_batch(
                    [latent[None, :, None] for latent in latents], [state.output_mimi for state in states],
                )
            for state, cache in zip(states, caches, strict=True):
                state.output_mimi = cache
            return [value[0, 0] for value in decoded]
        outputs = []
        for value, state in zip(audio, states, strict=True):
            waveform, state.output_mimi = self.decode_agent_audio(value, state.output_mimi, state.agent_delay)
            outputs.append(waveform)
        return outputs

    @torch.inference_mode()
    def encode_agent_audio(
        self,
        waveform: Tensor,
        codec_state: ContinuousMimiState | MimiStreamingState,
        delay: DelayedMimiState | None,
    ) -> tuple[Tensor, ContinuousMimiState | MimiStreamingState]:
        """Encode the next waveform rows without resetting the conversation."""
        with torch.autocast(
            device_type=waveform.device.type,
            dtype=self.vllm_config.model_config.dtype,
            enabled=waveform.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
        ):
            batched = waveform.reshape(1, 1, -1)
            if isinstance(self.audio_codec, PocketMimi):
                latent, codec_state = self.audio_codec.encode(batched, codec_state)
                return self.audio_representation.normalize(latent[0].transpose(0, 1)), codec_state
            assert delay is not None
            raw_codes = self.audio_codec.encode(
                batched, self.audio_representation.num_codebooks, codec_state,
            )
            return self.audio_representation.encode_sequence(raw_codes[0].transpose(0, 1), delay), codec_state

    @torch.inference_mode()
    def decode_agent_audio(
        self,
        audio: Tensor,
        codec_state: ContinuousMimiState | MimiStreamingState,
        delay: DelayedMimiState | None,
    ) -> tuple[Tensor, ContinuousMimiState | MimiStreamingState]:
        """Decode one prediction, retaining codec history and quantized delay."""
        with (
            torch.profiler.record_function("duplexio.output_codec_decode"),
            torch.autocast(
                device_type=audio.device.type,
                dtype=self.vllm_config.model_config.dtype,
                enabled=audio.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
            ),
        ):
            if isinstance(self.audio_codec, PocketMimi):
                latent = self.audio_representation.denormalize(audio)
                decoded, codec_state = self.audio_codec.decode(latent[None, :, None], codec_state)
                return decoded[0, 0], codec_state
            raw_codes = self.audio_representation.decode_column(audio, delay)
            if raw_codes is None:
                return audio.new_empty(0, dtype=torch.float32), codec_state
            return self.audio_codec.decode(raw_codes[None, :, None], codec_state)[0, 0], codec_state

    def initial_agent_audio(self, frames: int) -> Tensor:
        """Placeholder before the first prediction and on text-only rows."""
        if isinstance(self.audio_representation, ContinuousAudioRepresentation):
            return self.llm.channel_emb.new_zeros(frames, self.audio_representation.embedding_dim)
        return torch.full(
            (frames, self.audio_representation.num_codebooks),
            self.audio_representation.codebook_size,
            dtype=torch.long,
            device=self.llm.channel_emb.device,
        )

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return gdn_cache_dtypes(vllm_config.model_config.dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        text_config = vllm_config.model_config.hf_text_config
        return gdn_cache_shapes(
            vllm_config.parallel_config.tensor_parallel_size,
            text_config.linear_num_key_heads,
            text_config.linear_num_value_heads,
            text_config.linear_key_head_dim,
            text_config.linear_value_head_dim,
            text_config.linear_conv_kernel_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, ...]:
        return (get_conv_copy_spec,) + (get_temporal_copy_spec,) * 6


def text_suppression_ids(tokenizer: TokenizerLike, silence_token_id: int) -> tuple[list[int], list[int]]:
    """Derive immutable vocabulary policy for the agent and tool heads."""
    tool_ids = set(tokenizer.all_special_ids) | {silence_token_id}
    for text in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            tool_ids.add(encoded[0])
    agent_ids = tool_ids.copy()
    for text in ("<tool_call>", "</tool_call>"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            agent_ids.add(encoded[0])
    return sorted(agent_ids), sorted(tool_ids)


def frame_inputs(
    text_ids: Tensor,
    text_hidden: Tensor,
    channel_embedding: Tensor,
    user_hidden: Tensor,
    agent_hidden: Tensor,
    pad_token_id: int,
    silence_token_id: int,
    active_text_tokens: int,
    audio_position: int,
    audio_window_frames: int,
    audio_active: bool,
    prompt_position: int,
    prompt_frames: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Assemble six-cell inputs and cache addressing without changing CPU state.

    Besides the flattened embeddings this returns one entry per cell: whether the
    cell contributes a key, its 1-based text emission ordinal (0 when it emits
    nothing), how many text keys strictly earlier rows emitted, and the inclusive
    range of audio frames the cell may attend. Audio frames are numbered from
    one in audio time, which only advances on frames carrying real audio, so a
    text-only append leaves the range frozen and writes no audio key.

    A pinned voice-prompt burst is the one frame kind that carries audio without
    being live: its audio cells contribute keys, but audio time stays frozen and
    the keys land in the cache's pinned region, which no window expires.
    """
    frames, device = text_ids.shape[0], text_ids.device
    text_hidden = text_hidden.masked_fill((text_ids == silence_token_id).unsqueeze(-1), 0)
    embeddings = torch.cat(
        (text_hidden + channel_embedding, user_hidden.unsqueeze(1), agent_hidden.unsqueeze(1)), dim=1,
    ).flatten(0, 1)
    text_active = (text_ids != pad_token_id) & (text_ids != silence_token_id)
    audio_shape = (frames, 2)
    audio_keyed = (prompt_frames | audio_active)[:, None].expand(-1, 2)
    key_active = torch.cat(
        (text_active, audio_keyed), dim=1,
    ).flatten()
    ordinals = (
        text_active.flatten().cumsum(0, dtype=torch.int32) + active_text_tokens
    ).view_as(text_active)
    text_ordinals = torch.cat(
        (torch.where(text_active, ordinals, 0), torch.zeros(audio_shape, dtype=torch.int32, device=device)),
        dim=1,
    ).flatten()
    # A row sees the text its predecessors emitted, never its own siblings'.
    row_emitted = text_active.sum(1, dtype=torch.int32)
    text_last = row_emitted.cumsum(0) - row_emitted + active_text_tokens
    rows = torch.arange(frames, dtype=torch.int32, device=device)
    audio_last = audio_position + (rows if audio_active else torch.zeros_like(rows))
    audio_first = (audio_last + int(audio_active) - audio_window_frames).clamp_min(1)
    # A prompt row's own pinned key is its self key, which the mask merges
    # separately, so a row sees only the prompt frames written before it.
    prompt_total = prompt_frames.cumsum(0, dtype=torch.int32) + prompt_position
    prompt_ordinal_rows = torch.where(prompt_frames, prompt_total, 0)
    prompt_last_rows = torch.where(prompt_frames, prompt_total - 1, prompt_total)
    prompt_ordinal = torch.cat(
        (
            torch.zeros(
                (frames, DUPLEXIO_NUM_TEXT_CELLS), dtype=torch.int32, device=device
            ),
            prompt_ordinal_rows[:, None].expand(-1, 2),
        ),
        dim=1,
    ).flatten()

    def per_cell(values: Tensor) -> Tensor:
        """Give every cell of a row the row's value."""
        return values[:, None].expand(-1, DUPLEXIO_NUM_CELLS).flatten()

    return (
        embeddings,
        key_active,
        text_ordinals,
        per_cell(text_last),
        per_cell(audio_first),
        per_cell(audio_last),
        prompt_ordinal,
        per_cell(prompt_last_rows),
    )


def sampling_source(runtime: Mapping[str, Any]) -> tuple[object, ...]:
    """Select only policy fields; per-frame metadata must not invalidate the cache."""
    return (
        runtime["duplexio_text_sampling"],
        runtime["duplexio_user_sampling"]["content"],
        runtime["duplexio_emit_temperatures"],
        runtime.get("duplexio_depth_sampling"),
    )


def _sample_content_token_ids(
    logits: Tensor,
    sampling: TokenSamplingOptions,
    *,
    generator: torch.Generator,
    distribution: Callable[[Tensor, TokenSamplingOptions], tuple[Tensor, Tensor]] = content_distribution,
) -> Tensor:
    if sampling.temperature == 0:
        if sampling.suppressed_token_ids.numel():
            logits = logits.clone()
            logits.index_fill_(-1, sampling.suppressed_token_ids, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)
    top_indices, probabilities = distribution(logits, sampling)
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
    distribution: Callable[[Tensor, TokenSamplingOptions], tuple[Tensor, Tensor]] = content_distribution,
) -> Tensor:
    """Sample emit/silence independently from the conditional content ID."""
    emit = _sample_emit(
        emit_logits,
        emit_temperature,
        generator=generator,
    )

    content_ids = _sample_content_token_ids(
        logits,
        sampling,
        generator=generator,
        distribution=distribution,
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


def sample_tool_token(
    logits: Tensor,
    *,
    constraint: ToolCallConstraintState | None,
    emit: bool,
    sampling: TokenSamplingOptions,
    generator: torch.Generator,
    distribution: Callable[[Tensor, TokenSamplingOptions], tuple[Tensor, Tensor]] = content_distribution,
) -> ToolTokenSample | None:
    """Sample and score the grammar-constrained distribution in one draw."""
    if constraint is None or not constraint.enabled:
        return None

    if not constraint.active:
        if not emit:
            return None
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
    if sampling.temperature == 0:
        token_id = _sample_content_token_ids(constrained_logits, sampling, generator=generator)
        return ToolTokenSample(token_id=token_id, logprob=torch.zeros_like(token_id, dtype=torch.float32))
    indices, probabilities = distribution(constrained_logits, sampling)
    selected = random_sample(probabilities.clone(), {0: generator}).unsqueeze(-1)
    return ToolTokenSample(
        token_id=indices.gather(-1, selected).squeeze(-1),
        logprob=probabilities.gather(-1, selected).squeeze(-1).float().log(),
    )


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


def _validate_vllm_runtime_contract(vllm_config: VllmConfig) -> None:
    if vllm_config.quant_config is not None:
        raise ValueError("DuplexIO requires unquantized backbone weights to preserve training's fused MLP math")
    if vllm_config.model_config.head_dtype not in (None, vllm_config.model_config.dtype):
        raise ValueError("DuplexIO vocabulary projection must retain the training model dtype")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Native DuplexIO does not support pipeline parallelism")
    if vllm_config.speculative_config is not None:
        raise ValueError("Native DuplexIO does not support speculative decoding")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("Native DuplexIO requires prefix caching to be disabled")
    scheduler = vllm_config.scheduler_config
    # All appends contain whole frames. Aligned budgets preserve that invariant
    # when the scheduler splits a prefill or mixes it with live requests.
    if (
        scheduler.max_num_batched_tokens % DUPLEXIO_NUM_CELLS
        or scheduler.long_prefill_token_threshold % DUPLEXIO_NUM_CELLS
    ):
        raise ValueError("DuplexIO scheduler token budgets must be multiples of six cells")
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
            raise ValueError("Batched DuplexIO CUDA graphs require FULL_DECODE_ONLY")
        if compilation.cudagraph_capture_sizes != [DUPLEXIO_NUM_CELLS]:
            raise ValueError("DuplexIO full CUDA graphs require the six-cell capture size")
    elif compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
        sizes = [DUPLEXIO_NUM_CELLS * count for count in range(1, vllm_config.scheduler_config.max_num_seqs + 1)]
        if compilation.cudagraph_capture_sizes != sizes:
            raise ValueError("DuplexIO decode graphs require one exact capture size per request count")


def _fork_generator(generator: torch.Generator) -> torch.Generator:
    fork = torch.Generator(device=generator.device)
    fork.set_state(generator.get_state())
    return fork


__all__ = ["DuplexIOForConditionalGeneration", "DuplexIORequestState"]
