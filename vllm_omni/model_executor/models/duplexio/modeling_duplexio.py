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
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.v1.sample.metadata import SamplingMetadata

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
    """Sampled text and log probabilities, batched in request order.

    Each text_ids row holds the user, agent, and tool token IDs. Log probabilities
    are ``[rows, 1]``, so one row is a request's output value.
    """

    text_ids: Tensor
    tool_calls: list[dict[str, Any] | None]
    agent_emit_logprobs: Tensor
    agent_token_logprobs: Tensor
    user_emit_logprobs: Tensor
    user_token_logprobs: Tensor
    tool_emit_logprobs: Tensor  # Raw emit-head scores, including forced decisions.
    tool_token_logprobs: Tensor
    # Start draws; the host reads them only for idle rows that may start a call.
    tool_starts: Tensor
    pending_tool_starts: list[int]
    # Rows already inside a call, whose sampled token the host must accept.
    tool_rows: list[int]
    logits: Tensor  # [rows, streams, vocab]


@dataclass
class ToolTokenSample:
    token_id: Tensor
    logprob: Tensor


@dataclass
class FrameBatch:
    """Queued samples for the eligible requests, before any host read."""

    indices: list[int]  # Request index of each row.
    text: TextSamplingResult
    audio: Tensor
    hiddens: Tensor  # [rows, cells, hidden] backbone outputs.


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
    tool_call_constraint: ToolCallConstraintState | None = None
    frames_seen: int = 0
    audio_position: int = 0
    active_text_tokens: int = 0
    tool_call_sequence: int = 0
    sampling: RequestSampling | None = None

    def fork(self) -> DuplexIORequestState:
        """Commit an append only after its model step succeeds."""
        result = copy.copy(self)
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


@dataclass
class PreparedFrames:
    """Views into one packed batch, consumed by the runner's request hook."""

    embeddings: Tensor
    updates: dict[str, Any]


class FrameInputGraph:
    """Capture packed projections; CPU request state remains outside the graph."""

    def __init__(
        self, project: Callable[..., tuple[Tensor, ...]], inputs: tuple[Tensor, ...],
    ) -> None:
        self.inputs = tuple(value.clone() for value in inputs)
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                project(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = project(*self.inputs)

    def __call__(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        torch._foreach_copy_(self.inputs, inputs)
        self.graph.replay()
        # Outputs can remain in request state after another batch replays.
        return tuple(value.clone() for value in self.outputs)


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
        self.frame_input_graphs: dict[int, FrameInputGraph] = {}
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
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
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
    ) -> dict[str, dict[str, PreparedFrames]]:
        requests = [model_intermediate_buffer[req_id] for req_id in req_ids]
        prepared = self.prepare_frames(
            [(info["_omni_num_scheduled_tokens"], info) for info in requests], device,
        )
        return {req_id: {"prepared_frames": frames} for req_id, frames in zip(req_ids, prepared, strict=True)}

    @torch.inference_mode()
    def preprocess(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        prepared_frames: PreparedFrames | None = None,
        **info: Any,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        del input_embeds
        if prepared_frames is None:
            prepared_frames, = self.prepare_frames([(input_ids.numel(), info)], input_ids.device)
        return input_ids, prepared_frames.embeddings, prepared_frames.updates

    def prepare_frames(
        self, requests: list[tuple[int, dict[str, Any]]], device: torch.device,
    ) -> list[PreparedFrames]:
        """Pack scheduled frames once; request boundaries own all running counters."""
        audio = self.prepare_audio_requests(requests, device)
        ids = []
        rows = []
        preceding_text = 0
        for item, (_, info) in zip(audio, requests, strict=True):
            state = item.state
            start, count = item.frame_start, item.frame_count
            duplex = info["duplex"]
            prefill = duplex.get("duplexio_prefill", False)
            system = duplex.get("duplexio_system_input", False)
            live = not (prefill or system)
            text = state.text_input_ids.expand(count, -1).clone()
            if prefill:
                text.fill_(self.silence_token_id)
                text[item.prompt_chunk_frames:, 0] = torch.tensor(
                    state.system_token_ids[max(0, start - item.prompt_count):max(0, start + count - item.prompt_count)],
                    dtype=torch.long,
                )
            elif system:
                text.fill_(self.silence_token_id)
                text[:, 0] = torch.tensor(duplex["duplexio_system_token_ids"][start:start + count], dtype=torch.long)
                state.text_input_ids = torch.full_like(state.text_input_ids, self.silence_token_id)
            else:
                text[:, 0] = self.silence_token_id
            prompt_written = min(state.frames_seen, state.voice_prompt.numel() // self.config.frame_size)
            # CPU state provides packed-row offsets; no device readback is needed.
            rows.extend(
                (state.active_text_tokens - preceding_text,
                 state.audio_position + (row if live else 0), live,
                 prompt_written + min(row + 1, item.prompt_chunk_frames),
                 row < item.prompt_chunk_frames, state.frames_seen + row)
                for row in range(count)
            )
            emitted = ((text != self.pad_token_id) & (text != self.silence_token_id)).sum().item()
            state.active_text_tokens += emitted
            preceding_text += emitted
            state.frames_seen += count
            if live:
                state.audio_position += count
            ids.append(text)
        text_ids = torch.cat(ids).to(device, non_blocking=True)
        metadata = torch.tensor(rows, dtype=torch.int32).to(device, non_blocking=True)
        user_features = torch.cat([item.user_features for item in audio])
        agent_codes = torch.cat([item.agent_codes for item in audio])
        with torch.profiler.record_function("duplexio.frame_inputs"):
            inputs = (text_ids, metadata, user_features, agent_codes)
            # Capture bounded one-frame batches; variable-length prefixes stay packed.
            if self.full_cudagraph_enabled and all(item.frame_count == 1 for item in audio):
                size = len(audio)
                if size not in self.frame_input_graphs:
                    self.frame_input_graphs[size] = FrameInputGraph(self.project_frames, inputs)
                outputs = self.frame_input_graphs[size](inputs)
            else:
                outputs = self.project_frames(*inputs)
        embeddings, *addressing = outputs
        positions = (
            metadata[:, 5, None] * DUPLEXIO_NUM_CELLS + torch.arange(DUPLEXIO_NUM_CELLS, device=device)
        ).flatten()
        live_mask, prompt_mask = metadata[:, 2] != 0, metadata[:, 4] != 0
        results = []
        offset = 0
        for item, (_, info) in zip(audio, requests, strict=True):
            end = offset + item.frame_count
            cells = slice(offset * DUPLEXIO_NUM_CELLS, end * DUPLEXIO_NUM_CELLS)
            replay = {}
            if info["duplex"]["runtime_config"].get("duplexio_record_inputs", False):
                replay = {
                    "text_ids": text_ids[offset:end], "user_features": user_features[offset:end],
                    "agent_audio": agent_codes[offset:end], "audio_mask": live_mask[offset:end],
                    "prompt_frames": prompt_mask[offset:end],
                }
            results.append(PreparedFrames(embeddings[cells], {
                "duplexio_working_state": item.state,
                "duplexio_replay": replay,
                "duplexio": dict(zip(
                    ("positions", "key_active", "text_ordinals", "text_last",
                     "audio_first", "audio_last", "prompt_ordinal", "prompt_last"),
                    (value[cells] for value in (positions, *addressing)), strict=True,
                )),
            }))
            offset = end
        return results

    def project_frames(
        self, text_ids: Tensor, metadata: Tensor, user_features: Tensor, agent_codes: Tensor,
    ) -> tuple[Tensor, ...]:
        """Tensor-only packed projections and frame construction, shared by all requests."""
        with torch.autocast(
            device_type=text_ids.device.type,
            dtype=self.vllm_config.model_config.dtype,
            enabled=text_ids.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
        ):
            user_hidden = self.user_audio_input_adapter(user_features)
            agent_hidden = self.agent_audio_input_adapter(self.agent_audio_embedding(agent_codes))
        text_hidden = self.llm.base_model.model.embed_input_ids(text_ids.flatten()).view(
            text_ids.shape[0], len(TEXT_STREAM_NAMES), -1
        )
        return self.frame_inputs(
            text_ids, text_hidden, self.llm.channel_emb, user_hidden, agent_hidden,
            self.pad_token_id, self.silence_token_id,
            metadata, self.config.audio_attention_window_frames,
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

        batch = self.sample_frames(hidden_states, request_token_spans, infos, request_sample_eligible)
        states: list[DuplexIORequestState] = []
        for request_index, info in enumerate(infos):
            state = info.get("duplexio_working_state")
            if not isinstance(state, DuplexIORequestState):
                raise RuntimeError(f"DuplexIO request {request_index} is missing working state")
            states.append(state)
        rows = {} if batch is None else {index: row for row, index in enumerate(batch.indices)}
        decode_rows = [row for index, row in rows.items() if infos[index]["duplex"].get("decode_audio", True)]
        predicted_audio = [] if batch is None else batch.audio.unbind(0)
        waveforms = self.decode_agent_audio_batch(
            [predicted_audio[row] for row in decode_rows],
            [states[batch.indices[row]] for row in decode_rows],
        )
        record_hiddens = [
            info["duplex"]["runtime_config"].get("duplexio_record_hiddens", False) for info in infos
        ]
        sampled: dict[str, list[Tensor]] = {}
        if batch is not None:
            text = batch.text
            sampled = {
                "text_ids": [text.text_ids],
                "tool_starts": [text.tool_starts],
                "audio": [batch.audio.detach()],
                "waveforms": [waveform.detach() for waveform in waveforms],
                "agent_emit_logprob": [text.agent_emit_logprobs],
                "agent_token_logprob": [text.agent_token_logprobs],
                "tool_emit_logprob": [text.tool_emit_logprobs],
                "tool_token_logprob": [text.tool_token_logprobs],
                "user_emit_logprob": [text.user_emit_logprobs],
                "user_token_logprob": [text.user_token_logprobs],
            }
            if any(record_hiddens[index] for index in batch.indices):
                sampled["predictor_hiddens"] = [batch.hiddens.detach()]
        replay_names = ("text_ids", "user_features", "agent_audio", "audio_mask", "prompt_frames")
        empty = torch.empty(0, dtype=hidden_states.dtype)
        # Everything is queued; this is the step's one wait for the device.
        host = to_host({
            "sampled": sampled,
            "replay": {
                name: [info.get("duplexio_replay", {}).get(name, empty) for info in infos] for name in replay_names
            },
        })
        host_rows: dict[str, list[Tensor]] = {}
        host_ids: list[list[int]] = []
        if batch is not None:
            sampled = host["sampled"]
            host_ids = sampled["text_ids"][0].tolist()
            started = self.finish_text_batch(
                batch.text, [infos[index] for index in batch.indices], host_ids, sampled["tool_starts"][0].tolist(),
            )
            if started:
                sampled["tool_token_logprob"] = [batch.text.tool_token_logprobs.cpu()]
            host_rows = {
                name: list(values[0].unbind(0))
                for name, values in sampled.items() if name not in ("text_ids", "tool_starts", "waveforms")
            }
            decoded = dict(zip(decode_rows, sampled["waveforms"], strict=True))

        # Placeholders and request metadata start on the host.
        empty_audio = torch.empty(0, dtype=torch.float32)
        empty_codes = torch.empty(0, dtype=torch.long)
        no_tool_call = torch.empty(0, dtype=torch.uint8)
        flags = {False: torch.tensor([False]), True: torch.tensor([True])}
        silence_ids = torch.tensor([self.silence_token_id], dtype=torch.long)
        policy_version = torch.tensor([self.policy_version], dtype=torch.long)
        sample_rate = torch.tensor([self.config.sample_rate])
        logprob_names = (
            "agent_emit_logprob", "agent_token_logprob", "tool_emit_logprob", "tool_token_logprob",
            "user_emit_logprob", "user_token_logprob",
        )
        forced_ids: list[int] = []
        retained_tokens: list[int] = []
        position_budgets: list[int] = []
        chunk: dict[str, list[Tensor]] = {
            name: [] for name in (
                "agent_audio_token_ids", "user_token_id", "agent_token_id", "tool_call_token_id", "model_listen",
                "end_of_turn", "duplex_epoch", "duplex_turn_id", "duplex_prefill", "duplex_prefill_complete",
                "duplex_system_input", "duplex_system_input_complete", "tool_call_complete", "tool_call_json",
                "predictor_hiddens", *logprob_names[:4], "user_emit", *logprob_names[4:], "policy_version",
            )
        }
        audio_outputs: list[Tensor] = []
        for request_index, ((start, end), info, state) in enumerate(zip(request_token_spans, infos, states, strict=True)):
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
            row = rows.get(request_index)
            predicting = row is not None
            chunk["end_of_turn"].append(flags[bool(duplex.get("final", False))])
            chunk["duplex_epoch"].append(torch.tensor([duplex.get("epoch", 0)]))
            chunk["duplex_turn_id"].append(torch.tensor([duplex.get("turn_id", 0)]))
            chunk["duplex_prefill"].append(flags[bool(is_prefill and not predicting)])
            chunk["duplex_prefill_complete"].append(flags[bool(is_prefill and predicting)])
            chunk["duplex_system_input"].append(flags[bool(is_system_input and not predicting)])
            chunk["duplex_system_input_complete"].append(flags[bool(is_system_input and predicting)])
            if row is None:
                for name in (*logprob_names, "user_emit", "policy_version", "predictor_hiddens"):
                    chunk[name].append(empty)
                forced_ids.append(self.silence_token_id)
                for name in ("user_token_id", "agent_token_id", "tool_call_token_id"):
                    chunk[name].append(silence_ids)
                audio_outputs.append(empty_audio)
                chunk["agent_audio_token_ids"].append(empty_codes)
                chunk["model_listen"].append(flags[False])
                chunk["tool_call_complete"].append(flags[False])
                chunk["tool_call_json"].append(no_tool_call)
                continue
            # Return sampling probabilities alongside each prediction.
            for name in logprob_names:
                chunk[name].append(host_rows[name][row])
            chunk["predictor_hiddens"].append(
                host_rows["predictor_hiddens"][row] if record_hiddens[request_index] else empty
            )
            user_token_id, agent_token_id, tool_token_id = host_ids[row]
            chunk["user_emit"].append(flags[user_token_id != self.silence_token_id])
            chunk["policy_version"].append(policy_version)
            tool_call = batch.text.tool_calls[row]
            state.text_input_ids = torch.tensor(
                [self.silence_token_id, *host_ids[row]], dtype=torch.long, device="cpu",
            )
            state.agent_audio_codes = predicted_audio[row]
            forced_ids.append(agent_token_id)
            chunk["user_token_id"].append(state.text_input_ids[1:2])
            chunk["agent_token_id"].append(state.text_input_ids[2:3])
            chunk["tool_call_token_id"].append(state.text_input_ids[3:4])
            audio_outputs.append(decoded.get(row, empty_audio))
            chunk["agent_audio_token_ids"].append(host_rows["audio"][row])
            model_listen = agent_token_id == self.silence_token_id and tool_token_id == self.silence_token_id
            chunk["model_listen"].append(flags[model_listen])
            chunk["tool_call_complete"].append(flags[tool_call is not None])
            if tool_call is not None:
                state.tool_call_sequence += 1
            chunk["tool_call_json"].append(serialize_tool_call(tool_call, state.tool_call_sequence))

        self._forced_next_token_ids = forced_ids
        multimodal_outputs = {
            "audio": audio_outputs,
            "chunk": {
                "sample_rate_hz": [sample_rate] * len(infos),
                **chunk,
                **{f"replay_{name}": values for name, values in host["replay"].items()},
            },
        }
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=cast(Any, multimodal_outputs),
            streaming_retained_tokens=retained_tokens,
            streaming_position_budget=position_budgets,
        )

    def sample_frames(
        self,
        hidden_states: Tensor,
        request_token_spans: list[tuple[int, int]],
        infos: list[dict[str, Any]],
        request_sample_eligible: list[bool],
    ) -> FrameBatch | None:
        """Queue text and audio sampling for every eligible request; nothing reads back."""
        indices = [index for index, eligible in enumerate(request_sample_eligible) if eligible]
        if not indices:
            return None
        ends = [request_token_spans[index][1] for index in indices]
        if ends == [DUPLEXIO_NUM_CELLS * (row + 1) for row in range(len(ends))]:
            # Single-frame appends in batch order are already contiguous rows.
            rows = hidden_states[: ends[-1]].unflatten(0, (len(ends), DUPLEXIO_NUM_CELLS))
        else:
            rows = torch.stack([hidden_states[end - DUPLEXIO_NUM_CELLS : end] for end in ends])
        with torch.profiler.record_function("duplexio.text_projection"):
            text_logits, emit_logits = self.project_text(rows)
        with torch.profiler.record_function("duplexio.text_sampling"):
            sample_infos = [infos[index] for index in indices]
            text = self.queue_text_batch(text_logits, emit_logits, sample_infos)
        with torch.profiler.record_function("duplexio.audio_sampling"), torch.autocast(
            rows.device.type, dtype=self.vllm_config.model_config.dtype,
            enabled=rows.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
        ):
            if isinstance(self.audio_sampler, FlowMapSampler):
                noise = torch.randn(
                    len(indices), self.audio_representation.embedding_dim,
                    device=rows.device, dtype=torch.float32,
                )
                audio = self.audio_sampler.sample(rows[:, AGENT_AUDIO_CELL].float(), noise)
            else:
                audio = self.sample_depth_audio(rows, text.text_ids[:, 1], sample_infos)
        return FrameBatch(indices=indices, text=text, audio=audio, hiddens=rows)

    def sample_depth_audio(self, rows: Tensor, agent_ids: Tensor, infos: list[dict[str, Any]]) -> Tensor:
        groups: dict[tuple[float, int | None], list[int]] = {}
        for row, info in enumerate(infos):
            depth = info["duplexio_working_state"].sampling.depth
            assert depth is not None
            groups.setdefault((depth.temperature, depth.top_k), []).append(row)
        conditioning = rows[:, AGENT_AUDIO_CELL]
        if len(groups) == 1:
            ((temperature, top_k),) = groups
            return self.audio_sampler.sample(conditioning, agent_ids, temperature=temperature, top_k=top_k)
        codes: list[Tensor] = [conditioning] * len(infos)
        for (temperature, top_k), members in groups.items():
            index = _to_device(members, torch.long, rows.device)
            sampled = self.audio_sampler.sample(
                conditioning.index_select(0, index), agent_ids.index_select(0, index),
                temperature=temperature, top_k=top_k,
            )
            for row, value in zip(members, sampled.unbind(0), strict=True):
                codes[row] = value
        return torch.stack(codes)

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
        """Sample agent/tool/user decisions and apply the host's tool decisions."""
        text = self.queue_text_batch(logits, emit_logits, infos)
        host_ids = text.text_ids.tolist()
        self.finish_text_batch(text, infos, host_ids, text.tool_starts.tolist())
        return text

    def queue_text_batch(
        self,
        logits: Tensor,
        emit_logits: Tensor,
        infos: list[dict[str, Any]],
    ) -> TextSamplingResult:
        """Queue every stream's draws; tool starts wait for finish_text_batch."""
        for info in infos:
            state = info["duplexio_working_state"]
            runtime = info["duplex"]["runtime_config"]
            if state.sampling is None or state.sampling.source != sampling_source(runtime):
                state.sampling = self.resolve_sampling(runtime)
        agent_ids, agent_emit_logprobs, agent_token_logprobs = self.sample_agent_tokens(
            logits[:, 0], emit_logits[:, 0], infos
        )
        user_ids, user_emit_logprobs, user_token_logprobs = self.sample_stream_tokens(
            logits[:, 2], emit_logits[:, 2], infos, stream="user",
        )
        # Rows inside a call, or forced to start one, sample now; idle rows draw
        # a start decision the host reads with the other sampled ids.
        rows = len(infos)
        tool_ids = torch.full((rows,), self.silence_token_id, dtype=torch.long, device=logits.device)
        tool_token_logprobs = torch.zeros(rows, 1, dtype=torch.float32, device=logits.device)
        emitting = [False] * rows
        pending: list[int] = []
        for row, info in enumerate(infos):
            constraint = info["duplexio_working_state"].tool_call_constraint
            if constraint is None or not constraint.enabled:
                continue
            if constraint.active or constraint.force_next_call:
                emitting[row] = True
                self._sample_tool_row(logits[row, 1:2], info, tool_ids, tool_token_logprobs, row)
            else:
                pending.append(row)
        start_logits = emit_logits[:, 1].float()
        if pending:
            tool_starts, _ = _sample_emits(
                start_logits, [info["duplexio_working_state"].sampling.emission.tool_call for info in infos],
            )
        else:
            tool_starts = torch.zeros(rows, dtype=torch.bool, device=logits.device)
        # Score the chosen decision with the raw head, even when serving forced it.
        if len(pending) == rows:
            tool_emitted = tool_starts
        else:
            tool_emitted = _to_device(emitting, torch.bool, logits.device)
            if pending:
                is_pending = _to_device([row in pending for row in range(rows)], torch.bool, logits.device)
                tool_emitted = torch.where(is_pending, tool_starts, tool_emitted)
        tool_emit_logprobs = F.logsigmoid(torch.where(tool_emitted, start_logits, -start_logits))
        return TextSamplingResult(
            text_ids=torch.stack((user_ids, agent_ids, tool_ids), dim=1),
            tool_calls=[None] * rows,
            agent_emit_logprobs=agent_emit_logprobs.unsqueeze(-1),
            agent_token_logprobs=agent_token_logprobs.unsqueeze(-1),
            user_emit_logprobs=user_emit_logprobs.unsqueeze(-1),
            user_token_logprobs=user_token_logprobs.unsqueeze(-1),
            tool_emit_logprobs=tool_emit_logprobs.unsqueeze(-1),
            tool_token_logprobs=tool_token_logprobs,
            tool_starts=tool_starts,
            pending_tool_starts=pending,
            tool_rows=[row for row in range(rows) if emitting[row]],
            logits=logits,
        )

    def finish_text_batch(
        self,
        text: TextSamplingResult,
        infos: list[dict[str, Any]],
        host_ids: list[list[int]],
        tool_starts: list[bool],
    ) -> list[int]:
        """Start sampled calls and advance constraints once the draws reach the host.

        Patches ``text`` and ``host_ids`` in place; returns the rows that started a
        call, whose token log probabilities changed after the host read.
        """
        started = [row for row in text.pending_tool_starts if tool_starts[row]]
        for row in started:
            self._sample_tool_row(
                text.logits[row, 1:2], infos[row], text.text_ids[:, 2], text.tool_token_logprobs, row,
            )
        if started:
            for row, token_id in zip(started, text.text_ids[started, 2].tolist(), strict=True):
                host_ids[row][2] = token_id
        for row in sorted((*text.tool_rows, *started)):
            constraint = infos[row]["duplexio_working_state"].tool_call_constraint
            if constraint.accept(host_ids[row][2]):
                text.tool_calls[row] = self.tool_call_compiler.take_completed_call(constraint)
        return started

    def _sample_tool_row(
        self, logits: Tensor, info: dict[str, Any], tool_ids: Tensor, logprobs: Tensor, row: int,
    ) -> None:
        state = info["duplexio_working_state"]
        sample = sample_tool_token(
            logits, constraint=state.tool_call_constraint, emit=True,
            sampling=state.sampling.tool, distribution=self.content_distribution,
        )
        assert sample is not None
        tool_ids[row : row + 1].copy_(sample.token_id)
        logprobs[row].copy_(sample.logprob)

    def sample_agent_tokens(
        self, logits: Tensor, emit_logits: Tensor, infos: list[dict[str, Any]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.sample_stream_tokens(
            logits, emit_logits, infos, stream="agent",
        )

    def sample_stream_tokens(
        self, logits: Tensor, emit_logits: Tensor, infos: list[dict[str, Any]],
        *, stream: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample one stream for all rows with one random draw per decision.

        Returns the sampled ids and, per row, the log probability of the emit decision
        and of the content draw. Both come from the distributions already materialized
        for sampling, so nothing is recomputed.
        """
        policies = [info["duplexio_working_state"].sampling for info in infos]
        samplings = [policy.user if stream == "user" else policy.agent for policy in policies]
        emitted, emit_logprobs = _sample_emits(
            emit_logits.float(),
            [policy.emission.user if stream == "user" else policy.emission.agent for policy in policies],
        )
        groups: dict[tuple[float, int | None, float | None], list[int]] = {}
        for row, sampling in enumerate(samplings):
            groups.setdefault((sampling.temperature, sampling.top_k, sampling.top_p), []).append(row)
        if len(groups) == 1:
            content, content_logprobs = self._sample_content(logits, samplings[0])
        else:
            content = torch.empty(len(infos), dtype=torch.long, device=logits.device)
            content_logprobs = torch.empty(len(infos), dtype=torch.float32, device=logits.device)
            for rows in groups.values():
                index = _to_device(rows, torch.long, logits.device)
                ids, values = self._sample_content(logits.index_select(0, index), samplings[rows[0]])
                content.index_copy_(0, index, ids)
                content_logprobs.index_copy_(0, index, values)
        content = torch.where(emitted, content, self.silence_token_id)
        # A discarded content draw on a wait frame is not an action.
        content_logprobs = torch.where(emitted, content_logprobs, 0)
        return content, emit_logprobs, content_logprobs

    def _sample_content(self, logits: Tensor, sampling: TokenSamplingOptions) -> tuple[Tensor, Tensor]:
        if sampling.temperature == 0:
            content = _sample_content_token_ids(logits, sampling)
            return content, torch.zeros(content.shape, dtype=torch.float32, device=content.device)
        indices, probabilities = self.content_distribution(logits, sampling)
        selected = _exponential_race(probabilities).unsqueeze(-1)
        return (
            indices.gather(-1, selected).squeeze(-1),
            probabilities.gather(-1, selected).squeeze(-1).float().log(),
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
        return AutoWeightsLoader(self).load_weights(weights)

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
    metadata: Tensor,
    audio_window_frames: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Assemble six-cell inputs and cache addressing without changing CPU state.

    Besides the flattened embeddings this returns one entry per cell: whether the
    cell contributes a key, its 1-based text emission ordinal (0 when it emits
    nothing), how many text keys strictly earlier rows emitted, and the inclusive
    range of audio frames the cell may attend. Audio frames are numbered from
    one in audio time, which only advances on frames carrying real audio, so a
    text-only append leaves the range frozen and writes no audio key.

    A pinned voice-prompt burst is the one frame kind that carries audio without
    being live: only its agent-audio cell contributes a key; audio time stays frozen and
    the keys land in the cache's pinned region, which no window expires.

    Metadata is int32, one row per packed frame: text-cumsum offset, audio
    position, live flag, total prompt frames seen, prompt flag, absolute frame.
    The text offset subtracts preceding requests' emissions, isolating the scan.
    """
    frames, device = text_ids.shape[0], text_ids.device
    text_offsets, audio_last = metadata[:, 0], metadata[:, 1]
    audio_active, prompt_frames = metadata[:, 2] != 0, metadata[:, 4] != 0
    prompt_total = metadata[:, 3]
    acoustic = (audio_active | prompt_frames)[:, None]
    user_hidden = user_hidden.masked_fill(~acoustic, 0)
    agent_hidden = agent_hidden.masked_fill(~acoustic, 0)
    text_hidden = text_hidden.masked_fill((text_ids == silence_token_id).unsqueeze(-1), 0)
    embeddings = torch.cat(
        (text_hidden + channel_embedding, user_hidden.unsqueeze(1), agent_hidden.unsqueeze(1)), dim=1,
    ).flatten(0, 1)
    text_active = (text_ids != pad_token_id) & (text_ids != silence_token_id)
    audio_shape = (frames, 2)
    audio_keyed = torch.stack(
        (audio_active, prompt_frames | audio_active), dim=-1,
    )
    key_active = torch.cat(
        (text_active, audio_keyed), dim=1,
    ).flatten()
    ordinals = (
        text_active.flatten().cumsum(0, dtype=torch.int32).view_as(text_active) + text_offsets[:, None]
    )
    text_ordinals = torch.cat(
        (torch.where(text_active, ordinals, 0), torch.zeros(audio_shape, dtype=torch.int32, device=device)),
        dim=1,
    ).flatten()
    # A row sees the text its predecessors emitted, never its own siblings'.
    row_emitted = text_active.sum(1, dtype=torch.int32)
    text_last = row_emitted.cumsum(0) - row_emitted + text_offsets
    audio_first = (audio_last + audio_active - audio_window_frames).clamp_min(1)
    # A prompt row's own pinned key is its self key, which the mask merges
    # separately, so a row sees only the prompt frames written before it.
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
    generator: torch.Generator | None = None,
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
    generator: torch.Generator | None = None,
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
    generator: torch.Generator | None = None,
) -> Tensor:
    if temperature == 0:
        return emit_logits >= 0
    return torch.bernoulli(
        torch.sigmoid(emit_logits.float() / temperature),
        generator=generator,
    ).bool()


def _sample_emits(emit_logits: Tensor, temperatures: list[float]) -> tuple[Tensor, Tensor]:
    """Draw every row's emit decision at its temperature; zero thresholds the logit.

    Returns the decisions and the log probability of each; thresholded decisions
    have probability one.
    """
    if len(set(temperatures)) == 1:
        if temperatures[0] == 0:
            return emit_logits >= 0, torch.zeros_like(emit_logits)
        probability = torch.sigmoid(emit_logits / temperatures[0])
        emitted = torch.rand_like(probability) < probability
        return emitted, torch.where(emitted, probability, 1 - probability).log()
    temperature = _to_device(temperatures, emit_logits.dtype, emit_logits.device)
    thresholded = temperature == 0
    probability = torch.sigmoid(emit_logits / temperature.masked_fill(thresholded, 1))
    emitted = torch.where(thresholded, emit_logits >= 0, torch.rand_like(probability) < probability)
    return emitted, torch.where(emitted, probability, 1 - probability).log().masked_fill(thresholded, 0)


def _exponential_race(probabilities: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Sample each row's index, as vLLM does, without multinomial's validation sync."""
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    return probabilities.div(noise).argmax(dim=-1)


def _to_device(values: list[Any], dtype: torch.dtype, device: torch.device) -> Tensor:
    """Upload host metadata without waiting for the queued backbone."""
    host = torch.tensor(values, dtype=dtype, pin_memory=device.type == "cuda")
    return host.to(device, non_blocking=True)


def sample_tool_token(
    logits: Tensor,
    *,
    constraint: ToolCallConstraintState | None,
    emit: bool,
    sampling: TokenSamplingOptions,
    generator: torch.Generator | None = None,
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
    selected = _exponential_race(probabilities, generator).unsqueeze(-1)
    return ToolTokenSample(
        token_id=indices.gather(-1, selected).squeeze(-1),
        logprob=probabilities.gather(-1, selected).squeeze(-1).float().log(),
    )


def to_host(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``outputs``, nested dicts of per-request tensor lists, to the host.

    Device values that share a dtype cross in one transfer and come back as views
    of it in their original shapes. Host tensors pass through untouched.
    """
    groups: dict[tuple[torch.device, torch.dtype], list[tuple[list[Tensor], int, Tensor]]] = {}

    def lists(values: Mapping[str, Any]) -> dict[str, Any]:
        host: dict[str, Any] = {}
        for name, value in values.items():
            if isinstance(value, Mapping):
                host[name] = lists(value)
                continue
            host[name] = value = list(value)
            for row, tensor in enumerate(value):
                if tensor.device.type != "cpu":
                    groups.setdefault((tensor.device, tensor.dtype), []).append((value, row, tensor))
        return host

    host = lists(outputs)
    for items in groups.values():
        packed = torch.cat([tensor.detach().reshape(-1) for _, _, tensor in items]).cpu()
        parts = packed.split([tensor.numel() for _, _, tensor in items])
        for (values, row, tensor), part in zip(items, parts, strict=True):
            values[row] = part.view(tensor.shape)
    return host


def serialize_tool_call(tool_call: Mapping[str, Any] | None, sequence: int) -> Tensor:
    if tool_call is None:
        return torch.empty(0, dtype=torch.uint8)
    payload = json.dumps(
        {
            "sequence": sequence,
            "name": tool_call["name"],
            "arguments": tool_call["arguments"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8)


def _validate_vllm_runtime_contract(vllm_config: VllmConfig) -> None:
    if vllm_config.quant_config is not None:
        raise ValueError("DuplexIO requires unquantized backbone weights to preserve training's fused MLP math")
    if vllm_config.model_config.head_dtype not in (None, torch.float32):
        raise ValueError("DuplexIO vocabulary logits require FP32 accumulation")
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


__all__ = ["DuplexIOForConditionalGeneration", "DuplexIORequestState"]
