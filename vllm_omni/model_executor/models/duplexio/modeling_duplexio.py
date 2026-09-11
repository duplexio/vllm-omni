# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM implementation of the DuplexIO full-duplex model."""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import xgrammar as xgr
from torch import Tensor, nn
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

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AgentAudioInputAdapter,
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_input_graph import AudioInputGraph
from vllm_omni.model_executor.models.duplexio.audio_representation import (
    ContinuousAudioRepresentation,
    DelayedMimiRepresentation,
    DelayedMimiState,
    MimiEmbedding,
)
from vllm_omni.model_executor.models.duplexio.checkpoint import (
    load_voice_pools,
    resolve_checkpoint_directory,
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
    FastConformerAudioStreamState,
    FastConformerRNNT,
    streaming_resample_chunk,
)
from vllm_omni.model_executor.models.duplexio.flowmap import FlowMapSampler
from vllm_omni.model_executor.models.duplexio.mimi import (
    MimiModel,
    MimiStreamingState,
)
from vllm_omni.model_executor.models.duplexio.numerics import FixedLinear, fixed_linear
from vllm_omni.model_executor.models.duplexio.pocket_mimi import (
    ContinuousMimiState,
    PocketMimi,
)
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOQwenModel,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    duplexio_frame_positions,
)
from vllm_omni.model_executor.models.duplexio.stream_gdn import gdn_cache_dtypes, gdn_cache_shapes
from vllm_omni.model_executor.models.duplexio.text_sampling import TokenSamplingOptions, content_distribution
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
class FramePrediction:
    """One sampled output, before request feedback and optional codec decoding."""

    text_ids: Tensor
    audio: Tensor
    tool_call: dict[str, Any] | None


@dataclass
class DuplexIORequestState:
    """Request-owned state; Qwen KV/GDN caches belong to the native runner."""

    text_input_ids: Tensor  # CPU feedback, also used for scheduler text counts.
    agent_audio_codes: Tensor
    user_asr: FastConformerAudioStreamState
    output_mimi: ContinuousMimiState | MimiStreamingState
    agent_delay: DelayedMimiState | None
    speaker_embedding: Tensor
    depth_speaker_conditioning: DepthSpeakerConditioning | None
    system_token_ids: tuple[int, ...]
    sampling_generator: torch.Generator
    tool_call_constraint: ToolCallConstraintState | None = None
    system_token_offset: int = 0
    frames_seen: int = 0
    audio_position: int = 0
    active_text_tokens: int = 0
    tool_call_sequence: int = 0
    # Transcript tokens the RNN-T has produced but the user stream has not
    # emitted yet, and whether any word has been emitted (all but the first
    # word of an utterance carries a leading space, as in training).
    user_text_pending: tuple[int, ...] = ()
    user_text_started: bool = False

    def fork(self) -> DuplexIORequestState:
        """Commit an append only after its model step succeeds."""
        result = copy.copy(self)
        result.sampling_generator = _fork_generator(self.sampling_generator)
        if self.agent_delay is not None:
            result.agent_delay = copy.copy(self.agent_delay)
        if isinstance(self.output_mimi, MimiStreamingState):
            result.output_mimi = self.output_mimi.fork()
        if self.tool_call_constraint is not None:
            result.tool_call_constraint = self.tool_call_constraint.fork()
        # Pocket codec state is functional. The mutable HF ASR cache is copied
        # only when processing raw audio, not for cached offline input frames.
        return result


@dataclass(frozen=True, slots=True)
class EmitSamplingTemperatures:
    user: float
    agent: float
    tool_call: float


class DuplexIOLogitsProcessor(LogitsProcessor):
    """Use the learner's fixed BF16 projection before vLLM's vocabulary gather."""

    def _apply_head(
        self,
        lm_head: ParallelLMHead,
        hidden_states: Tensor,
        embedding_bias: Tensor | None,
    ) -> Tensor:
        return fixed_linear(hidden_states, lm_head.weight, embedding_bias)


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
            {name: FixedLinear(text_config.hidden_size, text_config.hidden_size) for name in TARGET_STREAM_NAMES}
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
    # The training export includes a system-stream projection. Serving feeds
    # the system stream as context and predicts only user, agent, and tool_call.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"llm.output_head_proj.system.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        if not isinstance(config, DuplexIOConfig):
            raise TypeError("DuplexIOForConditionalGeneration requires DuplexIOConfig")
        _validate_vllm_runtime_contract(vllm_config)
        self.vllm_config = vllm_config
        self.config = config
        self.text_config = vllm_config.model_config.hf_text_config
        self.full_cudagraph_enabled = (
            vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and not vllm_config.model_config.enforce_eager
        )
        self.audio_input_graphs: dict[int, AudioInputGraph] = {}
        self.content_distribution = (
            torch.compile(
                content_distribution, fullgraph=True, dynamic=True,
                options={"emulate_precision_casts": True, "triton.cudagraphs": True},
            )
            if self.full_cudagraph_enabled else content_distribution
        )
        self.frame_inputs = (
            torch.compile(frame_inputs, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
            if self.full_cudagraph_enabled else frame_inputs
        )
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("duplexio", "key_active"),
            ("duplexio", "text_ordinals"),
            ("duplexio", "text_last"),
            ("duplexio", "audio_first"),
            ("duplexio", "audio_last"),
            ("duplexio", "user_token_id"),
            ("duplexio_replay", "text_ids"),
            ("duplexio_replay", "user_features"),
            ("duplexio_replay", "agent_audio"),
            ("duplexio_replay", "audio_mask"),
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
        speaker_dim = config.speaker_lda_dim or config.speaker_embed_dim
        root = resolve_checkpoint_directory(
            vllm_config.model_config.model,
            revision=vllm_config.model_config.revision,
        )
        self.user_asr = FastConformerRNNT.from_export(config.user_asr_config, root)
        if config.speaker_lda_dim is None:
            self.register_buffer("speaker_lda_projection", None)
            self.register_buffer("speaker_lda_mean", None)
        else:
            self.register_buffer(
                "speaker_lda_projection",
                torch.empty(
                    speaker_dim,
                    config.speaker_embed_dim,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "speaker_lda_mean",
                torch.empty(
                    config.speaker_embed_dim,
                    dtype=torch.float32,
                ),
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
                    speaker_embedding_dim=speaker_dim,
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
        self.agent_audio_input_adapter = AgentAudioInputAdapter(
            representation_dim,
            speaker_dim,
            adapter_hidden_size,
            hidden_size,
        )
        self.agent_emit_head = nn.Linear(DUPLEXIO_NUM_CELLS * hidden_size, 1)
        self.tool_call_emit_head = nn.Linear(DUPLEXIO_NUM_CELLS * hidden_size, 1)
        self.logits_processor = DuplexIOLogitsProcessor(self.text_config.vocab_size)
        self.make_empty_intermediate_tensors = self.llm.base_model.model.make_empty_intermediate_tensors
        self._voice_pools = load_voice_pools(
            vllm_config.model_config.model,
            speaker_embed_dim=config.speaker_embed_dim,
            default_voice=config.default_voice,
            revision=vllm_config.model_config.revision,
        )
        self._forced_next_token_ids: list[int] | None = None
        # Also encodes the streaming RNN-T's user words for the user text stream.
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
        )

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
    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> None:
        """Batch live user features and committed agent audio at the input boundary."""
        requests = [
            model_intermediate_buffer[request_id]
            for request_id in req_ids
            if not model_intermediate_buffer[request_id]["duplex"].get("duplexio_prefill", False)
            and not model_intermediate_buffer[request_id]["duplex"].get("duplexio_system_input", False)
            and model_intermediate_buffer[request_id]["duplex"]["payload"]["format"] == "duplexio_features"
        ]
        if not requests:
            return
        features = torch.cat([request["embed"]["speech_feat"] for request in requests]).to(device)
        # A session's first frame has no committed audio state yet.
        agent_requests = [request for request in requests if "duplexio_model_state" in request]
        if agent_requests:
            states = [request["duplexio_model_state"] for request in agent_requests]
            codes = torch.stack([state.agent_audio_codes for state in states])
            speakers = torch.stack([state.speaker_embedding for state in states])
        if self.full_cudagraph_enabled and len(agent_requests) == len(requests):
            batch_size = len(requests)
            if batch_size not in self.audio_input_graphs:
                self.audio_input_graphs[batch_size] = AudioInputGraph(
                    self.user_audio_input_adapter, self.agent_audio_embedding, self.agent_audio_input_adapter,
                    features, codes, speakers, self.vllm_config.model_config.dtype,
                )
            hidden, agent_hidden = self.audio_input_graphs[batch_size](features, codes, speakers)
        else:
            with torch.autocast(
                device.type, dtype=self.vllm_config.model_config.dtype,
                enabled=device.type == "cuda" and self.vllm_config.model_config.dtype != torch.float32,
            ):
                hidden = self.user_audio_input_adapter(features)
                if agent_requests:
                    agent_hidden = self.agent_audio_input_adapter(self.agent_audio_embedding(codes), speakers)
        for index, request in enumerate(requests):
            request["embed"]["speech_feat"] = features[index : index + 1]
            request["embed"]["user_hidden"] = hidden[index : index + 1]
        if agent_requests:
            for index, request in enumerate(agent_requests):
                request["embed"]["agent_hidden"] = agent_hidden[index : index + 1]

    def user_text_token_ids(self, words: tuple[str, ...], started: bool) -> tuple[int, ...]:
        """Encode finished user words the way training's transcript stream does.

        Training assigns Qwen ids to the whole transcript by character overlap,
        so every word but the utterance's first carries its leading space, which
        the words already do — the first word of all just drops it.
        """
        text = "".join(words)
        return tuple(
            self.tokenizer.encode(
                text if started else text.lstrip(),
                add_special_tokens=False,
            )
        )

    @torch.inference_mode()
    def preprocess(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        **info: Any,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        del input_embeds
        # Token feedback stays on the CPU for scheduler bookkeeping; embeddings,
        # audio state and model metadata belong to the engine device.
        input_ids = input_ids.to(self.llm.channel_emb.device)
        duplex = info.get("duplex")
        if not isinstance(duplex, Mapping):
            raise ValueError("Native DuplexIO accepts only framed duplex appends")
        frame_count = duplex.get("frame_count")
        if not isinstance(frame_count, int) or frame_count < 1 or input_ids.numel() != frame_count * DUPLEXIO_NUM_CELLS:
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
            raise ValueError("DuplexIO text-input flags must be boolean when present")
        if prefill_final and not is_prefill:
            raise ValueError("DuplexIO prefill_final requires duplexio_prefill")
        if system_input_final and not is_system_input:
            raise ValueError("DuplexIO system_input_final requires duplexio_system_input")
        if is_prefill and is_system_input:
            raise ValueError("DuplexIO prefill and system input are mutually exclusive")
        if frame_count != 1 and not (is_prefill or is_system_input):
            raise ValueError("Native DuplexIO batches only silent text-input frames")

        text_ids = state.text_input_ids.expand(frame_count, -1).clone()
        if is_system_input:
            system_token_ids = duplex.get("duplexio_system_token_ids")
            if (
                not isinstance(system_token_ids, list)
                or len(system_token_ids) != frame_count
                or not all(isinstance(token_id, int) and token_id >= 0 for token_id in system_token_ids)
            ):
                raise ValueError("DuplexIO system input requires one token ID per frame")
            text_ids.fill_(self.silence_token_id)
            text_ids[:, 0] = torch.tensor(
                system_token_ids,
                dtype=torch.long,
                device="cpu",
            )
            state.text_input_ids = torch.full_like(
                state.text_input_ids,
                self.silence_token_id,
            )
        else:
            system_token_start = state.system_token_offset
            system_token_end = system_token_start + frame_count
            if is_prefill and system_token_end > len(state.system_token_ids):
                raise ValueError("DuplexIO prefill exceeds the remaining system tokens")
            system_tokens = state.system_token_ids[system_token_start:system_token_end]
            if system_tokens:
                text_ids[: len(system_tokens), 0] = torch.tensor(
                    system_tokens,
                    dtype=torch.long,
                    device="cpu",
                )
                state.system_token_offset += len(system_tokens)
            if len(system_tokens) < frame_count:
                text_ids[len(system_tokens) :, 0] = self.silence_token_id

        prepared_features = not (is_prefill or is_system_input) and duplex["payload"]["format"] == "duplexio_features"
        if is_prefill or is_system_input:
            user_features = self.llm.channel_emb.new_zeros(frame_count, self.user_asr.output_dim)
            agent_codes = self.initial_agent_audio(frame_count)
            if is_prefill:
                remaining = state.system_token_offset < len(state.system_token_ids)
                if prefill_final == remaining:
                    raise ValueError("DuplexIO final-prefill flag disagrees with remaining system tokens")
        else:
            user_features = (
                info["embed"]["speech_feat"]
                if prepared_features
                else None
            )
            user_words: tuple[str, ...] = ()
            if user_features is None:
                waveform = _decode_frame(duplex["payload"], input_ids.device, torch.float32)
                if waveform.shape[-1] != self.config.frame_size:
                    raise ValueError("A live DuplexIO append must contain one 80 ms audio frame")
                asr_state = copy.deepcopy(state.user_asr)
                waveform, tail = streaming_resample_chunk(
                    waveform[0, 0],
                    asr_state.resample_tail,
                    self.config.sample_rate,
                    16_000,
                )
                encoded, state.user_asr = self.user_asr.encode_audio_chunk(waveform, asr_state)
                state.user_asr.resample_tail = tail
                user_words, state.user_asr.rnnt = self.user_asr.decode_words(
                    encoded,
                    state.user_asr.rnnt,
                )
                user_features = encoded[0]
            # An input owner that has its own transcript supplies user tokens on
            # their actual arrival frame. A live websocket session has none, so
            # the streaming RNN-T fills the stream the way training does: word
            # tokens, one per frame, a few frames behind the speech itself.
            user_token_id = duplex["user_token_id"]
            if user_token_id is None:
                if user_words:
                    state.user_text_pending += self.user_text_token_ids(
                        user_words,
                        state.user_text_started,
                    )
                    state.user_text_started = True
                if state.user_text_pending:
                    user_token_id = state.user_text_pending[0]
                    state.user_text_pending = state.user_text_pending[1:]
            text_ids[:, USER_CELL] = (
                self.silence_token_id if user_token_id is None else user_token_id
            )
            agent_codes = state.agent_audio_codes.unsqueeze(0)
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=self.vllm_config.model_config.dtype,
            enabled=input_ids.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
        ):
            user_hidden = (
                info["embed"]["user_hidden"] if prepared_features else self.user_audio_input_adapter(user_features)
            )
            agent_hidden = info["embed"].get("agent_hidden") if prepared_features else None
            if agent_hidden is None:
                agent_hidden = self.agent_audio_input_adapter(
                    self.agent_audio_embedding(agent_codes),
                    state.speaker_embedding.unsqueeze(0),
                )
        active_text_count = ((text_ids != self.pad_token_id) & (text_ids != self.silence_token_id)).sum().item()
        text_ids = text_ids.to(input_ids.device, non_blocking=True)
        text_hidden = self.llm.base_model.model.embed_input_ids(text_ids.flatten()).view(
            frame_count, len(TEXT_STREAM_NAMES), -1
        )
        embeddings, key_active, text_ordinals, text_last, audio_first, audio_last = self.frame_inputs(
            text_ids, text_hidden, self.llm.channel_emb, user_hidden, agent_hidden,
            self.pad_token_id, self.silence_token_id, state.active_text_tokens,
            state.audio_position, self.config.audio_attention_window_frames,
            not is_prefill and not is_system_input,
        )
        state.active_text_tokens += active_text_count
        state.frames_seen += frame_count
        # Audio time advances only on frames that carry real audio: text-only
        # prefill and system-token bursts leave the counter frozen so they do
        # not consume the audio attention window.
        if not is_prefill and not is_system_input:
            state.audio_position += frame_count
        replay = {}
        if runtime_config.get("duplexio_record_inputs", False):
            replay = {
                "text_ids": text_ids,
                "user_features": user_features,
                "agent_audio": agent_codes,
                "audio_mask": key_active.view(frame_count, DUPLEXIO_NUM_CELLS)[:, 4],
            }
        return (
            input_ids,
            embeddings,
            {
                "duplexio_working_state": state,
                "duplexio_replay": replay,
                "duplexio": {
                    "key_active": key_active,
                    "user_token_id": text_ids[-1, USER_CELL],
                    "text_ordinals": text_ordinals,
                    "text_last": text_last,
                    "audio_first": audio_first,
                    "audio_last": audio_last,
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
                positions=duplexio_frame_positions(positions),
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
        if not all(request_sample_eligible):
            raise RuntimeError(
                "DuplexIO frames must be scheduled atomically; disable chunked prefill and speculative decoding"
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
        predictor_hiddens: list[Tensor] = []
        replay_outputs: dict[str, list[Tensor]] = {
            f"replay_{name}": [] for name in ("text_ids", "user_features", "agent_audio", "audio_mask")
        }
        predictions = self.sample_frames(hidden_states, request_token_spans, infos)
        sampled_ids = dict(zip(
            predictions,
            torch.stack([prediction.text_ids for prediction in predictions.values()]).tolist() if predictions else [],
            strict=True,
        ))
        for request_index, ((start, end), info) in enumerate(zip(request_token_spans, infos, strict=True)):
            state = info.get("duplexio_working_state")
            if not isinstance(state, DuplexIORequestState):
                raise RuntimeError(f"DuplexIO request {request_index} is missing working state")
            span_length = end - start
            if span_length < DUPLEXIO_NUM_CELLS or span_length % DUPLEXIO_NUM_CELLS:
                raise ValueError(f"DuplexIO request span must contain complete frames, got ({start}, {end})")
            duplex_info = info.get("duplexio")
            if not isinstance(duplex_info, Mapping):
                raise RuntimeError(f"DuplexIO request {request_index} is missing frame metadata")
            duplex = info.get("duplex", {})
            if not isinstance(duplex, Mapping):
                raise RuntimeError(f"DuplexIO request {request_index} is missing duplex metadata")
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
                raise RuntimeError(f"DuplexIO request {request_index} has invalid text-input flags")
            if prefill_final and not is_prefill:
                raise RuntimeError(f"DuplexIO request {request_index} has invalid final prefill flag")
            if system_input_final and not is_system_input:
                raise RuntimeError(f"DuplexIO request {request_index} has invalid final system-input flag")
            frame_count = duplex.get("frame_count")
            if frame_count != span_length // DUPLEXIO_NUM_CELLS:
                raise RuntimeError(f"DuplexIO request {request_index} frame span does not match metadata")
            row_hidden = hidden_states[end - DUPLEXIO_NUM_CELLS : end]
            prediction = predictions.get(request_index)
            record_hiddens = duplex["runtime_config"].get("duplexio_record_hiddens", False)
            predictor_hiddens.append(
                row_hidden.detach() if record_hiddens and prediction is not None else row_hidden.new_empty(0)
            )
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
                end_flags.append(torch.tensor([bool(duplex.get("final", False))]))
                epochs.append(torch.tensor([int(duplex.get("epoch", 0))]))
                turn_ids.append(torch.tensor([int(duplex.get("turn_id", 0))]))
                prefill_flags.append(torch.tensor([is_prefill]))
                prefill_complete_flags.append(torch.tensor([False]))
                system_input_flags.append(torch.tensor([is_system_input]))
                system_input_complete_flags.append(torch.tensor([False]))
                tool_call_complete_flags.append(torch.tensor([False]))
                tool_call_payloads.append(row_hidden.new_empty(0, dtype=torch.uint8))
                continue
            predicted_audio, tool_call = prediction.audio, prediction.tool_call
            _, agent_token_id, tool_token_id = sampled_ids[request_index]
            state.text_input_ids = torch.tensor(
                [self.silence_token_id, *sampled_ids[request_index]], dtype=torch.long, device="cpu",
            )
            state.agent_audio_codes = predicted_audio
            waveform = row_hidden.new_empty(0, dtype=torch.float32)
            if duplex.get("decode_audio", True):
                with (
                    torch.profiler.record_function("duplexio.output_codec_decode"),
                    torch.autocast(
                        device_type=row_hidden.device.type,
                        dtype=self.vllm_config.model_config.dtype,
                        enabled=row_hidden.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
                    ),
                ):
                    if isinstance(self.audio_codec, PocketMimi):
                        latent = self.audio_representation.denormalize(predicted_audio)
                        decoded, state.output_mimi = self.audio_codec.decode(
                            latent[None, :, None],
                            state.output_mimi,
                        )
                        waveform = decoded[0, 0]
                    else:
                        raw_codes = self.audio_representation.decode_column(predicted_audio, state.agent_delay)
                        if raw_codes is not None:
                            waveform = self.audio_codec.decode(
                                raw_codes[None, :, None],
                                state.output_mimi,
                            )[0, 0]
            forced_ids.append(agent_token_id)

            user_ids.append(state.text_input_ids[1:2])
            agent_ids.append(state.text_input_ids[2:3])
            tool_ids.append(state.text_input_ids[3:4])
            audio_outputs.append(waveform.detach())
            audio_token_ids.append(predicted_audio.detach())
            model_listen = agent_token_id == self.silence_token_id and tool_token_id == self.silence_token_id
            listen_flags.append(torch.tensor([model_listen]))
            end_flags.append(torch.tensor([bool(duplex.get("final", False))]))
            epochs.append(torch.tensor([int(duplex.get("epoch", 0))]))
            turn_ids.append(torch.tensor([int(duplex.get("turn_id", 0))]))
            prefill_flags.append(torch.tensor([is_prefill and not prefill_final]))
            prefill_complete_flags.append(torch.tensor([is_prefill and prefill_final]))
            system_input_flags.append(torch.tensor([is_system_input and not system_input_final]))
            system_input_complete_flags.append(torch.tensor([is_system_input and system_input_final]))
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
                **replay_outputs,
            },
        )
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def sample_frames(
        self,
        hidden_states: Tensor,
        request_token_spans: list[tuple[int, int]],
        infos: list[dict[str, Any]],
    ) -> dict[int, FramePrediction]:
        """Batch deterministic heads while preserving each request's RNG order."""
        indices: list[int] = []
        for index, info in enumerate(infos):
            duplex = info["duplex"]
            if (duplex.get("duplexio_prefill", False) and not duplex.get("duplexio_prefill_final", False)) or (
                duplex.get("duplexio_system_input", False) and not duplex.get("duplexio_system_input_final", False)
            ):
                continue
            indices.append(index)
        if not indices:
            return {}
        ends = [request_token_spans[index][1] for index in indices]
        rows = torch.stack([hidden_states[end - DUPLEXIO_NUM_CELLS : end] for end in ends])
        with torch.profiler.record_function("duplexio.text_projection"):
            text_logits, emit_logits = self.project_text(rows)
        with torch.profiler.record_function("duplexio.text_sampling"):
            texts, calls = self.sample_text_batch(text_logits, emit_logits, [infos[index] for index in indices])
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
                temperature, top_k = _depth_sampling(info)
                with torch.autocast(
                    rows.device.type, dtype=self.vllm_config.model_config.dtype,
                    enabled=rows.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
                ):
                    depth_audio.append(self.audio_sampler.sample(
                        rows[row, AGENT_AUDIO_CELL : AGENT_AUDIO_CELL + 1], texts[row][1:2],
                        state.depth_speaker_conditioning, temperature=temperature, top_k=top_k,
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
            index: FramePrediction(text, latent, call)
            for index, text, latent, call in zip(indices, texts, audio, calls, strict=True)
        }

    def project_text(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Project the entire request batch before any CPU-side tool decisions."""
        projected = torch.cat(
            (
                self.llm.output_head_proj["agent"](rows[:, AGENT_CELL]),
                self.llm.output_head_proj["tool_call"](rows[:, TOOL_CALL_CELL]),
            )
        )
        logits = self.logits_processor(self.llm.base_model.lm_head, projected)
        logits = logits.view(2, rows.shape[0], -1).transpose(0, 1)
        full_frames = rows.flatten(1)
        emit_logits = torch.cat(
            (self.agent_emit_head(full_frames), self.tool_call_emit_head(full_frames)), dim=-1,
        )
        return logits, emit_logits

    def sample_text_batch(
        self,
        logits: Tensor,
        emit_logits: Tensor,
        infos: list[dict[str, Any]],
    ) -> tuple[list[Tensor], list[dict[str, Any] | None]]:
        """Batch tool decisions and token readback without changing per-request RNG."""
        agent_ids = self.sample_agent_tokens(logits[:, 0], emit_logits[:, 0], infos)
        samplings: list[TokenSamplingOptions] = []
        tool_starts = [False] * len(infos)
        pending_indices: list[int] = []
        pending_starts: list[Tensor] = []
        for row, info in enumerate(infos):
            state = info["duplexio_working_state"]
            sampling = _text_sampling(info, self.tool_suppressed_token_ids)
            samplings.append(sampling)
            temperatures = _emit_temperatures(info)
            constraint = state.tool_call_constraint
            if constraint is not None and constraint.enabled and not constraint.active:
                if constraint.force_next_call:
                    tool_starts[row] = True
                else:
                    pending_indices.append(row)
                    pending_starts.append(_sample_emit(
                        emit_logits[row, 1:2],
                        0.0 if sampling.mode in {"argmax", "max"} else temperatures.tool_call,
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
            tool_id = sample_tool_token_id(
                logits[row, 1:2], constraint=state.tool_call_constraint,
                emit=tool_starts[row],
                sampling=samplings[row], generator=state.sampling_generator,
                distribution=self.content_distribution,
            )
            if tool_id is None:
                tool_id = logits.new_full((1,), self.silence_token_id, dtype=torch.long)
            else:
                token_indices.append(row)
                tool_tokens.append(tool_id)
            texts.append(torch.cat((info["duplexio"]["user_token_id"].view(1), agent_ids[row], tool_id)))
        if tool_tokens:
            for row, token_id in zip(token_indices, torch.cat(tool_tokens).tolist(), strict=True):
                constraint = infos[row]["duplexio_working_state"].tool_call_constraint
                if constraint.accept(token_id):
                    calls[row] = self.tool_call_compiler.take_completed_call(constraint)
        return texts, calls

    def sample_agent_tokens(self, logits: Tensor, emit_logits: Tensor, infos: list[dict[str, Any]]) -> list[Tensor]:
        """Filter equal-policy requests together; keep their random draws independent."""
        samplings = [_text_sampling(info, self.agent_suppressed_token_ids) for info in infos]
        groups: dict[tuple[str, float, int, float], list[int]] = {}
        for row, sampling in enumerate(samplings):
            key = sampling.mode, sampling.temperature, sampling.top_k, sampling.top_p
            groups.setdefault(key, []).append(row)
        tokens: dict[int, Tensor] = {}
        for rows in groups.values():
            sampling = samplings[rows[0]]
            group_logits = torch.stack([logits[row] for row in rows])
            greedy = sampling.mode in {"argmax", "max"}
            if greedy:
                content = _sample_content_token_ids(
                    group_logits, sampling, generator=infos[rows[0]]["duplexio_working_state"].sampling_generator,
                )
            else:
                indices, probabilities = self.content_distribution(group_logits, sampling)
            for index, row in enumerate(rows):
                info = infos[row]
                generator = info["duplexio_working_state"].sampling_generator
                emit = _sample_emit(
                    emit_logits[row:row + 1], 0.0 if greedy else _emit_temperatures(info).agent,
                    generator=generator,
                )
                if greedy:
                    token = content[index:index + 1]
                else:
                    selected = torch.multinomial(probabilities[index:index + 1], 1, generator=generator)
                    token = indices[index:index + 1].gather(-1, selected).squeeze(-1)
                tokens[row] = torch.where(emit, token, torch.full_like(token, self.silence_token_id))
        return [tokens[row] for row in range(len(infos))]

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
        voice = runtime_config.get("duplexio_voice")
        if not isinstance(voice, str) or voice not in self._voice_pools:
            raise ValueError(f"DuplexIO request selected unknown voice {voice!r}")
        pool = self._voice_pools[voice]
        embedding_index = runtime_config.get("duplexio_voice_embedding_index", 0)
        if not isinstance(embedding_index, int) or not 0 <= embedding_index < len(pool):
            raise ValueError(f"Voice embedding index {embedding_index!r} is invalid for {voice!r}")
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
        speaker_embedding = pool[embedding_index].to(
            device=device,
            dtype=self.llm.channel_emb.dtype,
        )
        if self.speaker_lda_projection is not None:
            speaker_embedding = (
                (speaker_embedding.float() - self.speaker_lda_mean) @ self.speaker_lda_projection.T
            ).to(self.llm.channel_emb.dtype)
        continuous = isinstance(self.audio_codec, PocketMimi)
        return DuplexIORequestState(
            text_input_ids=torch.full(
                (len(TEXT_STREAM_NAMES),),
                self.silence_token_id,
                dtype=torch.long,
                device="cpu",
            ),
            agent_audio_codes=self.initial_agent_audio(1)[0],
            user_asr=FastConformerAudioStreamState(),
            agent_delay=(None if continuous else self.audio_representation.new_state(device=device)),
            output_mimi=(self.audio_codec.new_state(1) if continuous else self.audio_codec.new_streaming_state()),
            speaker_embedding=speaker_embedding,
            depth_speaker_conditioning=(
                None if continuous else self.audio_sampler.prepare_speaker(speaker_embedding.unsqueeze(0))
            ),
            system_token_ids=cast(tuple[int, ...], tuple(system_tokens)),
            sampling_generator=sampling_generator,
            tool_call_constraint=self.tool_call_compiler.new_state(
                cast(list[Mapping[str, Any]], tools),
                tool_choice,
            ),
        )

    def initial_agent_audio(self, frames: int) -> Tensor:
        """The masked-prefix representation, not encoded silence."""
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Assemble six-cell inputs and cache addressing without changing CPU state.

    Besides the flattened embeddings this returns one entry per cell: whether the
    cell contributes a key, its 1-based text emission ordinal (0 when it emits
    nothing), how many text keys strictly earlier rows emitted, and the inclusive
    range of audio frames the cell may attend. Audio frames are numbered from
    one in audio time, which only advances on frames carrying real audio, so a
    text-only append leaves the range frozen and writes no audio key.
    """
    frames, device = text_ids.shape[0], text_ids.device
    text_hidden = text_hidden.masked_fill((text_ids == silence_token_id).unsqueeze(-1), 0)
    embeddings = torch.cat(
        (text_hidden + channel_embedding, user_hidden.unsqueeze(1), agent_hidden.unsqueeze(1)), dim=1,
    ).flatten(0, 1)
    text_active = (text_ids != pad_token_id) & (text_ids != silence_token_id)
    audio_shape = (frames, 2)
    key_active = torch.cat(
        (text_active, torch.full(audio_shape, audio_active, dtype=torch.bool, device=device)), dim=1,
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
    )


def _text_sampling(info: Mapping[str, object], suppressed_token_ids: Tensor) -> TokenSamplingOptions:
    duplex = info.get("duplex")
    if not isinstance(duplex, Mapping):
        raise ValueError("DuplexIO frame is missing runtime metadata")
    runtime = duplex.get("runtime_config")
    if not isinstance(runtime, Mapping):
        raise ValueError("DuplexIO frame is missing runtime configuration")
    sampling = runtime.get("duplexio_text_sampling")
    if not isinstance(sampling, Mapping):
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
    ):
        raise ValueError("Invalid DuplexIO text sampling configuration")
    return TokenSamplingOptions(
        mode=mode,
        temperature=float(temperature),
        top_k=top_k,
        top_p=float(top_p),
        suppressed_token_ids=suppressed_token_ids,
    )


def _emit_temperatures(info: Mapping[str, object]) -> EmitSamplingTemperatures:
    duplex = info.get("duplex")
    runtime = duplex.get("runtime_config") if isinstance(duplex, Mapping) else None
    temperatures = runtime.get("duplexio_emit_temperatures") if isinstance(runtime, Mapping) else None
    if not isinstance(temperatures, Mapping):
        raise ValueError("DuplexIO frame is missing emit temperatures")
    values = tuple(temperatures.get(name) for name in TARGET_STREAM_NAMES)
    if not all(isinstance(value, (int, float)) and value >= 0 for value in values):
        raise ValueError("Invalid DuplexIO emit temperatures")
    return EmitSamplingTemperatures(*(float(value) for value in values))


def _sample_content_token_ids(
    logits: Tensor,
    sampling: TokenSamplingOptions,
    *,
    generator: torch.Generator,
    distribution: Callable[[Tensor, TokenSamplingOptions], tuple[Tensor, Tensor]] = content_distribution,
) -> Tensor:
    if sampling.mode in {"argmax", "max"}:
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
        0.0 if sampling.mode in {"argmax", "max"} else emit_temperature,
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


def sample_tool_token_id(
    logits: Tensor,
    *,
    constraint: ToolCallConstraintState | None,
    emit: bool,
    sampling: TokenSamplingOptions,
    generator: torch.Generator,
    distribution: Callable[[Tensor, TokenSamplingOptions], tuple[Tensor, Tensor]] = content_distribution,
) -> Tensor | None:
    """Sample an active grammar token; the batch owner accepts the CPU token ID."""
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
    return _sample_content_token_ids(
        constrained_logits,
        sampling,
        generator=generator,
        distribution=distribution,
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


def _depth_sampling(info: Mapping[str, object]) -> tuple[float, int]:
    duplex = info.get("duplex")
    runtime = duplex.get("runtime_config") if isinstance(duplex, Mapping) else None
    sampling = runtime.get("duplexio_depth_sampling") if isinstance(runtime, Mapping) else None
    if not isinstance(sampling, Mapping):
        raise ValueError("DuplexIO append is missing depth sampling configuration")
    temperature = sampling.get("temperature")
    top_k = sampling.get("top_k")
    if not isinstance(temperature, (int, float)) or not isinstance(top_k, int):
        raise ValueError("Invalid DuplexIO depth sampling configuration")
    return float(temperature), top_k


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
