# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM implementation of the DuplexIO full-duplex model."""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from vllm.config import VllmConfig
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
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors
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
    user_delay: DelayedMimiState
    agent_delay: DelayedMimiState
    mimi: MimiStreamingState
    speaker_embedding: Tensor
    system_token_ids: tuple[int, ...]
    sampling_generator: torch.Generator
    system_token_offset: int = 0
    frames_seen: int = 0
    audio_position: int = 0
    active_text_tokens: int = 0
    cache_epoch: int = 0

    def fork(self) -> DuplexIORequestState:
        """Create an append-local state whose updates commit on success."""
        return DuplexIORequestState(
            text_input_ids=self.text_input_ids,
            agent_audio_codes=self.agent_audio_codes,
            user_delay=DelayedMimiState(
                previous_acoustic_codes=self.user_delay.previous_acoustic_codes,
                pending_semantic_code=self.user_delay.pending_semantic_code,
            ),
            agent_delay=DelayedMimiState(
                previous_acoustic_codes=self.agent_delay.previous_acoustic_codes,
                pending_semantic_code=self.agent_delay.pending_semantic_code,
            ),
            mimi=self.mimi.fork(),
            speaker_embedding=self.speaker_embedding,
            system_token_ids=self.system_token_ids,
            sampling_generator=_fork_generator(self.sampling_generator),
            system_token_offset=self.system_token_offset,
            frames_seen=self.frames_seen,
            audio_position=self.audio_position,
            active_text_tokens=self.active_text_tokens,
            cache_epoch=self.cache_epoch,
        )


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
    """Serve one DuplexIO frame per resumable vLLM request append."""

    packed_modules_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True
    postprocess_uses_hidden_states = False
    postprocess_uses_multimodal_outputs = False
    requires_request_sample_eligibility = True

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

        self.audio_codec = MimiModel(config.audio_codec_config).float()
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
        ).float()
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
        self.set_custom_preprocess(self.preprocess)
        self.set_custom_postprocess(self.postprocess)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.llm.base_model.model.embed_input_ids(input_ids)

    def update_attention_metadata(
        self,
        attn_metadata: object,
        request_infos: list[dict[str, Any]],
    ) -> None:
        """Expose accepted active-text counts to the compact Flex metadata."""
        active_text_tokens = []
        for info in request_infos:
            state = info.get("duplexio_model_state")
            active_text_tokens.append(
                state.active_text_tokens
                if isinstance(state, DuplexIORequestState)
                else 0
            )
        update_duplexio_attention_metadata(
            attn_metadata,
            active_text_tokens,
        )

    @torch.inference_mode()
    def preprocess(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        **info: Any,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        del input_embeds
        # Appends arrive with CPU token ids; bind the whole request state
        # (masks, speaker embedding, sampling generator) to the engine device
        # so nothing downstream mixes devices.
        input_ids = input_ids.to(self.llm.channel_emb.device)
        duplex = info.get("duplex")
        if not isinstance(duplex, Mapping):
            raise ValueError("Native DuplexIO accepts only framed duplex appends")
        frame_count = duplex.get("frame_count")
        if frame_count != 1 or input_ids.numel() != DUPLEXIO_NUM_CELLS:
            raise ValueError(
                "Native DuplexIO executes exactly one six-cell frame per append; "
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

        text_ids = state.text_input_ids.clone()
        if state.system_token_offset < len(state.system_token_ids):
            text_ids[0] = state.system_token_ids[state.system_token_offset]
            state.system_token_offset += 1
        else:
            text_ids[0] = self.silence_token_id

        codec_parameter = next(self.audio_codec.parameters())
        waveform = _decode_frame(
            duplex["payload"],
            input_ids.device,
            codec_parameter.dtype,
        )
        if waveform.shape[-1] != self.config.frame_size:
            raise ValueError(
                "DuplexIO append must contain exactly one PCM frame, got "
                f"{waveform.shape[-1]} samples"
            )
        raw_user_codes = self.audio_codec.encode(
            waveform,
            self.audio_representation.num_codebooks,
            state.mimi,
        )[0, :, 0]
        user_codes = self.audio_representation.encode_column(
            raw_user_codes,
            state.user_delay,
        )
        user_features = self.user_audio_embedding(user_codes)
        agent_features = self.agent_audio_embedding(state.agent_audio_codes)
        speaker = state.speaker_embedding.unsqueeze(0)
        request_index = torch.zeros(1, dtype=torch.long, device=input_ids.device)
        user_hidden = self.user_audio_input_adapter(user_features.unsqueeze(0))[0]
        agent_hidden, agent_skip = self.agent_audio_input_adapter(
            agent_features.unsqueeze(0),
            speaker,
            request_index,
        )

        text_hidden = self.llm.base_model.model.embed_input_ids(text_ids)
        text_hidden = text_hidden.masked_fill(
            (text_ids == self.silence_token_id).unsqueeze(-1),
            0,
        )
        text_hidden = text_hidden + self.llm.channel_emb
        embeddings = torch.cat((text_hidden, user_hidden, agent_hidden), dim=0)
        key_active = torch.cat(
            (
                (text_ids != self.pad_token_id)
                & (text_ids != self.silence_token_id),
                torch.ones(2, dtype=torch.bool, device=input_ids.device),
            )
        )
        text_active = key_active[: len(TEXT_STREAM_NAMES)]
        text_ordinals = torch.zeros(
            DUPLEXIO_NUM_CELLS,
            dtype=torch.long,
            device=input_ids.device,
        )
        frame_ordinals = (
            torch.cumsum(text_active.to(torch.long), dim=0)
            + state.active_text_tokens
        )
        text_ordinals[: len(TEXT_STREAM_NAMES)] = torch.where(
            text_active,
            frame_ordinals,
            0,
        )
        state.active_text_tokens += int(text_active.sum().item())
        state.frames_seen += 1
        # Audio time advances only on frames that carry real audio. Every
        # append is a PCM frame today; a future token-only append mode must
        # leave the counter frozen so burst frames do not consume the audio
        # attention window.
        state.audio_position += 1
        return input_ids, embeddings, {
            "duplexio_working_state": state,
            "duplexio_key_active": key_active,
            "duplexio_request_epochs": torch.full(
                (DUPLEXIO_NUM_CELLS,),
                state.cache_epoch,
                dtype=torch.long,
                device=input_ids.device,
            ),
            "duplexio_text_ordinals": text_ordinals,
            "duplexio_audio_positions": torch.full(
                (DUPLEXIO_NUM_CELLS,),
                state.audio_position,
                dtype=torch.long,
                device=input_ids.device,
            ),
            "duplexio_agent_audio_skip": agent_skip[0],
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
        infos = model_intermediate_buffer or []
        key_active = _gather_key_active(infos, inputs_embeds)
        request_epochs = _gather_frame_metadata(
            infos,
            "duplexio_request_epochs",
            inputs_embeds,
        )
        text_ordinals = _gather_frame_metadata(
            infos,
            "duplexio_text_ordinals",
            inputs_embeds,
        )
        audio_positions = _gather_frame_metadata(
            infos,
            "duplexio_audio_positions",
            inputs_embeds,
        )
        return self.llm.base_model.model(
            positions=duplexio_frame_positions(positions),
            logical_positions=positions,
            key_active=key_active,
            request_epochs=request_epochs,
            text_ordinals=text_ordinals,
            audio_positions=audio_positions,
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
        user_ids: list[Tensor] = []
        agent_ids: list[Tensor] = []
        tool_ids: list[Tensor] = []
        listen_flags: list[Tensor] = []
        end_flags: list[Tensor] = []
        epochs: list[Tensor] = []
        turn_ids: list[Tensor] = []
        for request_index, ((start, end), info) in enumerate(
            zip(request_token_spans, infos, strict=True)
        ):
            state = info.get("duplexio_working_state")
            if not isinstance(state, DuplexIORequestState):
                raise RuntimeError(
                    f"DuplexIO request {request_index} is missing working state"
                )
            if end - start != DUPLEXIO_NUM_CELLS:
                raise ValueError(
                    "DuplexIO request span must contain one complete frame, got "
                    f"({start}, {end})"
                )
            row_hidden = hidden_states[start:end]
            predicted_text = self._sample_text(
                row_hidden,
                info,
                state.sampling_generator,
            )
            # Session-side payloads cross the IPC hop as CPU tensors.
            speaker_embedding = state.speaker_embedding.to(row_hidden.device)
            audio_condition = self.agent_audio_output_adapter(
                row_hidden[AGENT_AUDIO_CELL : AGENT_AUDIO_CELL + 1],
                info["duplexio_agent_audio_skip"].to(row_hidden.device).unsqueeze(0),
                speaker_embedding.unsqueeze(0),
                torch.zeros(1, dtype=torch.long, device=row_hidden.device),
            )
            depth_sampling = _depth_sampling(info)
            predicted_audio = self.audio_sampler.sample(
                audio_condition.float(),
                predicted_text[1:2],
                speaker_embedding.unsqueeze(0).float(),
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
            waveform = (
                row_hidden.new_empty(0, dtype=torch.float32)
                if raw_audio_codes is None
                else self.audio_codec.decode(
                    raw_audio_codes.view(1, -1, 1),
                    state.mimi,
                )[0, 0]
            )
            forced_ids.append(int(predicted_text[1].item()))

            user_ids.append(predicted_text[0].detach())
            agent_ids.append(predicted_text[1].detach())
            tool_ids.append(predicted_text[2].detach())
            audio_outputs.append(waveform.detach())
            model_listen = bool(
                predicted_text[1].item() == self.silence_token_id
                and predicted_text[2].item() == self.silence_token_id
            )
            listen_flags.append(torch.tensor(model_listen))
            duplex = info.get("duplex", {})
            end_flags.append(torch.tensor(bool(duplex.get("final", False))))
            epochs.append(torch.tensor(int(duplex.get("epoch", 0))))
            turn_ids.append(torch.tensor(int(duplex.get("turn_id", 0))))

        self._forced_next_token_ids = forced_ids
        multimodal_outputs = cast(
            Any,
            {
                "audio": audio_outputs,
                "sample_rate_hz": [torch.tensor(self.config.sample_rate)]
                * len(audio_outputs),
                "user_token_id": user_ids,
                "agent_token_id": agent_ids,
                "tool_call_token_id": tool_ids,
                "model_listen": listen_flags,
                "end_of_turn": end_flags,
                "duplex_epoch": epochs,
                "duplex_turn_id": turn_ids,
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
        generator: torch.Generator,
    ) -> Tensor:
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
        temperature = _text_temperature(info)
        return _sample_factorized_text_ids(
            logits,
            emit_logits,
            silence_token_id=self.silence_token_id,
            temperature=temperature,
            generator=generator,
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
        cache_epoch = self._next_cache_epoch
        if cache_epoch >= 2**32:
            raise RuntimeError("DuplexIO cache epoch space is exhausted")
        self._next_cache_epoch += 1
        return DuplexIORequestState(
            text_input_ids=torch.full(
                (len(TEXT_STREAM_NAMES),),
                self.silence_token_id,
                dtype=torch.long,
                device=device,
            ),
            agent_audio_codes=self.audio_representation.initial_column(
                device=device
            ),
            user_delay=self.audio_representation.new_state(device=device),
            agent_delay=self.audio_representation.new_state(device=device),
            mimi=self.audio_codec.new_streaming_state(),
            speaker_embedding=pool[embedding_index].to(
                device=device,
                dtype=self.llm.channel_emb.dtype,
            ),
            system_token_ids=cast(tuple[int, ...], tuple(system_tokens)),
            sampling_generator=sampling_generator,
            cache_epoch=cache_epoch,
        )

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        # Exports carry the training-only system stream head; serving never
        # decodes the system stream, so the module doesn't exist here.
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["llm.output_head_proj.system."],
        )
        return loader.load_weights(weights)

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


def _gather_key_active(
    infos: list[dict[str, Any]],
    inputs_embeds: Tensor | None,
) -> Tensor:
    if inputs_embeds is None:
        raise ValueError("Native DuplexIO requires precomputed frame embeddings")
    values = [info.get("duplexio_key_active") for info in infos]
    if not infos:
        return torch.ones(
            inputs_embeds.shape[0],
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    if all(isinstance(value, Tensor) for value in values):
        key_active = torch.cat(cast(list[Tensor], values))
        if key_active.shape[0] == inputs_embeds.shape[0]:
            # Session-side appends build these on CPU; the engine runs on GPU.
            return key_active.to(inputs_embeds.device)
    raise RuntimeError(
        "DuplexIO key-activity rows do not match the scheduled embeddings"
    )


def _gather_frame_metadata(
    infos: list[dict[str, Any]],
    name: str,
    inputs_embeds: Tensor | None,
) -> Tensor:
    if inputs_embeds is None:
        raise ValueError("Native DuplexIO requires precomputed frame embeddings")
    if not infos:
        return torch.zeros(
            inputs_embeds.shape[0],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    values = [info.get(name) for info in infos]
    if all(isinstance(value, Tensor) for value in values):
        metadata = torch.cat(cast(list[Tensor], values))
        if metadata.shape == (inputs_embeds.shape[0],):
            # Session-side appends build these on CPU; the engine runs on GPU.
            return metadata.to(inputs_embeds.device)
    raise RuntimeError(
        f"DuplexIO {name} rows do not match the scheduled embeddings"
    )


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


def _text_temperature(info: Mapping[str, object]) -> float:
    duplex = info.get("duplex")
    if not isinstance(duplex, Mapping):
        return 0.0
    runtime = duplex.get("runtime_config")
    if not isinstance(runtime, Mapping):
        return 0.0
    value = runtime.get("duplexio_text_temperature", 0.0)
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError("duplexio_text_temperature must be non-negative")
    return float(value)


def _sample_factorized_text_ids(
    logits: Tensor,
    emit_logits: Tensor,
    *,
    silence_token_id: int,
    temperature: float,
    generator: torch.Generator,
) -> Tensor:
    """Sample emit/silence independently from the conditional content ID."""
    if temperature == 0:
        emit = emit_logits >= 0
    else:
        emit = torch.bernoulli(
            torch.sigmoid(emit_logits.float() / temperature),
            generator=generator,
        ).bool()

    content_logits = logits.clone()
    content_logits[:, silence_token_id] = -torch.inf
    if temperature == 0:
        content_ids = content_logits.argmax(dim=-1)
    else:
        content_ids = torch.multinomial(
            (content_logits / temperature).softmax(dim=-1),
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
    return torch.where(
        emit,
        content_ids,
        torch.full_like(content_ids, silence_token_id),
    )


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
    if vllm_config.scheduler_config.enable_chunked_prefill:
        raise ValueError("Native DuplexIO requires chunked prefill to be disabled")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("Native DuplexIO requires prefix caching to be disabled")
    if (
        vllm_config.parallel_config.decode_context_parallel_size != 1
        or vllm_config.parallel_config.prefill_context_parallel_size != 1
    ):
        raise ValueError("Native DuplexIO does not support context parallelism")
    if vllm_config.parallel_config.use_ubatching:
        raise ValueError("Native DuplexIO does not support microbatching")


def _fork_generator(generator: torch.Generator) -> torch.Generator:
    fork = torch.Generator(device=generator.device)
    fork.set_state(generator.get_state())
    return fork


__all__ = ["DuplexIOForConditionalGeneration", "DuplexIORequestState"]
