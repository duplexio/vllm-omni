# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DuplexIO, a full-duplex speech model that listens and speaks in the same 80 ms frames."""

from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, cast

import torch
import torch.nn.functional as F
import xgrammar as xgr
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

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_representation import ContinuousAudioRepresentation
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import (
    DuplexIOConfig,
)
from vllm_omni.model_executor.models.duplexio.fastconformer import SAMPLE_RATE as ASR_SAMPLE_RATE
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
    FastConformerEncoder,
    streaming_resample_batch,
)
from vllm_omni.model_executor.models.duplexio.flowmap import FlowMapSampler
from vllm_omni.model_executor.models.duplexio.frame_layout import (
    AGENT_AUDIO_CELL,
    AGENT_CELL,
    FRAME_SIZE,
    NUM_CELLS,
    SAMPLE_RATE,
    TEXT_STREAM_NAMES,
    TOOL_CALL_CELL,
)
from vllm_omni.model_executor.models.duplexio.pocket_mimi import LATENT_DIM, ContinuousMimiState, PocketMimi
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOQwenModel,
)
from vllm_omni.model_executor.models.duplexio.sampling_config import SamplingConfig
from vllm_omni.model_executor.models.duplexio.stream_gdn import gdn_cache_dtypes, gdn_cache_shapes
from vllm_omni.model_executor.models.duplexio.text_sampling import (
    FLOW_TEMPERATURE,
    TokenSamplingOptions,
    sample_streams,
    sampled_top_k,
    sampling_parameters,
    support_width,
)
from vllm_omni.model_executor.models.duplexio.tool_calling import (
    ToolCallConstraintCompiler,
    ToolCallConstraintState,
    token_bitmasks,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

OUTPUT_STREAM_NAMES = ("agent", "tool_call")


@dataclass
class TextSamplingResult:
    """Sampled text and log probabilities, batched in request order.

    Each text_ids row holds the user, agent, and tool token IDs.
    """

    text_ids: Tensor
    tool_starts: Tensor  # Idle rows that started a call; the host reads only pending ones.
    # Emit and token log probabilities of the agent, tool and user streams. The
    # tool emit is scored with the raw head, including forced decisions.
    frame_logprobs: Tensor
    support_ids: Tensor  # [rows, 2, width] agent and tool content supports, see sample_streams.
    # Idle rows, the only ones whose tool emit was drawn rather than forced.
    pending_tool_starts: list[int]
    # Rows already inside a call, whose sampled token the host must accept.
    tool_rows: list[int]
    tool_calls: list[dict[str, Any] | None]


@dataclass
class SamplingInputs:
    """A batch's per-row policy and tool state, uploaded before the draws."""

    parameters: Tensor  # [rows, SAMPLING_PARAMETERS], see text_sampling.
    tool_bitmask: Tensor
    top_k: int | None
    support_width: int
    pending: list[int]
    calling: list[int]


@dataclass
class FrameBatch:
    """Queued samples for the eligible requests, before any host read."""

    indices: list[int]  # Request index of each row.
    text: TextSamplingResult
    audio: Tensor
    hiddens: Tensor  # [rows, cells, hidden] backbone outputs.


@dataclass(eq=False)
class DuplexIOStepOutput:
    """A step's queued draws; ``finalize`` applies them on the host once.

    Under async scheduling the runner finalizes after the next step's launch.
    A request scheduled again before that finalizes its step early.
    """

    batch: FrameBatch | None
    infos: list[dict[str, Any]]
    states: list[DuplexIORequestState]
    rows: dict[int, int]
    decode_rows: list[int]
    record_hiddens: list[bool]
    replays: list[PackedRows] | None
    policy_version: int
    empty: Tensor
    host: PendingHostOutputs
    result: dict[str, Any] | None = None

    def finalize(self, model: DuplexIOForConditionalGeneration) -> dict[str, Any]:
        if self.result is None:
            self.result = model._finish_step(self)
        return self.result


@dataclass
class DuplexIORequestState:
    """Request-owned state; the backbone's KV and GDN caches belong to the runner."""

    text_input_ids: tuple[int, ...]  # Host feedback: the ids the next frame feeds back.
    agent_latent: Tensor
    user_asr: FastConformerAudioStreamState
    input_mimi: ContinuousMimiState
    output_mimi: ContinuousMimiState
    # Raw reference audio; text-only context never advances either encoder.
    voice_prompt: Tensor
    system_token_ids: tuple[int, ...]
    sampling: RequestSampling
    tool_call_constraint: ToolCallConstraintState | None = None
    frames_seen: int = 0
    audio_position: int = 0
    # Keys in the never-expiring cache region: voice prompt, then emitted text.
    persistent_keys: int = 0
    tool_call_sequence: int = 0
    # Every live frame so far had given inputs. Only given agent audio advances
    # the input codec, so a given frame may not follow a sampled one.
    history_open: bool = True
    # The step whose sampled ids this request still waits for on the host.
    pending_output: DuplexIOStepOutput | None = None

    def fork(self) -> DuplexIORequestState:
        """Commit an append only after its model step succeeds."""
        result = copy.copy(self)
        if self.tool_call_constraint is not None:
            result.tool_call_constraint = self.tool_call_constraint.fork()
        # Codec updates are functional; ASR batches own their mutable HF caches.
        return result


@dataclass
class PreparedAudio:
    """One scheduled slice of an append, with audio encoded before per-request framing.

    An append's rows are ``[voice prompt, system tokens]`` (its first append
    only), then one live frame, then any tool-result tokens.
    """

    state: DuplexIORequestState
    frame_start: int
    frame_count: int
    prompt_count: int  # Voice-prompt rows of the whole append.
    prefix_count: int  # Voice-prompt and system rows of the whole append.
    prompt_chunk_frames: int  # Voice-prompt rows in this slice, which lead it.
    live_row: int | None  # The live frame's row in this slice, if it holds it.
    user_features: Tensor | None  # Rows without acoustic input are zero.
    agent_latents: Tensor


class PackedRows(Mapping[str, Tensor]):
    """One request's rows of tensors packed for the whole step.

    Not a dict, so the runner's request buffer holds it by reference instead of
    copying each view, and consumers can take the packed batch back whole.
    """

    __slots__ = ("packed", "rows")

    def __init__(self, packed: dict[str, Tensor], rows: slice) -> None:
        self.packed = packed
        self.rows = rows

    def __getitem__(self, name: str) -> Tensor:
        return self.packed[name][self.rows]

    def __iter__(self) -> Iterator[str]:
        return iter(self.packed)

    def __len__(self) -> int:
        return len(self.packed)


def shared_packed(values: list[Any], *, contiguous: bool) -> dict[str, Tensor] | None:
    """The packed batch behind every value, if they all share one.

    With ``contiguous``, the values must also tile its leading rows in order.
    """
    if not values or not all(isinstance(value, PackedRows) for value in values):
        return None
    packed = values[0].packed
    if any(value.packed is not packed for value in values):
        return None
    if not contiguous:
        return packed
    stop = 0
    for value in values:
        if value.rows.start != stop:
            return None
        stop = value.rows.stop
    return {name: tensor[:stop] for name, tensor in packed.items()}


@dataclass
class PreparedFrames:
    """Views into one packed batch, consumed by the runner's request hook."""

    embeddings: Tensor
    updates: dict[str, Any]


class FrameInputGraph:
    """Capture a tensor-only function; CPU request state remains outside the graph.

    Graphs may share a memory ``pool``: replays run one at a time on one stream,
    and only the cloned outputs outlive a replay.
    """

    def __init__(
        self, project: Callable[..., tuple[Tensor, ...]], inputs: tuple[Tensor, ...],
        pool: tuple[int, int] | None = None,
    ) -> None:
        self.inputs = tuple(value.clone() for value in inputs)
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                project(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.outputs = project(*self.inputs)

    def __call__(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        torch._foreach_copy_(self.inputs, inputs)
        self.graph.replay()
        # Outputs can remain in request state after another batch replays.
        return tuple(value.clone() for value in self.outputs)


@dataclass(frozen=True)
class RequestSampling:
    """A session's resolved sampling; suppression masks remain model-owned."""

    agent: TokenSamplingOptions
    user: TokenSamplingOptions
    flow_temperature: float
    parameters: tuple[float, ...]  # The row sample_streams reads, without the tool state.


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
    # make_omni_output returns fresh host tensors each step, so the runner can
    # hand them to the payload without another copy.
    omni_host_owned_multimodal_outputs = True
    # Under async scheduling the runner finalizes a step (finalize_multimodal_outputs_from_cpu_snapshot)
    # on the engine thread after launching the next one, so each step's device
    # work runs under the other's host work. Without it, it finalizes at once.
    use_async_omni_output = True
    supports_async_whole_payload = True
    eager_omni_postprocess_before_async_output = True
    omni_async_output_build_in_background = False
    # Per-request embeddings are views of one packed step tensor.
    preprocess_outputs_stable_within_step = True
    has_preprocess = True
    decode_query_len = NUM_CELLS
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
        self.frame_inputs = (
            torch.compile(frame_inputs, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
            if self.full_cudagraph_enabled else frame_inputs
        )
        self.frame_input_graphs: dict[int, FrameInputGraph] = {}
        self.sampling_graphs: dict[tuple[int, int | None], FrameInputGraph] = {}
        # Addressing and replay inputs are PackedRows, which the runner holds by
        # reference; nothing needs a per-request device copy.
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = set()
        self.pad_token_id = config.pad_token_id
        self.silence_token_id = config.silence_token_id

        self.llm = _DuplexIOMultiStreamQwen(
            vllm_config=vllm_config,
            prefix=f"{prefix}.llm" if prefix else "llm",
        )
        # Cell addressing for the backbone's cache, filled once per step and
        # read by every full-attention layer.
        self.frame = self.llm.base_model.model.frame
        hidden_size = self.text_config.hidden_size
        adapter_hidden_size = config.audio_adapter_config.get("hidden_size") or hidden_size
        self.user_asr = FastConformerEncoder.from_checkpoint(
            config.user_asr_config, vllm_config.model_config.model, revision=vllm_config.model_config.revision,
            use_cuda_graph=self.full_cudagraph_enabled,
        )
        self.user_audio_resampler = Resample(SAMPLE_RATE, ASR_SAMPLE_RATE, dtype=torch.float32).to(
            device=self.llm.channel_emb.device,
        )
        self.audio_codec = PocketMimi()
        self.audio_representation = ContinuousAudioRepresentation(LATENT_DIM)
        self.audio_sampler = FlowMapSampler(
            LATENT_DIM,
            hidden_size,
            config.flowmap_config["mlp_dim"],
            config.flowmap_config["mlp_depth"],
            inference_steps=config.flowmap_config["inference_steps"],
            compile=self.full_cudagraph_enabled,
        )
        self.user_audio_input_adapter = AudioInputAdapter(
            self.user_asr.output_dim,
            adapter_hidden_size,
            hidden_size,
        )
        self.agent_audio_input_adapter = AudioInputAdapter(
            LATENT_DIM,
            adapter_hidden_size,
            hidden_size,
        )
        self.user_token_projection = nn.Linear(NUM_CELLS * hidden_size, hidden_size)
        self.user_emit_head = nn.Linear(NUM_CELLS * hidden_size, 1)
        self.agent_emit_head = nn.Linear(NUM_CELLS * hidden_size, 1)
        self.tool_call_emit_head = nn.Linear(NUM_CELLS * hidden_size, 1)
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
        self.make_empty_intermediate_tensors = self.llm.base_model.model.make_empty_intermediate_tensors
        self._forced_next_token_ids: list[int] | Tensor | None = None
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
        # The user stream suppresses only the silence token.
        self.register_buffer(
            "user_suppressed_token_ids",
            torch.tensor([self.silence_token_id], dtype=torch.long, device=self.llm.channel_emb.device),
            persistent=False,
        )
        vocab_size = self.text_config.vocab_size
        self.init_text_sampling(vocab_size, vllm_config.scheduler_config.max_num_seqs)
        self.tool_call_compiler = ToolCallConstraintCompiler(self.tokenizer, vocab_size)
        self.set_custom_preprocess(self.preprocess)
        self.set_custom_postprocess(self.postprocess)

    def init_text_sampling(self, vocab_size: int, max_rows: int) -> None:
        """Device constants of ``sample_text``, from the suppressed-id buffers."""
        streams = (self.agent_suppressed_token_ids, self.tool_suppressed_token_ids, self.user_suppressed_token_ids)
        device = streams[0].device
        suppressed = torch.zeros(len(streams), vocab_size, dtype=torch.bool, device=device)
        for stream, ids in enumerate(streams):
            suppressed[stream, ids] = True
        # Rows in logits order (agent, tool, user).
        self.register_buffer("suppressed_token_mask", suppressed, persistent=False)
        self.register_buffer(
            "bitmask_shifts", torch.arange(32, dtype=torch.int32, device=device), persistent=False,
        )
        # Batches without a tool grammar replay with this mask.
        self.register_buffer(
            "allow_all_bitmask",
            torch.full(xgr.get_bitmask_shape(max_rows, vocab_size), -1, dtype=torch.int32, device=device),
            persistent=False,
        )

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
        # Requests scheduled in the order they were prepared already form the batch.
        cells = shared_packed(duplex_infos, contiguous=True) or {
            name: torch.cat([info[name] for info in duplex_infos]) for name in duplex_infos[0]
        }
        self.frame.update(
            key_active=cells["key_active"],
            persistent_ordinal=cells["persistent_ordinal"],
            persistent_last=cells["persistent_last"],
            audio_first=cells["audio_first"],
            audio_last=cells["audio_last"],
        )
        positions = cells["positions"]
        self.frame.positions[:positions.numel()].copy_(positions)

    def prepare_audio_request(self, tokens: int, info: dict[str, Any], device: torch.device) -> PreparedAudio:
        """Validate framing and fork state at the model's input boundary."""
        duplex = info["duplex"]
        append_frames = duplex["frame_count"]
        token_offset = info["duplex_token_offset"]
        prompt_len = info["duplex_prompt_len"]
        if (
            tokens == 0 or tokens % NUM_CELLS
            or token_offset % NUM_CELLS or prompt_len % NUM_CELLS
        ):
            raise ValueError(
                "DuplexIO requires complete six-cell frames; "
                f"got frame_count={append_frames}, tokens={tokens}, offset={token_offset}, end={prompt_len}"
            )
        frame_count = tokens // NUM_CELLS
        frame_start = append_frames - (prompt_len - token_offset) // NUM_CELLS
        frame_end = frame_start + frame_count
        assert 0 <= frame_start < frame_end <= append_frames
        prefix = duplex["duplexio_prefix"]
        state = info.get("duplexio_model_state")
        if not isinstance(state, DuplexIORequestState):
            if not prefix:
                raise ValueError("DuplexIO requires the voice and system prefix before live input")
            state = self._new_request_state(duplex["runtime_config"], device)
        else:
            if state.pending_output is not None:
                # Scheduled again before its last step was finalized: its
                # sampled ids are this frame's inputs, so wait for them here.
                state.pending_output.finalize(self)
            state = state.fork()
        prompt_count = state.voice_prompt.numel() // FRAME_SIZE if prefix else 0
        prefix_count = prompt_count + len(state.system_token_ids) if prefix else 0
        if append_frames != prefix_count + 1 + len(duplex["duplexio_tool_token_ids"]):
            raise ValueError("DuplexIO append rows do not match its prefix, live frame and tool result")
        if prefix and state.frames_seen != frame_start:
            raise ValueError("DuplexIO prefix must lead the request exactly once")
        live_row = prefix_count - frame_start if frame_start <= prefix_count < frame_end else None
        given = duplex.get("duplexio_given_frame")
        if live_row is not None:
            if given is None:
                state.history_open = False
            else:
                self.validate_given_frame(given, state)
        if live_row is not None and frame_count == 1:
            agent_latents = state.agent_latent.unsqueeze(0)
        else:
            agent_latents = self.initial_agent_latents(frame_count)
            if live_row is not None:
                agent_latents[live_row] = state.agent_latent
        return PreparedAudio(
            state, frame_start, frame_count, prompt_count, prefix_count,
            max(0, min(frame_end, prompt_count) - frame_start), live_row, None, agent_latents,
        )

    def validate_given_frame(self, given: Mapping[str, Any], state: DuplexIORequestState) -> None:
        """A live frame whose inputs are given: conversation history replayed before the model runs free."""
        if not state.history_open:
            raise ValueError("DuplexIO given frames must lead the conversation, before any sampled frame")
        if state.tool_call_constraint is not None and state.tool_call_constraint.enabled:
            raise ValueError("DuplexIO given frames need a session without tools")
        if given["tool_call_token_id"] != self.silence_token_id:
            raise ValueError("DuplexIO given frames cannot hold tool calls")
        if len(given["agent_pcm"]) != FRAME_SIZE * 4:
            raise ValueError(f"DuplexIO given agent audio must be one {FRAME_SIZE}-sample float32 frame")

    @torch.inference_mode()
    def prepare_audio_requests(
        self, requests: list[tuple[int, dict[str, Any]]], device: torch.device,
    ) -> list[PreparedAudio]:
        prepared = [self.prepare_audio_request(tokens, info, device) for tokens, info in requests]
        live = {
            index: info["duplex"]["pcm"] for index, (_, info) in enumerate(requests)
            if prepared[index].live_row is not None
        }
        # Agent audio of the given live frames, heard instead of the last prediction.
        given = {
            index: info["duplex"]["duplexio_given_frame"]["agent_pcm"] for index, (_, info) in enumerate(requests)
            if prepared[index].live_row is not None and info["duplex"].get("duplexio_given_frame") is not None
        }
        # The user encoder reads only this step's audio and its own caches, so on
        # CUDA it runs on a side stream, overlapping the previous step's backbone.
        cuda = device.type == "cuda"
        if cuda and any(item.prompt_chunk_frames for item in prepared):
            # Voice prompts were uploaded on the main stream.
            self.user_audio_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.user_audio_stream) if cuda else contextlib.nullcontext():
            if live:
                # One upload for the step. A blocking one waits for every queued
                # kernel, including the previous step's; a pageable non_blocking one
                # stages the bytes and returns at once.
                chunks = [*live.values(), *given.values()]
                pcm = torch.frombuffer(bytearray().join(chunks), dtype=torch.float32)
                sizes = [len(chunk) // pcm.element_size() for chunk in chunks]
                uploaded = pcm.to(device, non_blocking=True).split(sizes)
                live = dict(zip(live, uploaded[:len(live)], strict=True))
                given = dict(zip(given, uploaded[len(live):], strict=True))
            acoustic = []
            user_waveforms = []
            prompt_waveforms: dict[int, Tensor] = {}
            for index, item in enumerate(prepared):
                # Acoustic rows, in stream order: the slice's voice-prompt rows
                # hear silence, then the live row hears the client.
                waveforms = []
                if item.prompt_chunk_frames:
                    start = item.frame_start * FRAME_SIZE
                    count = item.prompt_chunk_frames * FRAME_SIZE
                    prompt_waveform = item.state.voice_prompt[start:start + count]
                    prompt_waveforms[index] = prompt_waveform
                    waveforms.append(torch.zeros_like(prompt_waveform))
                if index in live:
                    waveforms.append(live[index])
                if waveforms:
                    acoustic.append(index)
                    user_waveforms.append(waveforms[0] if len(waveforms) == 1 else torch.cat(waveforms))
            user_features = self.encode_user_audio_batch(
                user_waveforms, [prepared[index].state for index in acoustic],
            ) if acoustic else []
        if cuda and acoustic:
            main = torch.cuda.current_stream(device)
            main.wait_stream(self.user_audio_stream)
            for features in (*user_features, *given.values()):
                features.record_stream(main)
        for index, features in zip(acoustic, user_features, strict=True):
            prepared[index].user_features = self.place_user_features(prepared[index], features)
        # The agent codec hears, in stream order, the slice's voice-prompt rows and
        # then a given live frame.
        heard = sorted({*prompt_waveforms, *given})
        if heard:
            waveforms = [
                torch.cat([waveform for waveform in (prompt_waveforms.get(index), given.get(index)) if waveform is not None])
                for index in heard
            ]
            codes = self.encode_agent_audio_batch(waveforms, [prepared[index].state for index in heard])
            for index, audio in zip(heard, codes, strict=True):
                item = prepared[index]
                voice = item.prompt_chunk_frames
                item.agent_latents = torch.cat((audio[:voice], item.agent_latents[voice:]))
                if index in given:
                    assert item.live_row is not None
                    item.agent_latents[item.live_row] = audio[voice]
        return prepared

    def place_user_features(self, item: PreparedAudio, features: Tensor) -> Tensor:
        """Lay encoder outputs onto the slice's acoustic rows in order; text-only rows and any shortfall are zero."""
        acoustic = item.prompt_chunk_frames + (item.live_row is not None)
        if features.shape[0] < acoustic:
            features = F.pad(features, (0, 0, 0, acoustic - features.shape[0]))
        if acoustic == item.frame_count:
            return features
        voice = item.prompt_chunk_frames
        if item.live_row is None:
            return torch.cat((features, features.new_zeros(item.frame_count - voice, features.shape[1])))
        return torch.cat((
            features[:voice],
            features.new_zeros(item.live_row - voice, features.shape[1]),
            features[voice:],
            features.new_zeros(item.frame_count - item.live_row - 1, features.shape[1]),
        ))

    @cached_property
    def user_audio_stream(self) -> torch.cuda.Stream:
        return torch.cuda.Stream()

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
        silence = self.silence_token_id
        quiet = (silence,) * (len(TEXT_STREAM_NAMES) - 1)
        # One row per packed frame: the text ids, then frame_inputs' metadata.
        rows = []
        preceding_keys = 0
        for item, (_, info) in zip(audio, requests, strict=True):
            state = item.state
            duplex = info["duplex"]
            tool_ids = duplex["duplexio_tool_token_ids"]
            text = []
            for frame in range(item.frame_start, item.frame_start + item.frame_count):
                if frame < item.prompt_count:
                    text.append((silence, *quiet))
                elif frame < item.prefix_count:
                    text.append((state.system_token_ids[frame - item.prompt_count], *quiet))
                elif frame == item.prefix_count:
                    given = duplex.get("duplexio_given_frame")
                    if given is None:
                        text.append((silence, *state.text_input_ids[1:]))
                    else:
                        text.append((silence, given["user_token_id"], given["agent_token_id"], given["tool_call_token_id"]))
                else:
                    text.append((tool_ids[frame - item.prefix_count - 1], *quiet))
            rows.extend(
                (*ids, state.persistent_keys - preceding_keys,
                 state.audio_position if row == item.live_row else 0, row == item.live_row,
                 row < item.prompt_chunk_frames, state.frames_seen + row)
                for row, ids in enumerate(text)
            )
            written = item.prompt_chunk_frames + sum(
                token != self.pad_token_id and token != silence for ids in text for token in ids
            )
            state.persistent_keys += written
            preceding_keys += written
            state.frames_seen += item.frame_count
            if item.live_row is not None:
                state.audio_position += 1
        packed = torch.tensor(rows, dtype=torch.long).to(device, non_blocking=True)
        text_ids, metadata = packed[:, :len(TEXT_STREAM_NAMES)], packed[:, len(TEXT_STREAM_NAMES):]
        user_features = torch.cat([
            self.llm.channel_emb.new_zeros(item.frame_count, self.user_asr.output_dim)
            if item.user_features is None else item.user_features
            for item in audio
        ])
        agent_latents = torch.cat([item.agent_latents for item in audio])
        with torch.profiler.record_function("duplexio.frame_inputs"):
            inputs = (packed, user_features, agent_latents)
            # Capture bounded one-frame batches; variable-length prefixes stay packed.
            if self.full_cudagraph_enabled and all(item.frame_count == 1 for item in audio):
                outputs = self.frame_input_graph(inputs)(inputs)
            else:
                outputs = self.project_frames(*inputs)
        embeddings, *addressing = outputs
        packed_cells = dict(zip(
            ("positions", "key_active", "persistent_ordinal", "persistent_last", "audio_first", "audio_last"),
            addressing, strict=True,
        ))
        records = [info["duplex"]["runtime_config"].get("duplexio_record_inputs", False) for _, info in requests]
        packed_replay = {
            "text_ids": text_ids, "user_features": user_features, "agent_audio": agent_latents,
            "audio_mask": metadata[:, 2] != 0, "prompt_frames": metadata[:, 3] != 0,
        } if any(records) else {}
        results = []
        offset = 0
        for item, record in zip(audio, records, strict=True):
            end = offset + item.frame_count
            cells = slice(offset * NUM_CELLS, end * NUM_CELLS)
            results.append(PreparedFrames(embeddings[cells], {
                "duplexio_working_state": item.state,
                "duplexio_replay": PackedRows(packed_replay, slice(offset, end)) if record else {},
                "duplexio": PackedRows(packed_cells, cells),
            }))
            offset = end
        return results

    def frame_input_graph(self, inputs: tuple[Tensor, ...]) -> FrameInputGraph:
        """The one-frame graph for this batch size, captured on first use into the shared pool."""
        size = inputs[0].shape[0]
        if size not in self.frame_input_graphs:
            pool = next(iter(self.frame_input_graphs.values())).graph.pool() if self.frame_input_graphs else None
            self.frame_input_graphs[size] = FrameInputGraph(self.project_frames, inputs, pool)
        return self.frame_input_graphs[size]

    @torch.inference_mode()
    def capture_frame_graphs(self) -> None:
        """Compile and capture frame construction for every batch size before serving starts.

        Otherwise the first batch of each size would stall all live streams while
        it compiles. The runner calls this after its own capture, under the
        default dtype that serving runs with (weight loading changes it, which
        compiled code guards on). Sampling graphs also key on the batch's top-k,
        which only requests know; they are captured here for the default
        sampling policy, which the demo page and unconfigured sessions use.
        """
        if not self.full_cudagraph_enabled or self.frame_input_graphs:
            return
        device = self.llm.channel_emb.device
        max_rows = self.vllm_config.scheduler_config.max_num_seqs
        # Silent live frames: text ids, then frame_inputs' metadata.
        silence = (self.silence_token_id,) * len(TEXT_STREAM_NAMES)
        packed = torch.tensor([(*silence, 0, 1, 1, 0, 0)] * max_rows, dtype=torch.long, device=device)
        user_features = self.llm.channel_emb.new_zeros(max_rows, self.user_asr.output_dim)
        agent_latents = self.initial_agent_latents(max_rows)
        for size in range(1, max_rows + 1):
            self.frame_input_graph((packed[:size], user_features[:size], agent_latents[:size]))
        sampling = self.resolve_sampling(SamplingConfig().model_dump())
        top_k = sampled_top_k((sampling.agent, sampling.user), self.text_config.vocab_size)
        width = support_width([sampling.agent], top_k)
        parameters = torch.tensor(
            [(*sampling.parameters, sampling.flow_temperature, 0)] * max_rows, dtype=torch.float32, device=device,
        )
        rows = self.llm.channel_emb.new_zeros(max_rows, NUM_CELLS, self.text_config.hidden_size)
        for size in range(1, max_rows + 1):
            tensors = (rows[:size], parameters[:size], self.allow_all_bitmask[:size])
            self.sampling_graph(tensors, top_k, width)

    def project_frames(self, packed: Tensor, user_features: Tensor, agent_latents: Tensor) -> tuple[Tensor, ...]:
        """Tensor-only packed projections and frame construction, shared by all requests."""
        # Split here, so eager and captured calls hand frame_inputs the same strides.
        text_ids, metadata = packed[:, :len(TEXT_STREAM_NAMES)], packed[:, len(TEXT_STREAM_NAMES):]
        with torch.autocast(
            device_type=text_ids.device.type,
            dtype=self.vllm_config.model_config.dtype,
            enabled=text_ids.is_cuda and self.vllm_config.model_config.dtype != torch.float32,
        ):
            user_hidden = self.user_audio_input_adapter(user_features)
            agent_hidden = self.agent_audio_input_adapter(agent_latents)
        text_hidden = self.llm.base_model.model.embed_input_ids(text_ids.flatten()).view(
            text_ids.shape[0], len(TEXT_STREAM_NAMES), -1
        )
        embeddings, *addressing = self.frame_inputs(
            text_ids, text_hidden, self.llm.channel_emb, user_hidden, agent_hidden,
            self.pad_token_id, self.silence_token_id,
            metadata, self.config.audio_attention_window_frames,
        )
        positions = metadata[:, 4, None] * NUM_CELLS + torch.arange(NUM_CELLS, device=metadata.device)
        return embeddings, positions.flatten(), *addressing

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
            raise ValueError("DuplexIO requires precomputed frame embeddings")
        del model_intermediate_buffer
        with torch.profiler.record_function("duplexio.backbone"):
            return self.llm.base_model.model(
                positions=self.frame.positions[:inputs_embeds.shape[0]] // NUM_CELLS,
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
            duplex = info.get("duplex", {})
            if not isinstance(duplex, Mapping):
                raise RuntimeError(f"DuplexIO request {request_index} is missing duplex metadata")
            start, end = request_token_spans[request_index]
            if end - start < NUM_CELLS or (end - start) % NUM_CELLS:
                raise ValueError(f"DuplexIO request span must contain complete frames, got ({start}, {end})")
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
                "frame_logprobs": [text.frame_logprobs],
                "support_ids": [text.support_ids],
            }
            if any(record_hiddens[index] for index in batch.indices):
                sampled["predictor_hiddens"] = [batch.hiddens.detach()]
        replay_names = ("text_ids", "user_features", "agent_audio", "audio_mask", "prompt_frames")
        empty = torch.empty(0, dtype=hidden_states.dtype)
        replays = [info.get("duplexio_replay", {}) for info in infos]
        packed_replay = shared_packed(replays, contiguous=False)
        if packed_replay is None:
            replay = {name: [value.get(name, empty) for value in replays] for name in replay_names}
        else:
            replay = {name: [packed_replay[name]] for name in replay_names}
        step = DuplexIOStepOutput(
            batch=batch, infos=infos, states=states, rows=rows, decode_rows=decode_rows,
            record_hiddens=record_hiddens, replays=replays if packed_replay is not None else None,
            policy_version=self.policy_version, empty=empty,
            host=to_host_async({"sampled": sampled, "replay": replay}),
        )
        # The next frame's inputs that are known on the device now; the sampled
        # ids reach the request once the step is finalized on the host.
        for index, row in rows.items():
            states[index].agent_latent = predicted_audio[row]
            states[index].pending_output = step
        if batch is None:
            self._forced_next_token_ids = [self.silence_token_id] * len(infos)
        else:
            forced = torch.full((len(infos),), self.silence_token_id, dtype=torch.long, device=hidden_states.device)
            agent_ids = batch.text.text_ids[:, 1]
            if batch.indices == list(range(len(batch.indices))):
                forced[: len(batch.indices)] = agent_ids
            else:
                forced.index_copy_(0, _to_device(batch.indices, torch.long, forced.device), agent_ids)
            self._forced_next_token_ids = forced
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=cast(Any, step),
            # Four persistent slots per scheduler row; audio KV has a fixed
            # reserved ring. Keep one row to preserve the recurrent-state marker.
            streaming_retained_tokens=[
                max(1, (state.persistent_keys + 3) // 4) * NUM_CELLS for state in states
            ],
            streaming_position_budget=[
                (self.text_config.max_position_embeddings - state.frames_seen) * NUM_CELLS
                for state in states
            ],
        )

    def finalize_multimodal_outputs_from_cpu_snapshot(self, outputs: Any) -> Any:
        """The host half of ``make_omni_output``: wait for the draws, then feed them back."""
        return outputs.finalize(self) if isinstance(outputs, DuplexIOStepOutput) else outputs

    def _finish_step(self, step: DuplexIOStepOutput) -> dict[str, Any]:
        host = step.host.wait()
        batch, infos, states, rows = step.batch, step.infos, step.states, step.rows
        if step.replays is not None:
            host["replay"] = {
                name: [values[0][value.rows] for value in step.replays] for name, values in host["replay"].items()
            }
        host_rows: dict[str, list[Tensor]] = {}
        host_ids: list[list[int]] = []
        decoded: dict[int, Tensor] = {}
        if batch is not None:
            sampled = host["sampled"]
            host_ids = sampled["text_ids"][0].tolist()
            self.finish_text_batch(
                batch.text, [infos[index] for index in batch.indices], host_ids, sampled["tool_starts"][0].tolist(),
            )
            host_rows = {
                name: list(values[0].unbind(0))
                for name, values in sampled.items() if name not in ("text_ids", "tool_starts", "waveforms")
            }
            decoded = dict(zip(step.decode_rows, sampled["waveforms"], strict=True))

        # Placeholders and request metadata start on the host.
        empty = step.empty
        empty_audio = torch.empty(0, dtype=torch.float32)
        empty_codes = torch.empty(0, dtype=torch.long)
        no_logprobs = torch.empty(0, dtype=torch.float32)
        no_support = torch.empty(0, dtype=torch.int32)
        drawn_tool_emits = set() if batch is None else set(batch.text.pending_tool_starts)
        no_tool_call = torch.empty(0, dtype=torch.uint8)
        silence = self.silence_token_id
        record_hiddens = any(step.record_hiddens)
        frames: list[list[int]] = []
        chunk: dict[str, list[Tensor]] = {
            name: [] for name in (
                "frame_logprobs", "frame_support_ids", "agent_audio_token_ids", "tool_call_json",
                *(("predictor_hiddens",) if record_hiddens else ()),
            )
        }
        audio_outputs: list[Tensor] = []
        for request_index, (info, state) in enumerate(zip(infos, states, strict=True)):
            if state.pending_output is step:
                state.pending_output = None
            duplex = info["duplex"]
            row = rows.get(request_index)
            predicting = row is not None
            if row is None:
                user_token_id = agent_token_id = tool_token_id = silence
                policy_version, tool_call = -1, None
                chunk["frame_logprobs"].append(no_logprobs)
                chunk["frame_support_ids"].append(no_support)
                if record_hiddens:
                    chunk["predictor_hiddens"].append(empty)
                audio_outputs.append(empty_audio)
                chunk["agent_audio_token_ids"].append(empty_codes)
                chunk["tool_call_json"].append(no_tool_call)
            else:
                # Return sampling probabilities alongside each prediction.
                chunk["frame_logprobs"].append(host_rows["frame_logprobs"][row])
                chunk["frame_support_ids"].append(host_rows["support_ids"][row])
                if record_hiddens:
                    chunk["predictor_hiddens"].append(
                        host_rows["predictor_hiddens"][row] if step.record_hiddens[request_index] else empty
                    )
                user_token_id, agent_token_id, tool_token_id = host_ids[row]
                policy_version = step.policy_version
                tool_call = batch.text.tool_calls[row]
                state.text_input_ids = (silence, *host_ids[row])
                audio_outputs.append(decoded.get(row, empty_audio))
                chunk["agent_audio_token_ids"].append(host_rows["audio"][row])
                if tool_call is not None:
                    state.tool_call_sequence += 1
                chunk["tool_call_json"].append(serialize_tool_call(tool_call, state.tool_call_sequence))
            # Order follows frame_output.FRAME_FIELDS.
            frames.append([
                duplex.get("epoch", 0), duplex.get("turn_id", 0),
                duplex["duplexio_prefix"], bool(duplex["duplexio_tool_token_ids"]), duplex["duplexio_tool_generation"],
                duplex.get("duplexio_given_frame") is not None,
                bool(duplex.get("final", False)), predicting,
                predicting and agent_token_id == silence and tool_token_id == silence,
                tool_call is not None, row in drawn_tool_emits, predicting and user_token_id != silence,
                user_token_id, agent_token_id, tool_token_id, policy_version, SAMPLE_RATE,
            ])

        replay: dict[str, list[Tensor]] = {}
        if any(info.get("duplexio_replay") for info in infos):
            replay = {f"replay_{name}": values for name, values in host["replay"].items()}
        return {
            "audio": audio_outputs,
            "chunk": {
                "frame": list(torch.tensor(frames, dtype=torch.long).unbind(0)),
                **chunk,
                **replay,
            },
        }

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
        if ends == [NUM_CELLS * (row + 1) for row in range(len(ends))]:
            # Single-frame appends in batch order are already contiguous rows.
            rows = hidden_states[: ends[-1]].unflatten(0, (len(ends), NUM_CELLS))
        else:
            rows = torch.stack([hidden_states[end - NUM_CELLS : end] for end in ends])
        sample_infos = [infos[index] for index in indices]
        inputs = self.sampling_inputs(sample_infos, rows.device)
        tensors = (rows, inputs.parameters, inputs.tool_bitmask)
        if self.full_cudagraph_enabled:
            graph = self.sampling_graph(tensors, inputs.top_k, inputs.support_width)
            text_ids, tool_starts, frame_logprobs, support_ids, audio = graph(tensors)
        else:
            text_ids, tool_starts, frame_logprobs, support_ids, audio = self.sample_rows(
                *tensors, top_k=inputs.top_k, support_width=inputs.support_width,
            )
        text = TextSamplingResult(
            text_ids, tool_starts, frame_logprobs, support_ids, inputs.pending, inputs.calling, [None] * len(indices),
        )
        return FrameBatch(indices=indices, text=text, audio=audio, hiddens=rows)

    def sampling_graph(self, tensors: tuple[Tensor, ...], top_k: int | None, width: int) -> FrameInputGraph:
        """The sampling graph for this batch size and top-k, captured on first use into the shared pool."""
        key = (tensors[0].shape[0], top_k, width)
        if key not in self.sampling_graphs:
            pool = next(iter(self.sampling_graphs.values())).graph.pool() if self.sampling_graphs else None
            sample = partial(self.sample_rows, top_k=top_k, support_width=width)
            self.sampling_graphs[key] = FrameInputGraph(sample, tensors, pool)
        return self.sampling_graphs[key]

    def autocast(self, value: Tensor) -> torch.autocast:
        dtype = self.vllm_config.model_config.dtype
        return torch.autocast(value.device.type, dtype=dtype, enabled=value.is_cuda and dtype != torch.float32)

    def sample_rows(
        self, rows: Tensor, parameters: Tensor, tool_bitmask: Tensor, *, top_k: int | None, support_width: int = 0,
    ) -> tuple[Tensor, ...]:
        """Tensor-only draws after the backbone, captured once per batch size.

        Returns the text ids, tool starts, frame log probabilities and content
        supports, and the agent's next audio latent.
        """
        with torch.profiler.record_function("duplexio.text_projection"):
            logits, emit_logits = self.project_text(rows)
        with torch.profiler.record_function("duplexio.text_sampling"):
            sampled = self.sample_text(
                logits, emit_logits, parameters, tool_bitmask, top_k=top_k, support_width=support_width,
            )
        with torch.profiler.record_function("duplexio.audio_sampling"), self.autocast(rows):
            noise = torch.randn(rows.shape[0], LATENT_DIM, device=rows.device, dtype=torch.float32)
            audio = self.audio_sampler.sample(rows[:, AGENT_AUDIO_CELL].float(), noise, parameters[:, FLOW_TEMPERATURE])
            return (*sampled, audio)

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

    def sampling_inputs(self, infos: list[dict[str, Any]], device: torch.device) -> SamplingInputs:
        """Resolve each row's policy and tool state on the host, with one upload each."""
        vocab_size = self.text_config.vocab_size
        parameters: list[tuple[float, ...]] = []
        constraints: list[ToolCallConstraintState | None] = []
        streams: list[TokenSamplingOptions] = []
        pending: list[int] = []
        calling: list[int] = []
        for row, info in enumerate(infos):
            state = info["duplexio_working_state"]
            # Rows inside a call, or forced to start one, sample under its grammar.
            # Idle rows draw a start decision and, speculatively, the call's first
            # token, so the host only commits what the device already chose.
            constraint = state.tool_call_constraint
            tool_state = 0
            if constraint is not None and constraint.enabled:
                if constraint.active or constraint.force_next_call:
                    if not constraint.active:
                        constraint.begin()
                    tool_state = 1
                    calling.append(row)
                elif constraint.compiled:
                    # A session whose grammar is still compiling waits to start a call.
                    tool_state = 2
                    pending.append(row)
            constraints.append(constraint if tool_state else None)
            parameters.append((*state.sampling.parameters, state.sampling.flow_temperature, tool_state))
            streams += (state.sampling.agent, state.sampling.user)
        top_k = sampled_top_k(streams, vocab_size)
        return SamplingInputs(
            parameters=_to_device(parameters, torch.float32, device),
            tool_bitmask=(
                token_bitmasks(constraints, vocab_size, device) if pending or calling
                else self.allow_all_bitmask[: len(infos)]
            ),
            top_k=top_k,
            support_width=support_width(streams[0::2], top_k),
            pending=pending,
            calling=calling,
        )

    def sample_text(
        self, logits: Tensor, emit_logits: Tensor, parameters: Tensor, tool_bitmask: Tensor, *, top_k: int | None,
        support_width: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draw every stream, masking the tool stream with each row's grammar."""
        vocab_size = logits.shape[-1]
        allowed = (tool_bitmask.unsqueeze(-1) >> self.bitmask_shifts).bitwise_and(1).flatten(1)[:, :vocab_size]
        suppressed = self.suppressed_token_mask.unsqueeze(0).expand(logits.shape[0], -1, -1)
        blocked = torch.stack((suppressed[:, 0], suppressed[:, 1] | (allowed == 0), suppressed[:, 2]), dim=1)
        return sample_streams(
            logits, emit_logits, parameters, blocked, top_k=top_k, support_width=support_width,
            silence_token_id=self.silence_token_id,
        )

    def finish_text_batch(
        self,
        text: TextSamplingResult,
        infos: list[dict[str, Any]],
        host_ids: list[list[int]],
        tool_starts: list[bool],
    ) -> list[int]:
        """Start sampled calls and advance constraints once the draws reach the host.

        Host-only: a started call's first token was drawn with the batch. Fills
        ``text.tool_calls`` and returns the rows that started a call.
        """
        started = [row for row in text.pending_tool_starts if tool_starts[row]]
        for row in started:
            infos[row]["duplexio_working_state"].tool_call_constraint.begin()
        for row in sorted((*text.tool_rows, *started)):
            constraint = infos[row]["duplexio_working_state"].tool_call_constraint
            if constraint.accept(host_ids[row][2]):
                text.tool_calls[row] = self.tool_call_compiler.take_completed_call(constraint)
        return started

    def resolve_sampling(self, sampling: Mapping[str, Any]) -> RequestSampling:
        """Validate a session's sampling once; agent text and tool calls share the agent policy."""
        config = SamplingConfig.model_validate(sampling)
        agent, user = (
            TokenSamplingOptions(policy.content.temperature, policy.content.top_k, policy.content.top_p)
            for policy in (config.agent, config.user)
        )
        emit = (config.agent.emission.temperature, config.agent.emission.temperature, config.user.emission.temperature)
        flow_temperature = config.audio.temperature
        return RequestSampling(
            agent=agent,
            user=user,
            flow_temperature=self.config.flowmap_config["sampling_temperature"] if flow_temperature is None
            else flow_temperature,
            parameters=sampling_parameters(agent, user, emit, self.text_config.vocab_size),
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
        # Forced ids may already sit on the device; a host list crosses without
        # blocking the host behind the queued frame. A scalar scatter keeps the
        # zero on the host: index assignment would copy it to the device with a
        # stream synchronization, stalling the host until the frame finished.
        if not isinstance(token_ids, Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        logits.scatter_(1, token_ids.to(hidden_states.device, non_blocking=True).view(-1, 1), 0.0)
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
        samples = torch.frombuffer(bytearray(runtime_config["duplexio_voice_prompt_pcm"]), dtype=torch.float32)
        if device.type == "cuda":
            # A pageable copy would wait for the queued backbone.
            samples = samples.pin_memory()
        return DuplexIORequestState(
            text_input_ids=(self.silence_token_id,) * len(TEXT_STREAM_NAMES),
            agent_latent=self.initial_agent_latents(1)[0],
            user_asr=FastConformerAudioStreamState(),
            input_mimi=self.audio_codec.new_state(1),
            output_mimi=self.audio_codec.new_state(1),
            voice_prompt=samples.to(device, non_blocking=True),
            system_token_ids=tuple(runtime_config["duplexio_system_token_ids"]),
            sampling=self.resolve_sampling(runtime_config["duplexio_sampling"]),
            tool_call_constraint=self.tool_call_compiler.new_state(
                runtime_config["duplexio_tools"], runtime_config["duplexio_tool_choice"],
            ),
        )

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
        """Continue each request's codec stream over its agent audio; return normalized latents."""
        with self.autocast(waveforms[0]):
            encoded, caches = self.audio_codec.encode_batch(
                [waveform[None, None] for waveform in waveforms], [state.input_mimi for state in states],
            )
        for state, cache in zip(states, caches, strict=True):
            state.input_mimi = cache
        return [self.audio_representation.normalize(value[0].T) for value in encoded]

    def decode_agent_audio_batch(self, latents: list[Tensor], states: list[DuplexIORequestState]) -> list[Tensor]:
        if not latents:
            return []
        with torch.profiler.record_function("duplexio.output_codec_decode"), self.autocast(latents[0]):
            raw = self.audio_representation.denormalize(torch.stack(latents))
            decoded, caches = self.audio_codec.decode_batch(
                [latent[None, :, None] for latent in raw], [state.output_mimi for state in states],
            )
        for state, cache in zip(states, caches, strict=True):
            state.output_mimi = cache
        return [value[0, 0] for value in decoded]

    def initial_agent_latents(self, frames: int) -> Tensor:
        """Placeholder before the first prediction and on text-only rows."""
        return self.llm.channel_emb.new_zeros(frames, LATENT_DIM)

    # The checkpoint carries the user ASR's whole RNN-T; the model reads only its encoder.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "user_asr.model.encoder.": "user_asr.encoder.",
        "user_asr.model.": None,
    })

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if any(name.startswith("user_asr.") for name in loaded):
            self.user_asr.capture_graphs()
        return loaded

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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Assemble six-cell inputs and cache addressing without changing CPU state.

    Besides the flattened embeddings this returns one entry per cell: whether the
    cell contributes a key, its 1-based persistent ordinal (0 when it writes no
    persistent key), how many persistent keys strictly earlier rows wrote, and
    the inclusive range of audio frames the cell may attend. Audio frames are
    numbered from one in audio time, which only advances on frames carrying real
    audio, so a text-only append leaves the range frozen and writes no audio key.

    Persistent keys are those no window expires: emitted text, and the pinned
    voice prompt. A prompt burst is the one frame kind that carries audio
    without being live: only its agent-audio cell contributes a key, and audio
    time stays frozen.

    Metadata is integer, one row per packed frame: persistent-cumsum offset, audio
    position, live flag, prompt flag, absolute frame. The offset subtracts
    preceding requests' keys, isolating the scan.
    """
    offsets, audio_last = metadata[:, 0], metadata[:, 1]
    audio_active, prompt_frames = metadata[:, 2] != 0, metadata[:, 3] != 0
    acoustic = (audio_active | prompt_frames)[:, None]
    user_hidden = user_hidden.masked_fill(~acoustic, 0)
    agent_hidden = agent_hidden.masked_fill(~acoustic, 0)
    text_hidden = text_hidden.masked_fill((text_ids == silence_token_id).unsqueeze(-1), 0)
    embeddings = torch.cat(
        (text_hidden + channel_embedding, user_hidden.unsqueeze(1), agent_hidden.unsqueeze(1)), dim=1,
    ).flatten(0, 1)
    text_active = (text_ids != pad_token_id) & (text_ids != silence_token_id)
    key_active = torch.cat(
        (text_active, audio_active[:, None], (prompt_frames | audio_active)[:, None]), dim=1,
    ).flatten()
    persistent = torch.cat(
        (text_active, torch.zeros_like(prompt_frames)[:, None], prompt_frames[:, None]), dim=1,
    )
    ordinals = persistent.flatten().cumsum(0, dtype=torch.int32).view_as(persistent) + offsets[:, None]
    # A row sees the keys its predecessors wrote; its own cell is its self key,
    # which the mask merges separately, and it never sees its siblings'.
    row_keys = persistent.sum(1, dtype=torch.int32)
    persistent_last = row_keys.cumsum(0) - row_keys + offsets
    audio_first = (audio_last + audio_active - audio_window_frames).clamp_min(1)

    def per_cell(values: Tensor) -> Tensor:
        """Give every cell of a row the row's value."""
        return values[:, None].expand(-1, NUM_CELLS).flatten()

    return (
        embeddings,
        key_active,
        torch.where(persistent, ordinals, 0).flatten(),
        per_cell(persistent_last),
        per_cell(audio_first),
        per_cell(audio_last),
    )


def _to_device(values: list[Any], dtype: torch.dtype, device: torch.device) -> Tensor:
    """Upload host metadata without waiting for the queued backbone."""
    host = torch.tensor(values, dtype=dtype, pin_memory=device.type == "cuda")
    return host.to(device, non_blocking=True)


class PendingHostOutputs:
    """``to_host`` outputs whose device values are still crossing to the host.

    Until ``wait`` returns, the nested lists keep their device tensors, so a
    reader that skips ``wait`` still sees correct values, only synchronously.
    """

    def __init__(
        self,
        host: dict[str, Any],
        transfers: list[tuple[torch.cuda.Event, Tensor, list[tuple[list[Tensor], int, Tensor]]]],
    ) -> None:
        self._host = host
        self._transfers = transfers

    def wait(self) -> dict[str, Any]:
        for event, packed, items in self._transfers:
            event.synchronize()
            parts = packed.split([tensor.numel() for _, _, tensor in items])
            for (values, row, tensor), part in zip(items, parts, strict=True):
                values[row] = part.view(tensor.shape)
        self._transfers = []
        return self._host


def to_host_async(outputs: Mapping[str, Any]) -> PendingHostOutputs:
    """Queue the ``to_host`` transfer without blocking the host.

    Each dtype still crosses in one copy, into pinned memory behind an event
    recorded on the current stream, so the copy lands as soon as the kernels
    that produced it finish. Host tensors pass through untouched.
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
    transfers = []
    for (device, dtype), items in groups.items():
        packed = torch.cat([tensor.detach().reshape(-1) for _, _, tensor in items])
        staged = torch.empty(packed.shape, dtype=dtype, pin_memory=True)
        staged.copy_(packed, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        transfers.append((event, staged, items))
    return PendingHostOutputs(host, transfers)


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
        raise ValueError("DuplexIO requires unquantized backbone weights")
    if vllm_config.model_config.head_dtype not in (None, torch.float32):
        raise ValueError("DuplexIO vocabulary logits require FP32 accumulation")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("DuplexIO does not support pipeline parallelism")
    if vllm_config.speculative_config is not None:
        raise ValueError("DuplexIO does not support speculative decoding")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("DuplexIO requires prefix caching to be disabled")
    scheduler = vllm_config.scheduler_config
    # All appends contain whole frames. Aligned budgets preserve that invariant
    # when the scheduler splits a prefill or mixes it with live requests.
    if (
        scheduler.max_num_batched_tokens % NUM_CELLS
        or scheduler.long_prefill_token_threshold % NUM_CELLS
    ):
        raise ValueError("DuplexIO scheduler token budgets must be multiples of six cells")
    if (
        vllm_config.parallel_config.decode_context_parallel_size != 1
        or vllm_config.parallel_config.prefill_context_parallel_size != 1
    ):
        raise ValueError("DuplexIO does not support context parallelism")
    if vllm_config.parallel_config.use_ubatching:
        raise ValueError("DuplexIO does not support microbatching")
    compilation = vllm_config.compilation_config
    if compilation.cudagraph_mode == CUDAGraphMode.FULL:
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("Batched DuplexIO CUDA graphs require FULL_DECODE_ONLY")
        if compilation.cudagraph_capture_sizes != [NUM_CELLS]:
            raise ValueError("DuplexIO full CUDA graphs require the six-cell capture size")
    elif compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
        sizes = [NUM_CELLS * count for count in range(1, vllm_config.scheduler_config.max_num_seqs + 1)]
        if compilation.cudagraph_capture_sizes != sizes:
            raise ValueError("DuplexIO decode graphs require one exact capture size per request count")


__all__ = ["DuplexIOForConditionalGeneration", "DuplexIORequestState"]
