# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DuplexIO, a full-duplex speech model that listens and speaks in the same 80 ms frames.

The backbone reads the flat sequence (``frame_layout``): each request appends whole
frames of positions, laid out by ``flat_plan`` and addressed through ``qwen_backbone.FlatStep``.
"""

from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any, cast

import torch
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
from vllm_omni.model_executor.models.duplexio.flat_plan import (
    PROMPT,
    Cursor,
    Embed,
    Frame,
    FrameKind,
    FramePlanner,
    Position,
    append_frames,
)
from vllm_omni.model_executor.models.duplexio.flowmap import FlowMap
from vllm_omni.model_executor.models.duplexio.frame_layout import (
    AGENT,
    AGENT_AUDIO_INDEX,
    AGENT_CELL_INDEX,
    FRAME_POSITIONS,
    FRAME_SIZE,
    QUERY_SLOT_OFFSET,
    SAMPLE_RATE,
    SLOT_COUNT,
    TEXT_STREAM_NAMES,
    TOOL_CALL,
    TOOL_CELL_INDEX,
    USER,
    USER_AUDIO_INDEX,
)
from vllm_omni.model_executor.models.duplexio.frame_output import frame_row
from vllm_omni.model_executor.models.duplexio.pocket_mimi import LATENT_DIM, ContinuousMimiState, PocketMimi
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    STEP_COLUMNS,
    DuplexIOQwenModel,
)
from vllm_omni.model_executor.models.duplexio.sampling_config import SamplingConfig
from vllm_omni.model_executor.models.duplexio.stream_gdn import GDN_REQUEST_COLUMNS, gdn_cache_dtypes, gdn_cache_shapes
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


@dataclass
class TextSamplingResult:
    """Sampled text and log probabilities, batched in request order.

    Each text_ids row holds the user, agent, and tool token IDs.
    """

    text_ids: Tensor
    tool_starts: Tensor  # Idle rows that started a call; the host reads only pending ones.
    # Emit and token log probabilities of the agent, tool and user streams. Each
    # emit scores the decision taken under its tempered head, forced tool emits too.
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

    parameters: Tensor  # [rows, columns], with the columns text_sampling names.
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
    hiddens: Tensor  # [rows, FRAME_POSITIONS, hidden]: each request's deciding frame.


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

    # The next live row's tokens (system, user, agent, tool), decided by the frame before it.
    text_input_ids: tuple[int, ...]
    agent_latent: Tensor  # The next live row's agent audio frame, decided with them.
    user_asr: FastConformerAudioStreamState
    input_mimi: ContinuousMimiState
    output_mimi: ContinuousMimiState
    # Raw reference audio; text-only context never advances either encoder.
    voice_prompt: Tensor
    system_token_ids: tuple[int, ...]
    sampling: RequestSampling
    tool_call_constraint: ToolCallConstraintState | None = None
    cursor: Cursor = field(default_factory=Cursor)
    frames: list[Frame] = field(default_factory=list)  # The current append's frames.
    frames_seen: int = 0
    tool_call_sequence: int = 0
    # Every live frame so far had given inputs. Only given agent audio advances
    # the input codec, so a given frame may not follow a sampled one.
    history_open: bool = True
    # The step whose sampled ids this request still waits for on the host.
    pending_output: DuplexIOStepOutput | None = None

    def fork(self) -> DuplexIORequestState:
        """Commit an append only after its model step succeeds."""
        result = copy.copy(self)
        result.cursor = self.cursor.fork()
        if self.tool_call_constraint is not None:
            result.tool_call_constraint = self.tool_call_constraint.fork()
        # Codec updates are functional; ASR batches own their mutable HF caches.
        return result


@dataclass
class PreparedAudio:
    """One scheduled slice of an append, with its audio encoded.

    An append's frames are ``flat_plan.append_frames``; the slice is whole frames.
    The opening append holds no live frame.
    """

    state: DuplexIORequestState
    frame_start: int
    frame_count: int
    prompt_start: int  # The slice's voice-prompt rows: [prompt_start, prompt_start + prompt_count).
    prompt_count: int
    live_frame: int | None  # The live frame's index in the slice, if it holds it.
    given: Mapping[str, Any] | None  # That frame's given inputs, if any.
    user_feature: Tensor | None = None  # That frame's user encoder features.
    prompt_latents: Tensor | None = None  # The slice's voice-prompt rows' agent latents.
    given_latent: Tensor | None = None  # The given frame's agent latent.


@dataclass
class PreparedFrames:
    """Views into one packed batch, consumed by the runner's request hook."""

    embeddings: Tensor
    updates: dict[str, Any]


@dataclass
class PassInputs:
    """One backbone pass: per-position addressing and embedding inputs, with the requests they belong to."""

    positions: list[Position] = field(default_factory=list)
    requests: list[tuple[int, int, int, bool]] = field(default_factory=list)  # start, end, batch row, fresh.
    embed: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    user_sources: list[Tensor] = field(default_factory=list)
    agent_sources: list[Tensor] = field(default_factory=list)


REPLAY_NAMES = ("text_ids", "user_features", "agent_audio", "audio_mask", "prompt_frames")
# A recorded input row: text ids by stream, user features or None, agent latent or None, live, prompt.
ReplayRow = tuple[tuple[int, ...], Tensor | None, Tensor | None, bool, bool]


class FrameInputGraph:
    """Capture a tensor-only function; CPU request state remains outside the graph.

    Graphs may share a memory ``pool``: replays run one at a time on one stream,
    and only the cloned outputs outlive a replay.
    """

    def __init__(
        self,
        project: Callable[..., tuple[Tensor, ...]],
        inputs: tuple[Tensor, ...],
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
    decode_query_len = FRAME_POSITIONS
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
        if self.full_cudagraph_enabled:
            # Frame embedding and the draws each compile whole, then capture.
            self.project_positions = torch.compile(self.project_positions, fullgraph=True, dynamic=True)
            self.sample_rows = torch.compile(self.sample_rows, fullgraph=True, dynamic=True)
        self.frame_input_graphs: dict[int, FrameInputGraph] = {}
        self.sampling_graphs: dict[tuple[int, int | None], FrameInputGraph] = {}
        self.pad_token_id = config.pad_token_id
        self.silence_token_id = config.silence_token_id
        hidden_size = self.text_config.hidden_size

        self.model = DuplexIOQwenModel(vllm_config=vllm_config, prefix=f"{prefix}.model" if prefix else "model")
        if self.text_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )
        # Each text stream's role embedding, added to its tokens.
        self.channel_emb = nn.Parameter(torch.zeros(len(TEXT_STREAM_NAMES), hidden_size))
        # The learned BOS audio cells and queries, by flat slot.
        self.flat_cell_emb = nn.Parameter(torch.zeros(SLOT_COUNT, hidden_size))
        # The step's flat addressing, installed before the backbone runs.
        self.step = self.model.step
        self.step_inputs: PassInputs | None = None
        adapter_hidden_size = config.audio_adapter_config.get("hidden_size") or hidden_size
        self.user_asr = FastConformerEncoder.from_checkpoint(
            config.user_asr_config,
            vllm_config.model_config.model,
            revision=vllm_config.model_config.revision,
            use_cuda_graph=self.full_cudagraph_enabled,
            max_rows=vllm_config.scheduler_config.max_num_seqs,
        )
        self.user_audio_resampler = Resample(SAMPLE_RATE, ASR_SAMPLE_RATE, dtype=torch.float32).to(
            device=self.channel_emb.device,
        )
        # FP32 like Pocket TTS and Mimi; encoding follows training's bf16 autocast.
        self.audio_codec = PocketMimi().float()
        self.audio_representation = ContinuousAudioRepresentation(LATENT_DIM)
        self.audio_sampler = FlowMap(
            LATENT_DIM,
            config.flowmap_config["mlp_dim"],
            hidden_size,
            config.flowmap_config["mlp_depth"],
            inference_steps=config.flowmap_config["inference_steps"],
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
        # Decisions read single positions: the user's at U, the agent's and tool's at A.
        self.user_token_projection = nn.Linear(hidden_size, hidden_size)
        self.user_emit_head = nn.Linear(hidden_size, 1)
        self.agent_emit_head = nn.Linear(hidden_size, 1)
        self.tool_call_emit_head = nn.Linear(hidden_size, 1)
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
        self._forced_next_token_ids: list[int] | Tensor | None = None
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        agent_suppressed, tool_suppressed = text_suppression_ids(self.tokenizer, self.silence_token_id)
        self.register_buffer(
            "agent_suppressed_token_ids",
            torch.tensor(agent_suppressed, dtype=torch.long, device=self.channel_emb.device),
            persistent=False,
        )
        self.register_buffer(
            "tool_suppressed_token_ids",
            torch.tensor(tool_suppressed, dtype=torch.long, device=self.channel_emb.device),
            persistent=False,
        )
        # The user stream suppresses only the silence token.
        self.register_buffer(
            "user_suppressed_token_ids",
            torch.tensor([self.silence_token_id], dtype=torch.long, device=self.channel_emb.device),
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
            "bitmask_shifts",
            torch.arange(32, dtype=torch.int32, device=device),
            persistent=False,
        )
        # Batches without a tool grammar replay with this mask.
        self.register_buffer(
            "allow_all_bitmask",
            torch.full(xgr.get_bitmask_shape(max_rows, vocab_size), -1, dtype=torch.int32, device=device),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.model.embed_input_ids(input_ids)

    def update_graph_inputs(
        self,
        request_infos: list[dict[str, Any]],
    ) -> None:
        """Install the step's positions at stable device addresses; without requests, an inert step."""
        if not request_infos or self.step_inputs is None:
            self.step.reset()
            return
        self.install_pass(self.step_inputs)

    def install_pass(self, inputs: PassInputs) -> None:
        """Upload one pass's addressing (``qwen_backbone.STEP_COLUMNS``) into the shared step."""
        positions = [
            (
                position.rope,
                position.write,
                position.region,
                row,
                *position.ranges,
                position.own_key,
                position.gdn_stream,
                position.gdn_real,
            )
            for start, end, row, _ in inputs.requests
            for position in inputs.positions[start:end]
        ]
        device = self.channel_emb.device
        packed = _to_device(positions, torch.int32, device).view(-1, STEP_COLUMNS)
        requests = _to_device(inputs.requests, torch.int32, device).view(-1, GDN_REQUEST_COLUMNS)
        self.step.install(packed, requests)

    def prepare_audio_request(self, tokens: int, info: dict[str, Any], device: torch.device) -> PreparedAudio:
        """Validate framing and fork state at the model's input boundary."""
        duplex = info["duplex"]
        append_frame_count = duplex["frame_count"]
        token_offset = info["duplex_token_offset"]
        prompt_len = info["duplex_prompt_len"]
        if tokens == 0 or tokens % FRAME_POSITIONS or token_offset % FRAME_POSITIONS or prompt_len % FRAME_POSITIONS:
            raise ValueError(
                f"DuplexIO requires complete {FRAME_POSITIONS}-position frames; "
                f"got frame_count={append_frame_count}, tokens={tokens}, offset={token_offset}, end={prompt_len}"
            )
        frame_count = tokens // FRAME_POSITIONS
        frame_start = append_frame_count - (prompt_len - token_offset) // FRAME_POSITIONS
        frame_end = frame_start + frame_count
        assert 0 <= frame_start < frame_end <= append_frame_count
        prefix = duplex["duplexio_prefix"]
        state = info.get("duplexio_model_state")  # Absent until the request's first append.
        if state is None:
            if not prefix:
                raise ValueError("DuplexIO requires the voice and system prefix before live input")
            state = self._new_request_state(duplex["runtime_config"], device)
        else:
            if state.pending_output is not None:
                # Scheduled again before its last step was finalized: its
                # sampled ids are this frame's inputs, so wait for them here.
                state.pending_output.finalize(self)
            state = state.fork()
        prompt_rows = state.voice_prompt.numel() // FRAME_SIZE if prefix else 0
        if frame_start == 0:
            if prefix and state.frames_seen:
                raise ValueError("DuplexIO prefix must lead the request exactly once")
            state.frames = append_frames(
                prompt_rows,
                state.system_token_ids,
                tuple(duplex["duplexio_tool_token_ids"]),
                opening=prefix,
            )
            if len(state.frames) != append_frame_count:
                raise ValueError("DuplexIO append frames do not match its prefix, live frame and tool result")
        state.frames_seen += frame_count
        # The opening append decides live row 0 and holds no live frame.
        live = next((index for index, frame in enumerate(state.frames) if frame.kind == FrameKind.LIVE), None)
        live_frame = live - frame_start if live is not None and frame_start <= live < frame_end else None
        given = duplex["duplexio_given_frame"]
        if live_frame is not None:
            if given is None:
                state.history_open = False
            else:
                self.validate_given_frame(given, state)
        prompt_start = min(frame_start * FRAME_POSITIONS, prompt_rows)
        return PreparedAudio(
            state,
            frame_start,
            frame_count,
            prompt_start,
            min(frame_end * FRAME_POSITIONS, prompt_rows) - prompt_start,
            live_frame,
            given if live_frame is not None else None,
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
        self,
        requests: list[tuple[int, dict[str, Any]]],
        device: torch.device,
    ) -> list[PreparedAudio]:
        prepared = [self.prepare_audio_request(tokens, info, device) for tokens, info in requests]
        live = {
            index: info["duplex"]["pcm"]
            for index, (_, info) in enumerate(requests)
            if prepared[index].live_frame is not None
        }
        # Agent audio of the given live frames, heard instead of the last prediction.
        given = {index: item.given["agent_pcm"] for index, item in enumerate(prepared) if item.given is not None}
        # The user encoder reads only this step's audio and its own caches, so on
        # CUDA it runs on a side stream, overlapping the previous step's backbone.
        cuda = device.type == "cuda"
        if cuda and any(item.prompt_count for item in prepared):
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
                live = dict(zip(live, uploaded[: len(live)], strict=True))
                given = dict(zip(given, uploaded[len(live) :], strict=True))
            acoustic = []
            user_waveforms = []
            prompt_waveforms: dict[int, Tensor] = {}
            for index, item in enumerate(prepared):
                # The user encoder hears, in stream order, silence for the
                # slice's voice-prompt rows (as training does), then the client.
                waveforms = []
                if item.prompt_count:
                    start = item.prompt_start * FRAME_SIZE
                    prompt_waveform = item.state.voice_prompt[start : start + item.prompt_count * FRAME_SIZE]
                    prompt_waveforms[index] = prompt_waveform
                    waveforms.append(torch.zeros_like(prompt_waveform))
                if index in live:
                    waveforms.append(live[index])
                if waveforms:
                    acoustic.append(index)
                    user_waveforms.append(waveforms[0] if len(waveforms) == 1 else torch.cat(waveforms))
            user_features = (
                self.encode_user_audio_batch(
                    user_waveforms,
                    [prepared[index].state for index in acoustic],
                )
                if acoustic
                else []
            )
        if cuda and acoustic:
            main = torch.cuda.current_stream(device)
            main.wait_stream(self.user_audio_stream)
            for features in (*user_features, *given.values()):
                features.record_stream(main)
        for index, features in zip(acoustic, user_features, strict=True):
            item = prepared[index]
            if item.live_frame is not None:
                # The live frame's row follows the prompt rows; a shortfall is silence.
                item.user_feature = (
                    features[item.prompt_count]
                    if features.shape[0] > item.prompt_count
                    else features.new_zeros(self.user_asr.output_dim)
                )
        # The agent codec hears, in stream order, the slice's voice-prompt rows and
        # then a given live frame.
        heard = sorted({*prompt_waveforms, *given})
        if heard:
            waveforms = [
                torch.cat(
                    [waveform for waveform in (prompt_waveforms.get(index), given.get(index)) if waveform is not None]
                )
                for index in heard
            ]
            codes = self.encode_agent_audio_batch(waveforms, [prepared[index].state for index in heard])
            for index, audio in zip(heard, codes, strict=True):
                item = prepared[index]
                item.prompt_latents = audio[: item.prompt_count]
                if index in given:
                    item.given_latent = audio[item.prompt_count]
        return prepared

    @cached_property
    def user_audio_stream(self) -> torch.cuda.Stream:
        return torch.cuda.Stream()

    @torch.inference_mode()
    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> dict[str, dict[str, PreparedFrames]]:
        requests = [model_intermediate_buffer[req_id] for req_id in req_ids]
        prepared = self.prepare_frames(
            [(info["_omni_num_scheduled_tokens"], info) for info in requests],
            device,
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
            (prepared_frames,) = self.prepare_frames([(input_ids.numel(), info)], input_ids.device)
        return input_ids, prepared_frames.embeddings, prepared_frames.updates

    def live_tokens(self, state: DuplexIORequestState, given: Mapping[str, Any] | None) -> tuple[int, int, int]:
        """A live row's user, agent and tool tokens: given, or decided by the frame before it."""
        if given is not None:
            return given["user_token_id"], given["agent_token_id"], given["tool_call_token_id"]
        return state.text_input_ids[1:]

    def plan_frames(
        self,
        inputs: PassInputs,
        row: int,
        item: PreparedAudio,
        frames: list[Frame],
        replay: list[ReplayRow] | None,
    ) -> None:
        """Lay one request's frames out into ``inputs``, advancing its cursor."""
        state = item.state
        cursor = state.cursor
        planner = FramePlanner(
            cursor,
            audio_window_frames=self.config.audio_attention_window_frames,
            max_persistent_keys=self.vllm_config.model_config.max_model_len,
        )
        start = len(inputs.positions)
        silence = self.silence_token_id
        for frame in frames:
            if frame.kind == FrameKind.LIVE:
                tokens = self.live_tokens(state, item.given)
                live = {
                    stream: token
                    for stream, token in zip((USER, AGENT, TOOL_CALL), tokens, strict=True)
                    if token not in (silence, self.pad_token_id)
                }
                positions = planner.plan(frame, live)
                latent = state.agent_latent if item.given_latent is None else item.given_latent
                feature = item.user_feature
                if feature is None:
                    feature = self.channel_emb.new_zeros(self.user_asr.output_dim)
                if replay is not None:
                    replay.append(((silence, *tokens), feature, latent, True, False))
            else:
                positions = planner.plan(frame)
                feature = latent = None
            for position in positions:
                user_index = agent_index = 0
                if position.embed == Embed.USER_AUDIO:
                    inputs.user_sources.append(feature)
                    user_index = len(inputs.user_sources)
                elif position.embed == Embed.AGENT_AUDIO:
                    if frame.kind == FrameKind.LIVE:
                        inputs.agent_sources.append(latent)
                    else:
                        assert item.prompt_latents is not None
                        inputs.agent_sources.append(item.prompt_latents[position.value - item.prompt_start])
                    agent_index = len(inputs.agent_sources)
                inputs.embed.append((position.embed, position.value, position.stream, user_index, agent_index))
            if replay is not None and frame.kind == FrameKind.TEXT:
                quiet = (silence,) * (len(TEXT_STREAM_NAMES) - 1)
                for stream, value in frame.entries:
                    if stream == PROMPT:
                        latents = item.prompt_latents
                        assert latents is not None
                        replay.append(((silence, *quiet), None, latents[value - item.prompt_start], False, True))
                    else:
                        replay.append(((value, *quiet), None, None, False, False))
            inputs.positions += positions
        inputs.requests.append((start, len(inputs.positions), row, cursor.gdn_fresh))
        cursor.gdn_fresh = False

    def embed_pass(self, inputs: PassInputs, graph: bool) -> Tensor:
        """The input embeddings of one pass's positions."""
        device = self.channel_emb.device
        packed = _to_device(inputs.embed, torch.long, device).view(-1, 5)
        # Row zero of each source is silence; the adapters read FP32 rows.
        zero = torch.zeros(1, device=device)
        user_sources = torch.stack([zero.expand(self.user_asr.output_dim), *(x.float() for x in inputs.user_sources)])
        agent_sources = torch.stack([zero.expand(LATENT_DIM), *(x.float() for x in inputs.agent_sources)])
        projected = (packed, user_sources, agent_sources)
        with torch.profiler.record_function("duplexio.frame_inputs"):
            if graph and self.full_cudagraph_enabled:
                (embeddings,) = self.frame_input_graph(projected)(projected)
            else:
                (embeddings,) = self.project_positions(*projected)
        return embeddings

    def prepare_frames(
        self,
        requests: list[tuple[int, dict[str, Any]]],
        device: torch.device,
    ) -> list[PreparedFrames]:
        """Plan the step's positions once; request boundaries own all running counters."""
        audio = self.prepare_audio_requests(requests, device)
        inputs = PassInputs()
        spans: list[tuple[int, int]] = []
        replays: list[dict[str, Tensor]] = []
        offset = 0
        for index, (item, (tokens, info)) in enumerate(zip(audio, requests, strict=True)):
            frames = item.state.frames[item.frame_start : item.frame_start + item.frame_count]
            replay: list[ReplayRow] | None = [] if info["duplex"]["runtime_config"]["duplexio_record_inputs"] else None
            self.plan_frames(inputs, index, item, frames, replay)
            replays.append({} if replay is None else self.replay_tensors(replay))
            spans.append((offset, offset + tokens))
            offset += tokens
        # Single-frame live appends replay one captured embedding graph per batch size.
        uniform = all(item.frame_count == 1 and item.live_frame is not None and item.given is None for item in audio)
        embeddings = self.embed_pass(inputs, uniform)
        self.step_inputs = inputs
        return [
            PreparedFrames(
                embeddings[start:end],
                {"duplexio_working_state": item.state, "duplexio_replay": replay},
            )
            for item, (start, end), replay in zip(audio, spans, replays, strict=True)
        ]

    def frame_input_graph(self, inputs: tuple[Tensor, ...]) -> FrameInputGraph:
        """The one-frame graph for this batch size, captured on first use into the shared pool."""
        size = inputs[0].shape[0]
        if size not in self.frame_input_graphs:
            pool = next(iter(self.frame_input_graphs.values())).graph.pool() if self.frame_input_graphs else None
            self.frame_input_graphs[size] = FrameInputGraph(self.project_positions, inputs, pool)
        return self.frame_input_graphs[size]

    @torch.inference_mode()
    def capture_frame_graphs(self) -> None:
        """Compile and capture frame embedding for every batch size before serving starts.

        Otherwise the first batch of each size would stall all live streams while
        it compiles. The runner calls this after its own capture, under the
        default dtype that serving runs with (weight loading changes it, which
        compiled code guards on). Sampling graphs also key on the batch's top-k,
        which only requests know; they are captured here for the default
        sampling policy, which unconfigured sessions use.
        """
        if not self.full_cudagraph_enabled or self.frame_input_graphs:
            return
        device = self.channel_emb.device
        max_rows = self.vllm_config.scheduler_config.max_num_seqs
        for size in range(1, max_rows + 1):
            # Live frames without tokens: a no-op, two queries, then the frame's audio.
            frame = [
                (Embed.NOOP, 0, 0, 0, 0),
                (Embed.CELL, QUERY_SLOT_OFFSET + AGENT, 0, 0, 0),
                (Embed.CELL, QUERY_SLOT_OFFSET + TOOL_CALL, 0, 0, 0),
            ]
            packed = torch.tensor(
                [
                    entry
                    for request in range(size)
                    for entry in (
                        *frame,
                        (Embed.USER_AUDIO, 0, 0, request + 1, 0),
                        (Embed.AGENT_AUDIO, 0, 0, 0, request + 1),
                    )
                ],
                dtype=torch.long,
                device=device,
            )
            user_sources = torch.zeros(size + 1, self.user_asr.output_dim, device=device)
            agent_sources = torch.zeros(size + 1, LATENT_DIM, device=device)
            self.frame_input_graph((packed, user_sources, agent_sources))
        sampling = self.resolve_sampling(SamplingConfig().model_dump())
        top_k = sampled_top_k((sampling.agent, sampling.user), self.text_config.vocab_size)
        width = support_width([sampling.agent], top_k)
        parameters = torch.tensor(
            [(*sampling.parameters, sampling.flow_temperature, 0)] * max_rows,
            dtype=torch.float32,
            device=device,
        )
        rows = self.channel_emb.new_zeros(max_rows, FRAME_POSITIONS, self.text_config.hidden_size)
        for size in range(1, max_rows + 1):
            tensors = (rows[:size], parameters[:size], self.allow_all_bitmask[:size])
            self.sampling_graph(tensors, top_k, width)

    def project_positions(self, packed: Tensor, user_sources: Tensor, agent_sources: Tensor) -> tuple[Tensor]:
        """Tensor-only input embeddings of flat positions, shared by all requests.

        ``packed`` is ``(positions, 5)``: the ``flat_plan.Embed`` kind, the token
        id or cell slot, the token's stream, and the position's row in
        ``user_sources`` and ``agent_sources``, whose row zero is silence.
        """
        kind, value, stream, user_index, agent_index = packed.unbind(1)
        token, cell = kind == int(Embed.TOKEN), kind == int(Embed.CELL)
        with self.autocast(packed):
            user = self.user_audio_input_adapter(user_sources)
            agent = self.agent_audio_input_adapter(agent_sources)
        text = self.model.embed_input_ids(torch.where(token, value, 0)) + self.channel_emb[stream]
        cells = self.flat_cell_emb[torch.where(cell, value, 0)]
        embeddings = torch.where(token[:, None], text, torch.zeros_like(text))
        embeddings = torch.where(cell[:, None], cells.to(text.dtype), embeddings)
        embeddings = torch.where((kind == int(Embed.USER_AUDIO))[:, None], user[user_index].to(text.dtype), embeddings)
        embeddings = torch.where(
            (kind == int(Embed.AGENT_AUDIO))[:, None], agent[agent_index].to(text.dtype), embeddings
        )
        return (embeddings,)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Run the backbone over the embeddings ``preprocess`` built; the step holds the addressing."""
        del input_ids, positions, intermediate_tensors, kwargs
        assert inputs_embeds is not None
        with torch.profiler.record_function("duplexio.backbone"):
            return self.model(positions=self.step.rope[: inputs_embeds.shape[0]], inputs_embeds=inputs_embeds)

    def make_omni_output(self, model_output: Tensor, **kwargs: Any) -> OmniOutput:
        """Sample the three text streams and audio outside the model graph."""
        hidden_states = model_output
        infos: list[dict[str, Any]] = kwargs["model_intermediate_buffer"]
        if not infos:  # Warm-up and profiling runs carry no requests.
            self._forced_next_token_ids = None
            return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs={})
        batch = self.sample_frames(
            hidden_states,
            kwargs["request_token_spans"],
            infos,
            kwargs["request_sample_eligible"],
        )
        states: list[DuplexIORequestState] = [info["duplexio_working_state"] for info in infos]
        rows = {} if batch is None else {index: row for row, index in enumerate(batch.indices)}
        decode_rows = [row for index, row in rows.items() if infos[index]["duplex"]["decode_audio"]]
        predicted_audio = [] if batch is None else batch.audio.unbind(0)
        waveforms = self.decode_agent_audio_batch(
            [predicted_audio[row] for row in decode_rows],
            [states[batch.indices[row]] for row in decode_rows],
        )
        record_hiddens = [info["duplex"]["runtime_config"]["duplexio_record_hiddens"] for info in infos]
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
        empty = torch.empty(0, dtype=hidden_states.dtype)
        replays = [info["duplexio_replay"] for info in infos]
        replay = {name: [value.get(name, empty) for value in replays] for name in REPLAY_NAMES}
        step = DuplexIOStepOutput(
            batch=batch,
            infos=infos,
            states=states,
            rows=rows,
            decode_rows=decode_rows,
            record_hiddens=record_hiddens,
            policy_version=self.policy_version,
            empty=empty,
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
            # One retained scheduler slot per persistent key, in whole frames; audio
            # KV has fixed reserved rings. Keep one frame to preserve the
            # recurrent-state marker.
            streaming_retained_tokens=[
                max(1, -(-state.cursor.persistent_keys // FRAME_POSITIONS)) * FRAME_POSITIONS for state in states
            ],
            # An append's positions take at most one row each.
            streaming_position_budget=[self.text_config.max_position_embeddings - state.cursor.row for state in states],
        )

    def replay_tensors(self, rows: list[ReplayRow]) -> dict[str, Tensor]:
        """One request's recorded rows as tensors (``REPLAY_NAMES``)."""
        device = self.channel_emb.device
        user = [features for _, features, _, _, _ in rows]
        agent = [latent for _, _, latent, _, _ in rows]
        zero_user = self.channel_emb.new_zeros(self.user_asr.output_dim, dtype=torch.float32)
        zero_agent = self.channel_emb.new_zeros(LATENT_DIM, dtype=torch.float32)
        return {
            "text_ids": _to_device([ids for ids, _, _, _, _ in rows], torch.long, device).view(
                -1, len(TEXT_STREAM_NAMES)
            ),
            # A slice of only a BOS frame records no row.
            "user_features": torch.stack([zero_user if value is None else value.float() for value in user])
            if rows
            else zero_user.new_zeros(0, zero_user.numel()),
            "agent_audio": torch.stack([zero_agent if value is None else value.float() for value in agent])
            if rows
            else zero_agent.new_zeros(0, LATENT_DIM),
            "audio_mask": _to_device([live for *_, live, _ in rows], torch.bool, device),
            "prompt_frames": _to_device([prompt for *_, prompt in rows], torch.bool, device),
        }

    def finalize_multimodal_outputs_from_cpu_snapshot(self, outputs: Any) -> Any:
        """The host half of ``make_omni_output``: wait for the draws, then feed them back."""
        return outputs.finalize(self) if isinstance(outputs, DuplexIOStepOutput) else outputs

    def _finish_step(self, step: DuplexIOStepOutput) -> dict[str, Any]:
        host = step.host.wait()
        batch, infos, states, rows = step.batch, step.infos, step.states, step.rows
        host_rows: dict[str, list[Tensor]] = {}
        host_ids: list[list[int]] = []
        decoded: dict[int, Tensor] = {}
        if batch is not None:
            sampled = host["sampled"]
            host_ids = sampled["text_ids"][0].tolist()
            self.finish_text_batch(
                batch.text,
                [states[index] for index in batch.indices],
                host_ids,
                sampled["tool_starts"][0].tolist(),
            )
            host_rows = {
                name: list(values[0].unbind(0))
                for name, values in sampled.items()
                if name not in ("text_ids", "tool_starts", "waveforms")
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
            name: []
            for name in (
                "frame_logprobs",
                "frame_support_ids",
                "agent_audio_token_ids",
                "tool_call_json",
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
            frames.append(
                frame_row(
                    duplex_epoch=duplex["epoch"],
                    duplex_turn_id=duplex["turn_id"],
                    prefix=duplex["duplexio_prefix"],
                    tool_result=bool(duplex["duplexio_tool_token_ids"]),
                    tool_generation=duplex["duplexio_tool_generation"],
                    given=duplex["duplexio_given_frame"] is not None,
                    end_of_turn=duplex["final"],
                    predicted=predicting,
                    model_listen=predicting and agent_token_id == silence and tool_token_id == silence,
                    tool_call_complete=tool_call is not None,
                    tool_emit_sampled=row in drawn_tool_emits,
                    user_emit=predicting and user_token_id != silence,
                    user_token_id=user_token_id,
                    agent_token_id=agent_token_id,
                    tool_call_token_id=tool_token_id,
                    policy_version=policy_version,
                    sample_rate_hz=SAMPLE_RATE,
                )
            )

        replay: dict[str, list[Tensor]] = {}
        if any(info["duplexio_replay"] for info in infos):
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
        # Each request's last frame decides its next row.
        ends = [request_token_spans[index][1] for index in indices]
        if ends == [FRAME_POSITIONS * (row + 1) for row in range(len(ends))]:
            # Single-frame appends in batch order are already contiguous rows.
            rows = hidden_states[: ends[-1]].unflatten(0, (len(ends), FRAME_POSITIONS))
        else:
            rows = torch.stack([hidden_states[end - FRAME_POSITIONS : end] for end in ends])
        inputs = self.sampling_inputs([infos[index]["duplexio_working_state"] for index in indices], rows.device)
        tensors = (rows, inputs.parameters, inputs.tool_bitmask)
        if self.full_cudagraph_enabled:
            graph = self.sampling_graph(tensors, inputs.top_k, inputs.support_width)
            text_ids, tool_starts, frame_logprobs, support_ids, audio = graph(tensors)
        else:
            text_ids, tool_starts, frame_logprobs, support_ids, audio = self.sample_rows(
                *tensors,
                top_k=inputs.top_k,
                support_width=inputs.support_width,
            )
        text = TextSamplingResult(
            text_ids,
            tool_starts,
            frame_logprobs,
            support_ids,
            inputs.pending,
            inputs.calling,
            [None] * len(indices),
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
        self,
        rows: Tensor,
        parameters: Tensor,
        tool_bitmask: Tensor,
        *,
        top_k: int | None,
        support_width: int = 0,
    ) -> tuple[Tensor, ...]:
        """Tensor-only draws after the backbone, captured once per batch size.

        Returns the text ids, tool starts, frame log probabilities and content
        supports, and the agent's next audio latent.
        """
        with torch.profiler.record_function("duplexio.text_projection"):
            logits, emit_logits = self.project_text(rows)
        with torch.profiler.record_function("duplexio.text_sampling"):
            sampled = self.sample_text(
                logits,
                emit_logits,
                parameters,
                tool_bitmask,
                top_k=top_k,
                support_width=support_width,
            )
        with torch.profiler.record_function("duplexio.audio_sampling"), self.autocast(rows):
            noise = torch.randn(rows.shape[0], LATENT_DIM, device=rows.device, dtype=torch.float32)
            audio = self.audio_sampler.sample(
                rows[:, AGENT_AUDIO_INDEX].float(), noise, parameters[:, FLOW_TEMPERATURE]
            )
            return (*sampled, audio)

    def project_text(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Project the entire request batch before any CPU-side tool decisions.

        ``rows`` are deciding frames: the agent and tool cells feed the LM head;
        U carries the user's word and emit, A the agent's and tool's emits.
        """
        user, agent = rows[:, USER_AUDIO_INDEX], rows[:, AGENT_AUDIO_INDEX]
        projected = torch.cat((rows[:, AGENT_CELL_INDEX], rows[:, TOOL_CELL_INDEX], self.user_token_projection(user)))
        logits = self.logits_processor(self.lm_head, projected)
        logits = logits.view(3, rows.shape[0], -1).transpose(0, 1)
        emit_logits = torch.cat(
            (
                self.agent_emit_head(agent),
                self.tool_call_emit_head(agent),
                self.user_emit_head(user),
            ),
            dim=-1,
        )
        return logits, emit_logits

    def sampling_inputs(self, states: list[DuplexIORequestState], device: torch.device) -> SamplingInputs:
        """Resolve each row's policy and tool state on the host, with one upload each."""
        vocab_size = self.text_config.vocab_size
        parameters: list[tuple[float, ...]] = []
        constraints: list[ToolCallConstraintState | None] = []
        streams: list[TokenSamplingOptions] = []
        pending: list[int] = []
        calling: list[int] = []
        for row, state in enumerate(states):
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
                token_bitmasks(constraints, vocab_size, device)
                if pending or calling
                else self.allow_all_bitmask[: len(states)]
            ),
            top_k=top_k,
            support_width=support_width(streams[0::2], top_k),
            pending=pending,
            calling=calling,
        )

    def sample_text(
        self,
        logits: Tensor,
        emit_logits: Tensor,
        parameters: Tensor,
        tool_bitmask: Tensor,
        *,
        top_k: int | None,
        support_width: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draw every stream, masking the tool stream with each row's grammar."""
        vocab_size = logits.shape[-1]
        allowed = (tool_bitmask.unsqueeze(-1) >> self.bitmask_shifts).bitwise_and(1).flatten(1)[:, :vocab_size]
        suppressed = self.suppressed_token_mask.unsqueeze(0).expand(logits.shape[0], -1, -1)
        blocked = torch.stack((suppressed[:, 0], suppressed[:, 1] | (allowed == 0), suppressed[:, 2]), dim=1)
        return sample_streams(
            logits,
            emit_logits,
            parameters,
            blocked,
            top_k=top_k,
            support_width=support_width,
            silence_token_id=self.silence_token_id,
        )

    def finish_text_batch(
        self,
        text: TextSamplingResult,
        states: list[DuplexIORequestState],
        host_ids: list[list[int]],
        tool_starts: list[bool],
    ) -> None:
        """Start sampled calls and advance constraints once the draws reach the host.

        Host-only: a started call's first token was drawn with the batch. Fills
        ``text.tool_calls``.
        """
        started = [row for row in text.pending_tool_starts if tool_starts[row]]
        for row in started:
            states[row].tool_call_constraint.begin()
        for row in sorted((*text.tool_rows, *started)):
            constraint = states[row].tool_call_constraint
            if constraint.accept(host_ids[row][2]):
                text.tool_calls[row] = self.tool_call_compiler.take_completed_call(constraint)

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
            flow_temperature=self.config.flowmap_config["sampling_temperature"]
            if flow_temperature is None
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
                runtime_config["duplexio_tools"],
                runtime_config["duplexio_tool_choice"],
            ),
        )

    def encode_user_audio_batch(self, waveforms: list[Tensor], states: list[DuplexIORequestState]) -> list[Tensor]:
        with torch.profiler.record_function("duplexio.asr_resample"):
            resampled, tails = streaming_resample_batch(
                waveforms,
                [state.user_asr.resample_tail for state in states],
                self.user_audio_resampler,
            )
        encoded, caches = self.user_asr.encode_audio_batch(
            resampled,
            [state.user_asr for state in states],
        )
        for state, cache, tail in zip(states, caches, tails, strict=True):
            state.user_asr = cache
            cache.resample_tail = tail
        return [value[0] for value in encoded]

    def encode_agent_audio_batch(self, waveforms: list[Tensor], states: list[DuplexIORequestState]) -> list[Tensor]:
        """Continue each request's codec stream over its agent audio; return normalized latents."""
        with self.autocast(waveforms[0]):
            encoded, caches = self.audio_codec.encode_batch(
                [waveform[None, None] for waveform in waveforms],
                [state.input_mimi for state in states],
            )
        for state, cache in zip(states, caches, strict=True):
            state.input_mimi = cache
        return [self.audio_representation.normalize(value[0].T) for value in encoded]

    def decode_agent_audio_batch(self, latents: list[Tensor], states: list[DuplexIORequestState]) -> list[Tensor]:
        if not latents:
            return []
        with torch.profiler.record_function("duplexio.output_codec_decode"):
            raw = self.audio_representation.denormalize(torch.stack(latents))
            decoded, caches = self.audio_codec.decode_batch(
                [latent[None, :, None] for latent in raw],
                [state.output_mimi for state in states],
            )
        for state, cache in zip(states, caches, strict=True):
            state.output_mimi = cache
        return [value[0, 0] for value in decoded]

    def initial_agent_latents(self, frames: int) -> Tensor:
        """Placeholder before the first prediction and on text-only rows."""
        return self.channel_emb.new_zeros(frames, LATENT_DIM)

    # The checkpoint carries the user ASR's whole RNN-T; the model reads only its encoder.
    # Applied in order: checkpoints nest the backbone under ``llm.base_model``
    # and carry the ASR model's decoder, which serving does not run.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm.base_model.model.": "model.",
            "llm.base_model.lm_head.": "lm_head.",
            "llm.": "",
            "user_asr.model.encoder.": "user_asr.encoder.",
            "user_asr.model.": None,
            # Exported for runtimes without the processor; the encoder derives its own.
            "user_asr.mel_filters": None,
            "user_asr.stft_window": None,
            "audio_sampler.flow.": "audio_sampler.",
        }
    )

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if any(name.startswith("user_asr.") for name in loaded):
            self.user_asr.capture_graphs()
        return loaded

    def set_policy_version(self, version: int) -> None:
        """Tag every later frame with the weights just loaded; versions only increase."""
        if version <= self.policy_version:
            raise ValueError(f"Policy version {version} must exceed {self.policy_version}")
        self.policy_version = version

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
    """The ids the agent and tool heads never sample: chat markup, and for the agent tool-call markup."""

    def token_id(text: str) -> int:
        (encoded,) = tokenizer.encode(text, add_special_tokens=False)
        return encoded

    tool_ids = {*tokenizer.all_special_ids, silence_token_id}
    tool_ids |= {token_id(text) for text in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")}
    agent_ids = tool_ids | {token_id(text) for text in ("<tool_call>", "</tool_call>")}
    return sorted(agent_ids), sorted(tool_ids)


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
    if scheduler.max_num_batched_tokens % FRAME_POSITIONS or scheduler.long_prefill_token_threshold % FRAME_POSITIONS:
        raise ValueError(f"DuplexIO scheduler token budgets must be whole {FRAME_POSITIONS}-position frames")
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
        if compilation.cudagraph_capture_sizes != [FRAME_POSITIONS]:
            raise ValueError(f"DuplexIO full CUDA graphs require the {FRAME_POSITIONS}-position capture size")
    elif compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
        sizes = [FRAME_POSITIONS * count for count in range(1, vllm_config.scheduler_config.max_num_seqs + 1)]
        if compilation.cudagraph_capture_sizes != sizes:
            raise ValueError("DuplexIO decode graphs require one exact capture size per request count")


__all__ = ["DuplexIOForConditionalGeneration", "DuplexIORequestState"]
