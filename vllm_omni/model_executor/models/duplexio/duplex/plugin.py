# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DuplexIO full-duplex model plugin: engine policy and session policy in one class.

Every append is one 80 ms client (or silence) frame. The first append of an
epoch also carries the pinned voice prompt and system tokens before that
frame, and a pending tool result follows it as text rows. The worker samples
once, at the append's last row.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputDecision
from vllm_omni.engine.duplex.intermediate import build_duplex_append_prompt
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    EncodeAudio,
    reject_changed_runtime_value,
)
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.common.duplex.pcm_buffer import FixedFramePcmAppendBuffer
from vllm_omni.model_executor.models.duplexio.duplex.data_plane import DuplexIODataPlane
from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_SIZE, NUM_CELLS, SAMPLE_RATE
from vllm_omni.model_executor.models.duplexio.frame_output import frame_fields
from vllm_omni.model_executor.models.duplexio.sampling_config import SamplingConfig
from vllm_omni.model_executor.models.duplexio.tool_calling import tool_call_grammar

if TYPE_CHECKING:
    from vllm.config import ModelConfig

CHUNK_PERIOD_MS = 80
#: Tool results the session may queue before the model has taken the oldest.
MAX_PENDING_TOOL_RESULTS = 8

# The chat turn the system prompt opens, by which role speaks first.
INITIAL_PREFIXES = {"agent": "<|im_start|>assistant\n", "user": "<|im_start|>user\n"}

PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "duplexio_scheduler_token_id",
        "duplexio_start_role",
        "duplexio_voice_prompt_pcm",
        "duplexio_voice_prompt_frames",
        "duplexio_system_token_ids",
        "duplexio_tools",
        "duplexio_tool_choice",
        "duplexio_tool_generation",
        "duplexio_tool_results",
        "duplexio_record_inputs",
        "duplexio_record_hiddens",
    }
)


def stage_sampling_params(defaults: tuple[object, ...]) -> tuple[object, ...]:
    """One forced token per append, with DELTA outputs so each append emits only its own."""
    configured = []
    for params in defaults:
        if isinstance(params, SamplingParams):
            params = params.clone()
            params.max_tokens = 1
            params.output_kind = RequestOutputKind.DELTA
        configured.append(params)
    return tuple(configured)


class GivenFrame(TypedDict):
    """One frame of conversation history, in place of the model's prediction for it."""

    user_token_id: int
    agent_token_id: int
    tool_call_token_id: int
    agent_pcm: bytes  # One frame of float32 agent audio at the model's sample rate.


def append_fields(
    runtime_config: Mapping[str, Any],
    pcm: bytes,
    *,
    prefix: bool,
    tool_token_ids: list[int],
    tool_generation: int,
    decode_audio: bool,
    given_frame: GivenFrame | None = None,
) -> tuple[list[int], dict[str, object]]:
    """Scheduler slots and worker fields of one append.

    Rows are ``[voice prompt + system tokens if prefix] + [live frame] +
    [tool-result tokens]``, six scheduler slots each. A ``given_frame`` replays
    conversation history: the live frame hears its tokens and agent audio instead
    of the model's last prediction. Given frames must lead the conversation.
    """
    prefix_frames = (
        runtime_config["duplexio_voice_prompt_frames"] + len(runtime_config["duplexio_system_token_ids"])
        if prefix
        else 0
    )
    frame_count = prefix_frames + 1 + len(tool_token_ids)
    prompt_token_ids = [runtime_config["duplexio_scheduler_token_id"]] * (frame_count * NUM_CELLS)
    return prompt_token_ids, {
        "pcm": pcm,
        "frame_count": frame_count,
        "duplexio_prefix": prefix,
        "duplexio_tool_token_ids": tool_token_ids,
        "duplexio_tool_generation": tool_generation,
        "decode_audio": decode_audio,
        "duplexio_given_frame": given_frame,
    }


class DuplexIODuplexPlugin(DuplexModelPlugin):
    """DuplexIO-owned sampling policy, append planning, session state and output projection."""

    plugin_id = "duplexio"
    private_runtime_config_keys = PRIVATE_RUNTIME_CONFIG_KEYS
    silence_continuation_samples = FRAME_SIZE
    silence_continuation_sample_rate_hz = SAMPLE_RATE

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = DuplexIODataPlane(encode_audio)
        # One checkpoint per engine, so one tokenizer and voice cap, set at the first session open.
        self.tokenizer: Any | None = None
        self.voice_prompt_max_frames = 0

    # ---- engine policy ----

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        return stage_sampling_params(defaults)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        pcm = decode_pcm_f32le_payload(payload, sample_rate_hz=SAMPLE_RATE, exact_samples=FRAME_SIZE, model="DuplexIO")
        assert isinstance(payload, Mapping)
        results = self.data_plane.take_tool_results(request_id, runtime_config["duplexio_tool_results"])
        prompt_token_ids, fields = append_fields(
            runtime_config,
            pcm,
            prefix=seq <= 1,
            tool_token_ids=[token for result in results for token in result["token_ids"]],
            tool_generation=results[-1]["generation"] if results else 0,
            decode_audio=True,
        )
        return DuplexAppendPlan(
            prompt=build_duplex_append_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload={key: value for key, value in payload.items() if key != "audio"},
                final=final,
                prompt_token_ids=prompt_token_ids,
                model_fields=fields,
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        # Single stage: every segment already reaches the data plane as the raw stage output.
        del stage_id, final_stage_id, segment_finished, segment_token_ids, segment_output_metadata, output
        return None

    # ---- session policy ----

    def create_session_state(self) -> DefaultDuplexModelSessionState:
        return DefaultDuplexModelSessionState(
            audio_buffer=FixedFramePcmAppendBuffer(
                sample_rate_hz=SAMPLE_RATE,
                frame_samples=FRAME_SIZE,
                chunk_period_ms=CHUNK_PERIOD_MS,
                model="DuplexIO",
            )
        )

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        multi_session = max_sessions > 1
        return DuplexCapabilities(
            supports_model_native_turn_policy=True,
            supports_external_turn_signal=False,
            supports_client_commit=False,
            supports_barge_in=False,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_turn_commit_only=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_core_resumable_request=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=multi_session,
            supports_multi_session_same_replica=multi_session,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_audio_truncate=True,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            adapter_patterns=["scheduler_data_plane"],
            signal_sources=["model_native", "client_event"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=CHUNK_PERIOD_MS,
            target_barge_in_latency_ms=None,
        )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        require_full_duplex(config)
        assert model_config is not None
        runtime_config = session_runtime_config(config, model_config)
        self.tokenizer = cached_tokenizer_from_config(model_config)
        self.voice_prompt_max_frames = model_config.hf_config.voice_prompt_max_frames
        self.data_plane.configure(
            self.tokenizer.backend_tokenizer, silence_token_id=model_config.hf_config.silence_token_id
        )
        return runtime_config

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        """Nothing may change: sampling, like the prompt, is fixed when a session starts."""
        self.validate_client_extra_body(config.extra_body)
        require_full_duplex(config)
        reject_changed_runtime_value(
            session_sampling(config),
            current["duplexio_sampling"],
            message="DuplexIO sampling parameters cannot change after a session is created",
            code="sampling_update_unsupported",
        )
        reject_changed_runtime_value(
            config.instructions,
            current["instructions"],
            message="DuplexIO cannot change instructions after a session is created",
            code="instructions_update_unsupported",
        )
        tools = normalize_tools(config.extra_body.get("realtime_tools"))
        reject_changed_runtime_value(
            (tools, normalize_tool_choice(config.extra_body.get("realtime_tool_choice"), tools)),
            (current["duplexio_tools"], current["duplexio_tool_choice"]),
            message="DuplexIO cannot change tools after a session is created",
            code="tools_update_unsupported",
        )
        reject_changed_runtime_value(
            parse_start_role(config.extra_body),
            current["duplexio_start_role"],
            message="DuplexIO cannot change start_role after a session is created",
            code="start_role_update_unsupported",
        )
        if config.voice is not None:
            raise DuplexRuntimeConfigError(
                "DuplexIO has no named voices; supply ref_audio_data instead",
                code="voice_update_unsupported",
            )
        voice_prompt, _ = voice_prompt_from_session(config, self.voice_prompt_max_frames)
        reject_changed_runtime_value(
            voice_prompt,
            current["duplexio_voice_prompt_pcm"],
            message="DuplexIO cannot change the reference audio after a session is created",
            code="voice_update_unsupported",
        )
        return dict(current)

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
    ) -> dict[str, object]:
        """Queue a client tool result; the next append feeds it after its live frame."""
        del config
        output = item.get("output")
        if not isinstance(item.get("call_id"), str) or not isinstance(output, str):
            raise DuplexRuntimeConfigError(
                "function_call_output requires a call_id and a string output",
                code="invalid_function_call_output",
            )
        assert self.tokenizer is not None
        token_ids = tool_result_token_ids(self.tokenizer, output)
        results = current["duplexio_tool_results"]
        if len(results) >= MAX_PENDING_TOOL_RESULTS:
            raise DuplexRuntimeConfigError(
                f"DuplexIO has {MAX_PENDING_TOOL_RESULTS} tool results the model has not consumed yet",
                code="function_response_backlog",
            )
        generation = current["duplexio_tool_generation"] + 1
        return {
            **current,
            "duplexio_tool_generation": generation,
            "duplexio_tool_results": [*results, {"generation": generation, "token_ids": token_ids}],
        }

    def runtime_config_after_model_output(
        self,
        current: Mapping[str, object],
        output_metadata: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Retire tool results once the worker has fed them."""
        results = current["duplexio_tool_results"]
        if not results or "frame" not in output_metadata:
            return None
        consumed = frame_fields(output_metadata)["tool_generation"]
        remaining = [result for result in results if result["generation"] > consumed]
        if len(remaining) == len(results):
            return None
        return {**current, "duplexio_tool_results": remaining}


def session_runtime_config(config: DuplexSessionConfig, model_config: ModelConfig) -> dict[str, object]:
    """Validate a session and render its prompt, voice, tools and sampling; realtime and offline sessions share it."""
    hf_config = model_config.hf_config
    start_role = parse_start_role(config.extra_body)
    voice_prompt, voice_prompt_frames = voice_prompt_from_session(config, hf_config.voice_prompt_max_frames)
    tokenizer = cached_tokenizer_from_config(model_config)
    tools = normalize_tools(config.extra_body.get("realtime_tools"))
    tool_choice = normalize_tool_choice(config.extra_body.get("realtime_tool_choice"), tools)
    try:
        tool_call_grammar(tools, tool_choice)
    except ValueError as exc:
        raise DuplexRuntimeConfigError(str(exc), code="invalid_tools") from exc
    system_prompt = config.instructions or hf_config.default_system_prompt
    if tools:
        system_prompt = render_tool_system_prompt(tokenizer, system_prompt, tools)
    initial_prefix = INITIAL_PREFIXES[start_role]
    system_token_ids = [
        *tokenizer.encode(system_prompt, add_special_tokens=False),
        *tokenizer.encode(initial_prefix, add_special_tokens=False),
    ]
    return {
        "instructions": config.instructions,
        "duplexio_sampling": session_sampling(config),
        "duplexio_system_token_ids": system_token_ids,
        "duplexio_start_role": start_role,
        "duplexio_voice_prompt_pcm": voice_prompt,
        "duplexio_voice_prompt_frames": voice_prompt_frames,
        "duplexio_scheduler_token_id": hf_config.pad_token_id,
        "duplexio_tools": tools,
        "duplexio_tool_choice": tool_choice,
        "duplexio_tool_generation": 0,
        "duplexio_tool_results": [],
        # Rollouts that train on their own trajectories switch these on.
        "duplexio_record_inputs": False,
        "duplexio_record_hiddens": False,
    }


def tool_result_token_ids(tokenizer: Any, output: str) -> list[int]:
    """The rows a tool result is fed as, after the live frame."""
    return tokenizer.encode(f"<tool_response>\n{output}\n</tool_response>", add_special_tokens=False)


def require_full_duplex(config: DuplexSessionConfig) -> None:
    # The session runner hands the model's outputs to a response only in auto-response mode.
    if config.extra_body.get("auto_response") is not True:
        raise DuplexRuntimeConfigError(
            "DuplexIO decides when to speak: set extra_body.auto_response=true",
            code="full_duplex_required",
        )


def parse_start_role(extra_body: Mapping[str, object]) -> str:
    role = extra_body.get("start_role", "user")
    if role not in ("user", "agent"):
        raise DuplexRuntimeConfigError("DuplexIO start_role must be 'user' or 'agent'", code="start_role_invalid")
    return str(role)


def session_sampling(config: DuplexSessionConfig) -> dict[str, Any]:
    """``extra_body.duplexio_sampling``, with the session's ``temperature`` setting the agent's text."""
    try:
        sampling = SamplingConfig.model_validate(config.extra_body.get("duplexio_sampling", {}), strict=True)
    except ValidationError as exc:
        raise DuplexRuntimeConfigError(
            f"Invalid DuplexIO sampling configuration: {exc}", code="invalid_sampling"
        ) from exc
    if config.temperature is not None:
        sampling.agent.content.temperature = config.temperature
    return sampling.model_dump()


def voice_prompt_from_session(config: DuplexSessionConfig, max_frames: int) -> tuple[bytes, int]:
    """Decode the reference PCM (``extra_body.ref_audio_data``) and its pinned frame count."""
    extra_body = config.extra_body
    if config.ref_audio or "ref_audio_path" in extra_body:
        raise DuplexRuntimeConfigError(
            "DuplexIO takes reference audio only as extra_body.ref_audio_data",
            code="ref_audio_path_rejected",
        )
    audio_data = extra_body.get("ref_audio_data")
    if not isinstance(audio_data, str) or not audio_data:
        raise DuplexRuntimeConfigError(
            "DuplexIO requires ref_audio_data: base64 pcm_f32le reference audio for the agent's voice",
            code="ref_audio_required",
        )
    if extra_body.get("ref_audio_format", "pcm_f32le") != "pcm_f32le":
        raise DuplexRuntimeConfigError(
            "DuplexIO ref_audio_format must be pcm_f32le", code="ref_audio_format_unsupported"
        )
    if extra_body.get("ref_audio_sample_rate", SAMPLE_RATE) != SAMPLE_RATE:
        raise DuplexRuntimeConfigError(
            f"DuplexIO reference audio must be {SAMPLE_RATE} Hz mono",
            code="ref_audio_sample_rate",
        )
    try:
        raw = base64.b64decode(audio_data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise DuplexRuntimeConfigError("Invalid DuplexIO ref_audio_data", code="ref_audio_invalid") from error
    if len(raw) % 4:
        raise DuplexRuntimeConfigError("DuplexIO ref_audio_data is not whole 32-bit samples", code="ref_audio_invalid")
    frames = min(len(raw) // (4 * FRAME_SIZE), max_frames)
    if frames < 1:
        raise DuplexRuntimeConfigError(
            "DuplexIO reference audio is shorter than one 80 ms frame", code="ref_audio_too_short"
        )
    return raw[: frames * 4 * FRAME_SIZE], frames


def normalize_tools(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DuplexRuntimeConfigError("DuplexIO tools must be a list", code="invalid_tools")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_tool in value:
        try:
            tool = ChatCompletionToolsParam.model_validate(raw_tool).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise DuplexRuntimeConfigError(f"Invalid DuplexIO tool definition: {exc}", code="invalid_tools") from exc
        function = tool["function"]
        if function["name"] in names:
            raise DuplexRuntimeConfigError(f"Duplicate DuplexIO tool name {function['name']!r}", code="invalid_tools")
        names.add(function["name"])
        function.setdefault("parameters", {"type": "object", "properties": {}})
        tools.append(tool)
    return tools


def normalize_tool_choice(value: object, tools: list[dict[str, Any]]) -> dict[str, str]:
    if value is None:
        return {"mode": "auto" if tools else "none"}
    if isinstance(value, str):
        if value not in ("auto", "none", "required"):
            raise DuplexRuntimeConfigError(f"Unsupported DuplexIO tool_choice {value!r}", code="invalid_tool_choice")
        if value != "none" and not tools:
            raise DuplexRuntimeConfigError(
                f"DuplexIO tool_choice {value!r} requires at least one tool",
                code="invalid_tool_choice",
            )
        return {"mode": value}
    function = value.get("function") if isinstance(value, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    if not isinstance(value, Mapping) or value.get("type") != "function" or not isinstance(name, str) or not name:
        raise DuplexRuntimeConfigError(
            "DuplexIO tool_choice must be auto, none, required or a function selection",
            code="invalid_tool_choice",
        )
    if not any(tool["function"]["name"] == name for tool in tools):
        raise DuplexRuntimeConfigError(
            f"DuplexIO tool_choice selected unknown function {name!r}",
            code="invalid_tool_choice",
        )
    return {"mode": "named", "name": name}


def render_tool_system_prompt(tokenizer: Any, system_prompt: str, tools: list[dict[str, Any]]) -> str:
    """The system blocks the chat template renders for these tools."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": ""}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    blocks = []
    for block in rendered.split("<|im_start|>")[1:]:
        header, _, body = block.partition("\n")
        content = body.split("<|im_end|>", 1)[0].strip()
        if header.strip() == "system" and content:
            blocks.append(content)
    if not blocks:
        raise ValueError("DuplexIO tool template did not render a system block")
    return "\n".join(blocks)


__all__ = ["DuplexIODuplexPlugin", "GivenFrame", "append_fields", "stage_sampling_params"]
