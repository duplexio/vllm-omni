# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving adapter for vLLM-Omni's native DuplexIO runtime."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tokenizers.registry import cached_tokenizer_from_config

from vllm_omni.experimental.fullduplex.duplexio.data_plane import (
    DuplexIODataPlaneContext,
    DuplexIODataPlaneSession,
)
from vllm_omni.experimental.fullduplex.duplexio.input import (
    DUPLEXIO_FRAME_SIZE,
    DUPLEXIO_SAMPLE_RATE,
)
from vllm_omni.experimental.fullduplex.duplexio.session import (
    DuplexIOServingSessionState,
)
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    ServingRuntimeConfigError,
)
from vllm_omni.model_executor.models.duplexio.checkpoint import (
    resolve_checkpoint_directory,
)
from vllm_omni.model_executor.models.duplexio.tool_calling import (
    tool_call_grammar,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]
PREFILL_CHUNK_FRAMES = 128
CLIENT_SAMPLING_CONFIG_KEY = "duplexio_sampling"


class DuplexIOTextSamplingConfig(BaseModel):
    """Client overrides for the DuplexIO text-token sampler."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["argmax", "top_k", "top_p"] | None = None
    temperature: float | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)


class DuplexIOAudioSamplingConfig(BaseModel):
    """Client overrides for the DuplexIO audio-code sampler."""

    model_config = ConfigDict(extra="forbid", strict=True)

    temperature: float | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, ge=1)


class DuplexIOEmitSamplingConfig(BaseModel):
    """Per-stream temperatures for binary emit decisions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    user: float | None = Field(default=None, ge=0)
    agent: float | None = Field(default=None, ge=0)
    tool_call: float | None = Field(default=None, ge=0)


class DuplexIOClientSamplingConfig(BaseModel):
    """Validated public sampling configuration for one DuplexIO session."""

    model_config = ConfigDict(extra="forbid", strict=True)

    seed: int | None = Field(default=None, ge=0, lt=2**63)
    text: DuplexIOTextSamplingConfig | None = None
    audio: DuplexIOAudioSamplingConfig | None = None
    emit: DuplexIOEmitSamplingConfig | None = None


def parse_client_sampling_config(
    extra_body: Mapping[str, object],
) -> DuplexIOClientSamplingConfig:
    raw = extra_body.get(CLIENT_SAMPLING_CONFIG_KEY, {})
    try:
        return DuplexIOClientSamplingConfig.model_validate(raw)
    except ValidationError as exc:
        raise DuplexIOClientRuntimeConfigError(
            f"Invalid DuplexIO sampling configuration: {exc}",
            code="invalid_sampling",
        ) from exc


class DuplexIOClientRuntimeConfigError(ServingRuntimeConfigError):
    pass


class DuplexIOServingRuntimeAdapter:
    """DuplexIO-owned session framing, policy, and output projection."""

    adapter_id = "duplexio"
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = frozenset(
        {
            "duplexio_scheduler_token_id",
            "duplexio_sampling_seed",
            "duplexio_start_role",
            "duplexio_voice_ids",
            "duplexio_voice_embedding_index",
            "duplexio_depth_sampling",
            "duplexio_emit_temperatures",
            "duplexio_text_sampling",
            "duplexio_suppressed_token_ids",
            "duplexio_agent_suppressed_token_ids",
            "duplexio_tools",
            "duplexio_tool_choice",
            "duplex_stage_max_tokens",
        }
    )

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, DuplexIOServingSessionState] = {}
        self.data_plane = DuplexIODataPlaneSession(encode_audio)
        self.tokenizer: Any | None = None

    def create_session_state(self) -> DuplexIOServingSessionState:
        return DuplexIOServingSessionState()

    def session_state(self, session_id: str) -> DuplexIOServingSessionState:
        state = self.session_states.get(session_id)
        if state is None:
            state = self.create_session_state()
            self.session_states[session_id] = state
        return state

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    @staticmethod
    def is_enabled(config: object) -> bool:
        del config
        return True

    @staticmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        multi_session = max_sessions > 1
        return DuplexCapabilities(
            supports_model_native_turn_policy=True,
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
            implementation_level="model_native_duplex",
            adapter_patterns=["scheduler_data_plane"],
            input_modes=["append_audio_chunk"],
            signal_sources=["model_native", "client_event", "server_policy"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=80,
            target_barge_in_latency_ms=None,
        )

    @classmethod
    def validate_client_extra_body(cls, extra_body: object) -> None:
        if not isinstance(extra_body, dict):
            return
        private_keys = sorted(cls.private_runtime_config_keys.intersection(extra_body))
        if private_keys:
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO runtime configuration is server-owned: "
                + ", ".join(private_keys)
            )

    async def prepare_runtime_config(
        self,
        config: object,
        *,
        model_config: Any,
    ) -> dict[str, object]:
        if not isinstance(config, DuplexSessionConfig):
            raise TypeError("DuplexIO serving requires DuplexSessionConfig")
        self.validate_client_extra_body(config.extra_body)
        _validate_full_duplex_mode(config)
        start_role = _start_role(config.extra_body)
        hf_config = getattr(model_config, "hf_config", model_config)
        voice_ids, manifest_default = _load_voice_manifest(model_config)
        configured_default = getattr(hf_config, "default_voice", None)
        voice = config.voice or configured_default or manifest_default
        if voice is None:
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO requires a named exported voice for speaker conditioning",
                code="voice_required",
            )
        if voice is not None and voice not in voice_ids:
            raise DuplexIOClientRuntimeConfigError(
                f"Unknown DuplexIO voice {voice!r}",
                code="voice_not_found",
            )

        depth = getattr(hf_config, "depth_transformer_config", {})
        if not isinstance(depth, Mapping):
            depth = {}
        scheduler_token_id = getattr(hf_config, "pad_token_id", None)
        if not isinstance(scheduler_token_id, int) or scheduler_token_id < 0:
            scheduler_token_id = 0
        system_prompt = config.instructions or getattr(
            hf_config,
            "default_system_prompt",
            "",
        )
        tokenizer = cached_tokenizer_from_config(model_config)
        tools = normalize_realtime_tools(config.extra_body.get("realtime_tools"))
        tool_choice = normalize_realtime_tool_choice(
            config.extra_body.get("realtime_tool_choice"),
            tools,
        )
        try:
            tool_call_grammar(tools, tool_choice)
        except ValueError as exc:
            raise DuplexIOClientRuntimeConfigError(
                str(exc),
                code="invalid_tools",
            ) from exc
        if tools:
            system_prompt = render_tool_system_prompt(
                tokenizer,
                system_prompt,
                tools,
            )
        silence_token_id = getattr(hf_config, "silence_token_id", None)
        if not isinstance(silence_token_id, int) or silence_token_id < 0:
            raise ValueError("DuplexIO checkpoint is missing silence_token_id")
        self.tokenizer = tokenizer
        self.data_plane.configure_text_decoder(
            lambda token_ids: tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ),
            silence_token_id=silence_token_id,
        )
        system_token_ids = tokenizer.encode(
            system_prompt,
            add_special_tokens=False,
        )
        initial_prefix = (
            hf_config.initial_agent_prefix
            if start_role == "agent"
            else hf_config.initial_user_prefix
        )
        initial_prefix_ids = tokenizer.encode(
            initial_prefix,
            add_special_tokens=False,
        )
        client_sampling = parse_client_sampling_config(config.extra_body)
        text_sampling = dict(getattr(hf_config, "rollout_sampling_config", {}))
        if client_sampling.text is not None:
            text_sampling.update(
                client_sampling.text.model_dump(exclude_none=True)
            )
        if config.temperature is not None:
            text_sampling["temperature"] = config.temperature
        emit_temperatures = {
            "user": 0.0,
            "agent": 1.0,
            "tool_call": 1.0,
        }
        if client_sampling.emit is not None:
            emit_temperatures.update(
                client_sampling.emit.model_dump(exclude_none=True)
            )
        depth_sampling = {
            "temperature": 0.7,
            "top_k": depth.get("sampling_top_k", 250),
        }
        if client_sampling.audio is not None:
            audio_sampling = client_sampling.audio.model_dump(exclude_none=True)
            audio_top_k = audio_sampling.get("top_k")
            if audio_top_k is not None:
                quantized_audio = getattr(hf_config, "quantized_audio_config", {})
                codebook_size = (
                    quantized_audio.get("codebook_size")
                    if isinstance(quantized_audio, Mapping)
                    else None
                )
                if not isinstance(codebook_size, int) or codebook_size < 1:
                    raise ValueError(
                        "DuplexIO checkpoint is missing its audio codebook size"
                    )
                if audio_top_k > codebook_size:
                    raise DuplexIOClientRuntimeConfigError(
                        "DuplexIO audio sampling top_k exceeds the "
                        f"{codebook_size}-entry codebook",
                        code="invalid_sampling",
                    )
            depth_sampling.update(audio_sampling)
        return {
            "instructions": config.instructions,
            "duplexio_system_token_ids": [
                int(token)
                for token in (*system_token_ids, *initial_prefix_ids)
            ],
            "duplexio_start_role": start_role,
            "duplexio_voice": voice,
            "duplexio_voice_ids": list(voice_ids),
            "duplexio_voice_embedding_index": 0,
            "duplexio_scheduler_token_id": scheduler_token_id,
            "duplexio_sampling_seed": (
                client_sampling.seed
                if client_sampling.seed is not None
                else secrets.randbits(63)
            ),
            "duplexio_emit_temperatures": emit_temperatures,
            "duplexio_text_sampling": text_sampling,
            "duplexio_suppressed_token_ids": _suppressed_special_token_ids(
                tokenizer,
                silence_token_id,
            ),
            "duplexio_agent_suppressed_token_ids": (
                _agent_suppressed_token_ids(tokenizer, silence_token_id)
            ),
            "duplexio_tools": tools,
            "duplexio_tool_choice": tool_choice,
            "duplexio_depth_sampling": depth_sampling,
            "duplex_stage_max_tokens": {"0": 1},
        }

    def initial_data_plane_payloads(
        self,
        session: object,
    ) -> tuple[dict[str, object], ...]:
        """Return bounded zero-audio frame batches for the training prompt."""
        runtime_config = getattr(session, "runtime_config", None)
        if not isinstance(runtime_config, Mapping):
            raise RuntimeError("DuplexIO session is missing runtime_config")
        system_token_ids = runtime_config.get("duplexio_system_token_ids")
        if not isinstance(system_token_ids, list | tuple) or not all(
            isinstance(token, int) for token in system_token_ids
        ):
            raise RuntimeError(
                "DuplexIO session runtime_config has invalid system token IDs"
            )
        turn_id = getattr(session, "turn_id", None)
        if not isinstance(turn_id, int):
            raise RuntimeError("DuplexIO session is missing an integer turn_id")
        if not system_token_ids:
            return ()
        payloads = []
        for offset in range(0, len(system_token_ids), PREFILL_CHUNK_FRAMES):
            frame_count = min(
                PREFILL_CHUNK_FRAMES,
                len(system_token_ids) - offset,
            )
            payloads.append({
                "type": "audio",
                "audio": "",
                "format": "pcm_f32le",
                "sample_rate_hz": DUPLEXIO_SAMPLE_RATE,
                "frame_size": DUPLEXIO_FRAME_SIZE,
                "frame_count": frame_count,
                "valid_samples": frame_count * DUPLEXIO_FRAME_SIZE,
                "force_listen": False,
                "is_speech": False,
                "duplex_turn_id": turn_id,
                "duplexio_prefill": True,
                "duplexio_prefill_final": (
                    offset + frame_count == len(system_token_ids)
                ),
            })
        return tuple(payloads)

    def tool_result_data_plane_payloads(
        self,
        session: object,
        output: str,
    ) -> tuple[dict[str, object], ...]:
        if self.tokenizer is None:
            raise RuntimeError("DuplexIO tokenizer is not configured")
        runtime_config = getattr(session, "runtime_config", None)
        turn_id = getattr(session, "turn_id", None)
        if not isinstance(runtime_config, Mapping) or not isinstance(turn_id, int):
            raise RuntimeError("DuplexIO session is missing runtime state")
        rendered = f"<tool_response>\n{output}\n</tool_response>"
        token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError("DuplexIO tokenizer produced no tool-result tokens")
        payloads = []
        for offset in range(0, len(token_ids), PREFILL_CHUNK_FRAMES):
            token_chunk = token_ids[offset : offset + PREFILL_CHUNK_FRAMES]
            frame_count = len(token_chunk)
            payloads.append({
                "type": "audio",
                "audio": "",
                "format": "pcm_f32le",
                "sample_rate_hz": DUPLEXIO_SAMPLE_RATE,
                "frame_size": DUPLEXIO_FRAME_SIZE,
                "frame_count": frame_count,
                "valid_samples": frame_count * DUPLEXIO_FRAME_SIZE,
                "force_listen": False,
                "is_speech": False,
                "duplex_turn_id": turn_id,
                "duplexio_system_input": True,
                "duplexio_system_input_final": (
                    offset + frame_count == len(token_ids)
                ),
                "duplexio_system_token_ids": token_chunk,
            })
        return tuple(payloads)

    @classmethod
    def runtime_config_for_update(
        cls,
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(config, DuplexSessionConfig):
            raise TypeError("DuplexIO serving requires DuplexSessionConfig")
        cls.validate_runtime_config_for_session(config, current)
        updated = dict(current)
        if config.temperature is not None:
            current_text_sampling = current["duplexio_text_sampling"]
            assert isinstance(current_text_sampling, Mapping)
            text_sampling = dict(current_text_sampling)
            text_sampling["temperature"] = config.temperature
            updated["duplexio_text_sampling"] = text_sampling
        return updated

    @staticmethod
    def validate_runtime_config_for_session(
        config: object,
        current: Mapping[str, object],
    ) -> None:
        if not isinstance(config, DuplexSessionConfig):
            raise TypeError("DuplexIO serving requires DuplexSessionConfig")
        _validate_full_duplex_mode(config)
        if CLIENT_SAMPLING_CONFIG_KEY in config.extra_body:
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO sampling parameters cannot change after a session is created",
                code="sampling_update_unsupported",
            )
        if config.instructions != current.get("instructions"):
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO cannot change instructions after a session is created",
                code="instructions_update_unsupported",
            )
        tools = normalize_realtime_tools(config.extra_body.get("realtime_tools"))
        tool_choice = normalize_realtime_tool_choice(
            config.extra_body.get("realtime_tool_choice"),
            tools,
        )
        if tools != current.get("duplexio_tools", []) or tool_choice != current.get(
            "duplexio_tool_choice",
            {"mode": "none"},
        ):
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO cannot change tools after a session is created",
                code="tools_update_unsupported",
            )
        if "start_role" in config.extra_body:
            start_role = _start_role(config.extra_body)
            if start_role != current.get("duplexio_start_role"):
                raise DuplexIOClientRuntimeConfigError(
                    "DuplexIO cannot change start_role after a session is created",
                    code="start_role_update_unsupported",
                )
        if (
            config.voice is not None
            and config.voice != current.get("duplexio_voice")
        ):
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO cannot change voice after a session is created",
                code="voice_update_unsupported",
            )
        voice = config.voice or current.get("duplexio_voice")
        if not isinstance(voice, str) or not voice:
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO requires a named exported voice for speaker conditioning",
                code="voice_required",
            )
        voice_ids = current.get("duplexio_voice_ids")
        if not isinstance(voice_ids, list) or voice not in voice_ids:
            raise DuplexIOClientRuntimeConfigError(
                f"Unknown DuplexIO voice {voice!r}",
                code="voice_not_found",
            )

    @staticmethod
    def data_plane_context(
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> DuplexIODataPlaneContext:
        return DuplexIODataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )


def normalize_realtime_tools(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DuplexIOClientRuntimeConfigError(
            "DuplexIO tools must be a list",
            code="invalid_tools",
        )
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_tool in value:
        try:
            tool = ChatCompletionToolsParam.model_validate(raw_tool)
        except ValidationError as exc:
            raise DuplexIOClientRuntimeConfigError(
                f"Invalid DuplexIO tool definition: {exc}",
                code="invalid_tools",
            ) from exc
        normalized = tool.model_dump(exclude_none=True)
        function = normalized["function"]
        name = function["name"]
        if name in names:
            raise DuplexIOClientRuntimeConfigError(
                f"Duplicate DuplexIO tool name {name!r}",
                code="invalid_tools",
            )
        names.add(name)
        function.setdefault(
            "parameters",
            {"type": "object", "properties": {}},
        )
        tools.append(normalized)
    return tools


def normalize_realtime_tool_choice(
    value: object,
    tools: list[dict[str, Any]],
) -> dict[str, str]:
    if value is None:
        return {"mode": "auto" if tools else "none"}
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise DuplexIOClientRuntimeConfigError(
                f"Unsupported DuplexIO tool_choice {value!r}",
                code="invalid_tool_choice",
            )
        if value != "none" and not tools:
            raise DuplexIOClientRuntimeConfigError(
                f"DuplexIO tool_choice {value!r} requires at least one tool",
                code="invalid_tool_choice",
            )
        return {"mode": value}
    if not isinstance(value, Mapping):
        raise DuplexIOClientRuntimeConfigError(
            "DuplexIO tool_choice must be a string or function selection",
            code="invalid_tool_choice",
        )
    function = value.get("function")
    name = function.get("name") if isinstance(function, Mapping) else None
    if value.get("type") != "function" or not isinstance(name, str) or not name:
        raise DuplexIOClientRuntimeConfigError(
            "DuplexIO named tool_choice must select a function name",
            code="invalid_tool_choice",
        )
    if not any(tool["function"]["name"] == name for tool in tools):
        raise DuplexIOClientRuntimeConfigError(
            f"DuplexIO tool_choice selected unknown function {name!r}",
            code="invalid_tool_choice",
        )
    return {"mode": "named", "name": name}


def render_tool_system_prompt(
    tokenizer: Any,
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> str:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("DuplexIO tokenizer chat template did not return text")
    system_blocks: list[str] = []
    cursor = 0
    while True:
        block_start = rendered.find("<|im_start|>", cursor)
        if block_start < 0:
            break
        role_start = block_start + len("<|im_start|>")
        header_end = rendered.find("\n", role_start)
        block_end = rendered.find("<|im_end|>", header_end + 1)
        if header_end < 0 or block_end < 0:
            raise ValueError("DuplexIO tokenizer rendered an invalid chat template")
        if rendered[role_start:header_end].strip() == "system":
            content = rendered[header_end + 1 : block_end].strip()
            if content:
                system_blocks.append(content)
        cursor = block_end + len("<|im_end|>")
    if not system_blocks:
        raise ValueError("DuplexIO tool template did not render a system block")
    return "\n".join(system_blocks)


def _load_voice_manifest(model_config: object) -> tuple[tuple[str, ...], str | None]:
    model_path = getattr(model_config, "model", None)
    if not isinstance(model_path, str) or not model_path:
        return (), None
    revision = getattr(model_config, "revision", None)
    root = resolve_checkpoint_directory(model_path, revision=revision)
    path = root / "voices.json"
    if not path.is_file():
        return (), None
    value = json.loads(path.read_text())
    voices = value.get("voices") if isinstance(value, dict) else None
    if not isinstance(voices, dict) or any(not isinstance(name, str) for name in voices):
        raise ValueError(f"Invalid DuplexIO voice manifest: {path}")
    default = value.get("default_voice")
    if default is not None and not isinstance(default, str):
        raise ValueError(f"Invalid DuplexIO default voice in {path}")
    if default is not None and default not in voices:
        raise ValueError(f"DuplexIO default voice is not present in {path}")
    return tuple(sorted(voices)), default


def _suppressed_special_token_ids(
    tokenizer: Any,
    silence_token_id: int,
) -> list[int]:
    token_ids = {
        int(token_id)
        for token_id in tokenizer.all_special_ids
        if int(token_id) != silence_token_id
    }
    for text in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1 and int(encoded[0]) != silence_token_id:
            token_ids.add(int(encoded[0]))
    return sorted(token_ids)


def _agent_suppressed_token_ids(
    tokenizer: Any,
    silence_token_id: int,
) -> list[int]:
    token_ids = []
    for text in ("<tool_call>", "</tool_call>"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1 and int(encoded[0]) != silence_token_id:
            token_ids.append(int(encoded[0]))
    return sorted(set(token_ids))


def _validate_full_duplex_mode(config: DuplexSessionConfig) -> None:
    if (
        config.extra_body.get("full_duplex") is True
        or config.extra_body.get("auto_response") is True
    ):
        return
    raise DuplexIOClientRuntimeConfigError(
        "DuplexIO requires full-duplex auto-response mode",
        code="full_duplex_required",
    )


def _start_role(extra_body: Mapping[str, object]) -> str:
    role = extra_body.get("start_role", "user")
    if not isinstance(role, str) or role not in {"user", "agent"}:
        raise DuplexIOClientRuntimeConfigError(
            "DuplexIO start_role must be 'user' or 'agent'",
            code="start_role_invalid",
        )
    return role


__all__ = ["DuplexIOServingRuntimeAdapter"]
