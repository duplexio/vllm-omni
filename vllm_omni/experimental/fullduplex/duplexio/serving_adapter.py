# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving adapter for vLLM-Omni's native DuplexIO runtime."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping
from typing import Any

from vllm.tokenizers.registry import cached_tokenizer_from_config

from vllm_omni.experimental.fullduplex.duplexio.data_plane import (
    DuplexIODataPlaneContext,
    DuplexIODataPlaneSession,
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

EncodeAudio = Callable[[object, int, str, float | None], str | None]


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
            "duplex_stage_max_tokens",
        }
    )

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, DuplexIOServingSessionState] = {}
        self.data_plane = DuplexIODataPlaneSession(encode_audio)

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

    @classmethod
    async def prepare_runtime_config(
        cls,
        config: object,
        *,
        model_config: Any,
    ) -> dict[str, object]:
        if not isinstance(config, DuplexSessionConfig):
            raise TypeError("DuplexIO serving requires DuplexSessionConfig")
        cls.validate_client_extra_body(config.extra_body)
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
            "duplexio_sampling_seed": secrets.randbits(63),
            "duplexio_text_temperature": (
                config.temperature if config.temperature is not None else 0.0
            ),
            "duplexio_depth_sampling": {
                "temperature": depth.get("sampling_temperature", 0.8),
                "top_k": depth.get("sampling_top_k", 250),
            },
            "duplex_stage_max_tokens": {"0": 1},
        }

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
        updated["duplexio_text_temperature"] = (
            config.temperature if config.temperature is not None else 0.0
        )
        return updated

    @staticmethod
    def validate_runtime_config_for_session(
        config: object,
        current: Mapping[str, object],
    ) -> None:
        if not isinstance(config, DuplexSessionConfig):
            raise TypeError("DuplexIO serving requires DuplexSessionConfig")
        _validate_full_duplex_mode(config)
        if config.instructions != current.get("instructions"):
            raise DuplexIOClientRuntimeConfigError(
                "DuplexIO cannot change instructions after a session is created",
                code="instructions_update_unsupported",
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
