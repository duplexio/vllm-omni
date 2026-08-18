# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Project native DuplexIO frame outputs into the generic duplex protocol."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import Tensor

from vllm_omni.experimental.fullduplex.duplexio.input import (
    DUPLEXIO_SAMPLE_RATE,
)
from vllm_omni.experimental.fullduplex.engine.contracts import (
    duplex_resource_request_belongs_to_session,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]
DecodeTokens = Callable[[list[int]], str]


@dataclass(frozen=True, slots=True)
class DuplexIODataPlaneContext:
    epoch: int = 0
    turn_id: int = 0
    active_response_turn_id: int | None = None
    active_response_id: str | None = None
    auto_responds: bool = True
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ("text", "audio")


class DuplexIODataPlaneSession:
    """Request lifecycle and output projection for native DuplexIO frames."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._decode_tokens: DecodeTokens | None = None
        self._silence_token_id: int | None = None
        self._terminal_request_ids: set[str] = set()
        self._audio_sample_cursors: dict[str, int] = {}
        self._agent_text_cursors: dict[str, str] = {}
        self._user_token_ids: dict[str, list[int]] = {}
        self._user_text_cursors: dict[str, str] = {}
        self._tool_call_sequences: dict[str, int] = {}

    def configure_text_decoder(
        self,
        decode_tokens: DecodeTokens,
        *,
        silence_token_id: int,
    ) -> None:
        self._decode_tokens = decode_tokens
        self._silence_token_id = silence_token_id

    @property
    def silence_token_id(self) -> int:
        if self._silence_token_id is None:
            raise RuntimeError("DuplexIO tokenizer is not configured")
        return self._silence_token_id

    def begin_request(self, request_id: str) -> None:
        self._terminal_request_ids.discard(request_id)
        self._audio_sample_cursors.setdefault(request_id, 0)
        self._agent_text_cursors.setdefault(request_id, "")
        self._user_token_ids.setdefault(request_id, [])
        self._user_text_cursors.setdefault(request_id, "")

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id in self._terminal_request_ids if request_id else False

    def mark_terminal(self, request_id: str) -> None:
        self._terminal_request_ids.add(request_id)
        self._discard_request_state(request_id)

    def close_stream(self, request_id: str) -> None:
        self._terminal_request_ids.discard(request_id)
        self._discard_request_state(request_id)

    def close_session(
        self,
        session_id: str,
        *,
        active_request_id: str | None = None,
    ) -> None:
        if active_request_id is not None:
            self._terminal_request_ids.discard(active_request_id)
            self._discard_request_state(active_request_id)
        self._terminal_request_ids = {
            request_id
            for request_id in self._terminal_request_ids
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }
        self._audio_sample_cursors = {
            request_id: cursor
            for request_id, cursor in self._audio_sample_cursors.items()
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }
        self._agent_text_cursors = self._without_session_requests(
            self._agent_text_cursors,
            session_id,
        )
        self._user_token_ids = self._without_session_requests(
            self._user_token_ids,
            session_id,
        )
        self._user_text_cursors = self._without_session_requests(
            self._user_text_cursors,
            session_id,
        )
        self._tool_call_sequences = self._without_session_requests(
            self._tool_call_sequences,
            session_id,
        )

    def _discard_request_state(self, request_id: str) -> None:
        self._audio_sample_cursors.pop(request_id, None)
        self._agent_text_cursors.pop(request_id, None)
        self._user_token_ids.pop(request_id, None)
        self._user_text_cursors.pop(request_id, None)
        self._tool_call_sequences.pop(request_id, None)

    @staticmethod
    def _without_session_requests(
        values: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        return {
            request_id: value
            for request_id, value in values.items()
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }

    def project(
        self,
        result: object,
        *,
        context: object | None = None,
    ) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        typed_context = (
            context
            if isinstance(context, DuplexIODataPlaneContext)
            else DuplexIODataPlaneContext()
        )
        for output in outputs:
            projected = self.project_output(output, context=typed_context)
            if projected is not None:
                yield projected

    def project_output(
        self,
        output: object,
        *,
        context: DuplexIODataPlaneContext,
    ) -> dict[str, object] | None:
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            return None
        if request_id in self._terminal_request_ids:
            return None

        completion = _first_completion(output)
        metadata = _multimodal_output(output, completion)
        output_epoch = _metadata_int(metadata, "duplex_epoch", "epoch")
        if output_epoch is not None and output_epoch != context.epoch:
            return None
        prefill_complete = _metadata_bool(
            metadata,
            "duplex_prefill_complete",
            default=False,
        )
        if _metadata_bool(
            metadata,
            "duplex_prefill",
            default=False,
        ) or _metadata_bool(
            metadata,
            "duplex_system_input",
            default=False,
        ):
            return None

        model_turn_id = _metadata_int(metadata, "duplex_turn_id", "turn_id")
        text = getattr(completion, "text", "") if completion is not None else ""
        metadata_text = metadata.get("agent_text")
        if isinstance(metadata_text, str):
            text = metadata_text
        if not isinstance(text, str):
            text = ""
        text = self._text_delta(
            request_id,
            text,
            cursors=self._agent_text_cursors,
            stream="agent",
        )

        user_token_id = _metadata_int(metadata, "user_token_id")
        input_text_delta = self._user_text_delta(request_id, user_token_id)
        tool_call = _metadata_json_mapping(metadata, "tool_call_json")
        if tool_call is not None:
            sequence = tool_call.pop("sequence")
            assert isinstance(sequence, int)
            previous_sequence = self._tool_call_sequences.get(request_id, 0)
            if sequence <= previous_sequence:
                tool_call = None
            else:
                self._tool_call_sequences[request_id] = sequence

        model_listen = _metadata_bool(metadata, "model_listen", default=False)
        end_of_turn = _metadata_bool(metadata, "end_of_turn", default=False)
        raw_audio = metadata.get("audio", metadata.get("model_outputs"))
        audio_data = None
        audio_duration_ms = 0
        sample_count = _audio_num_samples(raw_audio) if raw_audio is not None else 0
        emitted_samples = self._audio_sample_cursors.get(request_id, 0)
        if sample_count < emitted_samples:
            raise RuntimeError(
                "DuplexIO accumulated audio moved backwards for "
                f"{request_id}: {sample_count} < {emitted_samples}"
            )
        if raw_audio is not None:
            raw_audio = _audio_slice(raw_audio, emitted_samples)
            self._audio_sample_cursors[request_id] = sample_count
            sample_count -= emitted_samples
        if sample_count > 0 and "audio" in context.modalities:
            sample_rate_hz = _metadata_int(
                metadata,
                "sample_rate_hz",
                "sample_rate",
                "sr",
            )
            if sample_rate_hz is None:
                sample_rate_hz = DUPLEXIO_SAMPLE_RATE
            audio_data = self._encode_audio(
                raw_audio,
                sample_rate_hz,
                context.response_format,
                context.speed,
            )
            audio_duration_ms = round(sample_count * 1_000 / sample_rate_hz)

        if model_listen and not text and not audio_data and tool_call is None:
            result = _runtime_result(
                stage_role="duplexio",
                is_listen=True,
                model_listen=True,
                listen_source="model_native",
                data_plane_request_id=request_id,
                end_of_turn=end_of_turn,
            )
        elif (
            not text
            and not audio_data
            and not end_of_turn
            and not input_text_delta
            and tool_call is None
            and not prefill_complete
        ):
            return None
        else:
            result = _runtime_result(
                stage_role="duplexio",
                is_listen=False,
                model_listen=False,
                data_plane_request_id=request_id,
                text=text if "text" in context.modalities else "",
                audio_data=audio_data,
                audio_format=context.response_format,
                audio_duration_ms=audio_duration_ms,
                audio_text_mark=bool(text and audio_data),
                sample_rate_hz=DUPLEXIO_SAMPLE_RATE,
                end_of_turn=end_of_turn,
                abort_data_plane_request=False,
            )
        if model_turn_id is not None:
            result["model_turn_id"] = model_turn_id
        if input_text_delta and "text" in context.modalities:
            result["input_text_delta"] = input_text_delta
        if tool_call is not None:
            result["tool_call"] = tool_call
        if prefill_complete:
            result["initial_data_plane_complete"] = True
        for name in ("user_token_id", "agent_token_id", "tool_call_token_id"):
            token_id = _metadata_int(metadata, name)
            if token_id is not None:
                result[name] = token_id
        audio_token_ids = _metadata_int_list(metadata, "agent_audio_token_ids")
        if audio_token_ids:
            result["agent_audio_token_ids"] = audio_token_ids
        return result

    def _user_text_delta(
        self,
        request_id: str,
        token_id: int | None,
    ) -> str:
        if (
            token_id is None
            or token_id == self._silence_token_id
            or self._decode_tokens is None
        ):
            return ""
        token_ids = self._user_token_ids.setdefault(request_id, [])
        token_ids.append(token_id)
        text = self._decode_tokens(token_ids)
        return self._text_delta(
            request_id,
            text,
            cursors=self._user_text_cursors,
            stream="user",
        )

    @staticmethod
    def _text_delta(
        request_id: str,
        text: str,
        *,
        cursors: dict[str, str],
        stream: str,
    ) -> str:
        previous = cursors.get(request_id, "")
        if not text.startswith(previous):
            raise RuntimeError(
                f"DuplexIO accumulated {stream} text moved backwards for "
                f"{request_id}: {text!r} does not extend {previous!r}"
            )
        cursors[request_id] = text
        return text[len(previous) :]


def _first_completion(output: object) -> object | None:
    outputs = getattr(output, "outputs", None)
    return outputs[0] if isinstance(outputs, list) and outputs else None


def _multimodal_output(
    output: object,
    completion: object | None,
) -> dict[str, Any]:
    for candidate in (
        getattr(output, "multimodal_output", None),
        getattr(completion, "multimodal_output", None),
    ):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    inner = getattr(output, "request_output", None)
    if inner is not None and inner is not output:
        return _multimodal_output(inner, _first_completion(inner))
    return {}


def _metadata_int(metadata: Mapping[str, object], *names: str) -> int | None:
    nested = metadata.get("meta")
    for name in names:
        candidates = [metadata.get(name), metadata.get(f"meta.{name}")]
        if isinstance(nested, Mapping):
            candidates.append(nested.get(name))
        for value in candidates:
            scalar = _scalar(value)
            if scalar is not None:
                try:
                    return int(scalar)
                except (TypeError, ValueError):
                    pass
    return None


def _metadata_int_list(
    metadata: Mapping[str, object],
    name: str,
) -> list[int] | None:
    value = metadata.get(name)
    if isinstance(value, Tensor):
        return [int(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.flatten().tolist()]
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return value
    return None


def _metadata_bool(
    metadata: Mapping[str, object],
    name: str,
    *,
    default: bool,
) -> bool:
    nested = metadata.get("meta")
    candidates = [metadata.get(name), metadata.get(f"meta.{name}")]
    if isinstance(nested, Mapping):
        candidates.append(nested.get(name))
    for value in candidates:
        scalar = _scalar(value)
        if scalar is not None:
            return bool(scalar)
    return default


def _metadata_json_mapping(
    metadata: Mapping[str, object],
    name: str,
) -> dict[str, object] | None:
    nested = metadata.get("meta")
    candidates = [metadata.get(name), metadata.get(f"meta.{name}")]
    if isinstance(nested, Mapping):
        candidates.append(nested.get(name))
    for value in candidates:
        payload = _metadata_bytes(value)
        if payload:
            text = payload.decode("utf-8")
            decoder = json.JSONDecoder()
            offset = 0
            decoded = None
            while offset < len(text):
                decoded, offset = decoder.raw_decode(text, offset)
                while offset < len(text) and text[offset].isspace():
                    offset += 1
            if not isinstance(decoded, dict):
                raise RuntimeError(
                    "DuplexIO tool-call wire payload is not an object"
                )
            name_value = decoded.get("name")
            arguments = decoded.get("arguments")
            sequence = decoded.get("sequence")
            if (
                not isinstance(name_value, str)
                or not isinstance(arguments, dict)
                or not isinstance(sequence, int)
                or sequence < 1
            ):
                raise RuntimeError("DuplexIO tool-call wire payload is invalid")
            return {
                "sequence": sequence,
                "name": name_value,
                "arguments": arguments,
            }
    return None


def _metadata_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().reshape(-1)
        return bytes(tensor.tolist()) if tensor.numel() else None
    if isinstance(value, np.ndarray):
        return bytes(value.reshape(-1).tolist()) if value.size else None
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, int) for item in value):
            return bytes(value)
        return _metadata_bytes(value[-1]) if value else None
    return None


def _scalar(value: Any) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1)
            return value[-1].item() if value.numel() else None
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[-1].item() if value.size else None
    if isinstance(value, (list, tuple)):
        return _scalar(value[-1]) if value else None
    return value


def _audio_num_samples(audio: Any) -> int:
    if hasattr(audio, "numel"):
        return int(audio.numel())
    return int(np.asarray(audio, dtype=np.float32).size)


def _audio_slice(audio: Any, start: int) -> Any:
    if hasattr(audio, "reshape"):
        return audio.reshape(-1)[start:]
    return np.asarray(audio, dtype=np.float32).reshape(-1)[start:]


def _runtime_result(**values: object) -> dict[str, object]:
    return {
        "supported": True,
        **values,
        "uses_model_runner_scheduler": True,
        "runner_kv_backed": True,
        "runtime_impl": "scheduler_data_plane",
        "owned_runtime": False,
    }


__all__ = ["DuplexIODataPlaneContext", "DuplexIODataPlaneSession"]
