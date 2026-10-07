# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Project native DuplexIO frame outputs into internal duplex events."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from tokenizers import Tokenizer, decoders

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, DuplexDataPlaneContext, EncodeAudio
from vllm_omni.model_executor.common.request_outputs import (
    audio_sample_count,
    audio_value,
    multimodal_output,
    unwrap_request_output,
)
from vllm_omni.model_executor.models.duplexio.frame_output import frame_fields


class DuplexIODataPlane(DuplexDataPlane):
    """Output projection, and the per-request state its lifecycle calls clean up."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.encode_audio = encode_audio
        self.tokenizer: Tokenizer | None = None
        self.silence_token_id: int | None = None
        self.terminal: set[str] = set()
        self.user_decoders: dict[str, decoders.DecodeStream] = {}
        # The newest tool result each stage request has been fed.
        self.planned_tool_generations: dict[str, int] = {}

    def configure(self, tokenizer: Tokenizer, *, silence_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.silence_token_id = silence_token_id

    def take_tool_results(self, request_id: str, pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The pending tool results the request has not been fed yet, now marked fed.

        Results stay pending until the worker reports them, so an append planned
        before that must not feed them again.
        """
        planned = self.planned_tool_generations.get(request_id, 0)
        results = [result for result in pending if result["generation"] > planned]
        if results:
            self.planned_tool_generations[request_id] = results[-1]["generation"]
        return results

    def begin_request(self, request_id: str) -> None:
        self.terminal.discard(request_id)

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id in self.terminal

    def mark_terminal(self, request_id: str) -> None:
        self.terminal.add(request_id)
        self.user_decoders.pop(request_id, None)

    def close_stream(self, request_id: str) -> None:
        self.terminal.discard(request_id)
        self.user_decoders.pop(request_id, None)
        self.planned_tool_generations.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self.close_stream(active_request_id)
        for request_ids in (self.terminal, self.user_decoders, self.planned_tool_generations):
            for request_id in [
                key for key in request_ids if duplex_resource_request_belongs_to_session(key, session_id)
            ]:
                self.close_stream(request_id)

    def project(self, result: object, *, context: object | None = None) -> Iterator[dict[str, object]]:
        assert isinstance(result, dict) and isinstance(context, DuplexDataPlaneContext)
        # Append acknowledgements pass through here too; they carry no outputs.
        for output in result.get("data_plane_outputs", ()):
            yield from self.project_output(output, context)

    def project_output(self, output: Any, context: DuplexDataPlaneContext) -> Iterator[dict[str, object]]:
        request_id = output.request_id
        if request_id in self.terminal:
            return
        output, completion = unwrap_request_output(output)
        metadata = multimodal_output(output, completion)
        # Chunks of a long append before its last row carry no prediction.
        if "frame" not in metadata:
            return
        fields = frame_fields(metadata)
        if not fields["predicted"] or fields["duplex_epoch"] != context.epoch:
            return
        tool_call = metadata.get("tool_call_json")
        if tool_call is not None and tool_call.numel():
            call = json.loads(bytes(tool_call.tolist()))
            yield {
                "stage_role": "function",
                "data_plane_request_id": request_id,
                "function_call": True,
                "call_id": f"call_{uuid4().hex}",
                "name": call["name"],
                "arguments": json.dumps(call["arguments"]),
            }
        text = completion.text or ""
        # Step the user decoder on every frame, so it stays in sync if text is switched back on.
        user_text = self.user_text(request_id, fields["user_token_id"])
        if "text" not in context.modalities:
            text = user_text = ""
        audio = audio_value(metadata)
        samples = audio_sample_count(audio) or 0
        audio_data = (
            self.encode_audio(audio, fields["sample_rate_hz"], context.response_format, context.speed)
            if samples and "audio" in context.modalities
            else None
        )
        if fields["model_listen"] and not text and not audio_data:
            yield {
                "stage_role": "duplexio",
                "is_listen": True,
                "model_listen": True,
                "listen_source": "model_native",
                "data_plane_request_id": request_id,
                "model_turn_id": fields["duplex_turn_id"],
                "input_text_delta": user_text,
                "end_of_turn": fields["end_of_turn"],
            }
            return
        if not text and not audio_data and not user_text and not fields["end_of_turn"]:
            return
        yield {
            "stage_role": "duplexio",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "model_turn_id": fields["duplex_turn_id"],
            "text": text,
            "audio_data": audio_data,
            "audio_format": context.response_format,
            "sample_rate_hz": fields["sample_rate_hz"],
            "audio_duration_ms": round(samples * 1_000 / fields["sample_rate_hz"]),
            "audio_text_mark": bool(text and audio_data),
            "input_text_delta": user_text,
            "end_of_turn": fields["end_of_turn"],
        }

    def user_text(self, request_id: str, token_id: int) -> str:
        """The streamed user transcript this frame's user token completes."""
        if token_id == self.silence_token_id:
            return ""
        assert self.tokenizer is not None
        decoder = self.user_decoders.setdefault(request_id, decoders.DecodeStream(skip_special_tokens=True))
        return decoder.step(self.tokenizer, token_id) or ""


__all__ = ["DuplexIODataPlane"]
