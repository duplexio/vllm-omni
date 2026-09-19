# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler data-plane policy for native DuplexIO frame appends."""

from __future__ import annotations

import base64
import binascii
import copy
from typing import Any

from vllm_omni.experimental.fullduplex.duplexio.input import (
    DUPLEXIO_FRAME_BYTES,
    DUPLEXIO_FRAME_SIZE,
    DUPLEXIO_SAMPLE_RATE,
)
from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

DUPLEXIO_ROW_CELL_COUNT = 6


def prefix_payload(prompt_frames: int, system_frames: int, *, decode_audio: bool) -> dict[str, Any]:
    """Submit the known prefix; the scheduler may split it at frame boundaries."""
    frames = prompt_frames + system_frames
    return {
        "type": "audio", "audio": "", "format": "pcm_f32le",
        "sample_rate_hz": DUPLEXIO_SAMPLE_RATE, "frame_size": DUPLEXIO_FRAME_SIZE,
        "frame_count": frames, "valid_samples": frames * DUPLEXIO_FRAME_SIZE,
        "duplexio_prefill": True, "decode_audio": decode_audio,
    }


def tool_result_payload(token_ids: list[int], *, decode_audio: bool) -> dict[str, Any]:
    """Submit one complete tool result for scheduler-owned chunking."""
    frames = len(token_ids)
    return {
        "type": "audio", "audio": "", "format": "pcm_f32le",
        "sample_rate_hz": DUPLEXIO_SAMPLE_RATE, "frame_size": DUPLEXIO_FRAME_SIZE,
        "frame_count": frames, "valid_samples": frames * DUPLEXIO_FRAME_SIZE,
        "duplexio_system_input": True, "duplexio_system_token_ids": token_ids,
        "decode_audio": decode_audio,
    }


def build_duplexio_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, Any],
    runtime_config: dict[str, Any],
    seq: int,
    turn_seq: int,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
) -> dict[str, Any]:
    """Build one resumable Stage-0 append using six slots per audio frame."""
    if mode is not DuplexInputMode.APPEND_AUDIO_CHUNK:
        raise ValueError(f"DuplexIO does not support input mode {mode.value!r}")
    if not isinstance(payload, dict):
        raise ValueError("DuplexIO append payload must be a dictionary")
    frame_count, pcm = parse_pcm_append(payload)
    payload = {key: value for key, value in payload.items() if key != "audio"}
    scheduler_token_budget = frame_count * DUPLEXIO_ROW_CELL_COUNT
    duplexio_prefill = payload.get("duplexio_prefill", False)
    duplexio_system_input = payload.get("duplexio_system_input", False)
    if duplexio_prefill:
        expected_frames = runtime_config["duplexio_voice_prompt_frames"] + len(
            runtime_config["duplexio_system_token_ids"]
        )
        if frame_count != expected_frames:
            raise ValueError(
                f"DuplexIO prefix has {frame_count} frames; expected the complete {expected_frames}-frame prefix"
            )
    duplexio_system_token_ids = payload.get("duplexio_system_token_ids")
    if duplexio_system_input and (
        not isinstance(duplexio_system_token_ids, list)
        or len(duplexio_system_token_ids) != frame_count
        or not all(isinstance(token_id, int) and token_id >= 0 for token_id in duplexio_system_token_ids)
    ):
        raise ValueError("DuplexIO system input requires one token ID per frame")
    scheduler_token_id = runtime_config.get("duplexio_scheduler_token_id", 0)
    if not isinstance(scheduler_token_id, int) or scheduler_token_id < 0:
        raise ValueError("duplexio_scheduler_token_id must be a non-negative integer")

    return {
        "prompt_token_ids": [scheduler_token_id] * scheduler_token_budget,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "data_plane": True,
                "runtime": "duplexio",
                "fence": fence,
                "session_id": fence.session_id,
                "incarnation": fence.incarnation,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": dict(payload),
                "pcm": pcm,
                "final": final,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "frame_count": frame_count,
                "row_cell_count": DUPLEXIO_ROW_CELL_COUNT,
                "scheduler_token_budget": scheduler_token_budget,
                "scheduler_token_id": scheduler_token_id,
                "duplexio_prefill": duplexio_prefill,
                "duplexio_system_input": duplexio_system_input,
                "duplexio_system_token_ids": duplexio_system_token_ids,
                "decode_audio": payload.get("decode_audio", True),
            },
        },
    }


class DuplexIORuntimeExtension:
    """Pure DuplexIO policy consumed by vLLM-Omni's duplex control plane."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        from vllm.sampling_params import RequestOutputKind

        configured: list[object] = []
        for default in defaults:
            clone = getattr(default, "clone", None)
            params = clone() if callable(clone) else copy.copy(default)
            if hasattr(params, "max_tokens"):
                setattr(params, "max_tokens", 1)
            # DELTA emits only this chunk's outputs, including when a context
            # append spans multiple scheduler steps.
            if hasattr(params, "output_kind"):
                setattr(params, "output_kind", RequestOutputKind.DELTA)
            configured.append(params)
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        return DuplexAppendPlan(
            prompt=build_duplexio_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                mode=mode,
                payload=payload,
                final=final,
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        # Single-stage DuplexIO segments already reach the client as the raw
        # stage-0 OutputMessage (the orchestrator only suppresses duplex
        # stage-0 segments when a downstream stage exists). Emitting a
        # direct-response decision as well would deliver two messages per
        # frame and desynchronize per-frame collection.
        del (
            stage_id,
            final_stage_id,
            segment_finished,
            segment_token_ids,
            segment_output_metadata,
            output,
        )
        return None


def parse_pcm_append(payload: dict[str, Any]) -> tuple[int, bytes | None]:
    """Validate wire framing and decode PCM once, before worker transport."""
    if payload.get("format") != "pcm_f32le":
        raise ValueError("DuplexIO data plane requires format='pcm_f32le'")
    if payload.get("sample_rate_hz") != DUPLEXIO_SAMPLE_RATE:
        raise ValueError("DuplexIO data plane requires 24000 Hz PCM")
    if payload.get("frame_size") != DUPLEXIO_FRAME_SIZE:
        raise ValueError(f"DuplexIO data plane requires frame_size={DUPLEXIO_FRAME_SIZE}")
    frame_count = payload.get("frame_count")
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("DuplexIO frame_count must be a positive integer")
    is_prefill = payload.get("duplexio_prefill", False)
    is_system_input = payload.get("duplexio_system_input", False)
    if not all(isinstance(flag, bool) for flag in (is_prefill, is_system_input)):
        raise ValueError("DuplexIO context-input flags must be boolean when present")
    if is_prefill and is_system_input:
        raise ValueError("DuplexIO prefill and system input are mutually exclusive")
    is_context_input = is_prefill or is_system_input
    if frame_count != 1 and not is_context_input:
        raise ValueError("DuplexIO live audio requires exactly one frame per append")
    audio = payload.get("audio")
    if not isinstance(audio, str):
        raise ValueError("DuplexIO data plane requires base64 audio")
    raw = None
    if is_context_input:
        if audio:
            raise ValueError("DuplexIO context audio is supplied by the model")
    else:
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("DuplexIO data-plane audio is not valid base64") from exc
        expected_bytes = frame_count * DUPLEXIO_FRAME_BYTES
        if len(raw) != expected_bytes:
            raise ValueError(
                "DuplexIO data-plane frame count does not match PCM bytes: "
                f"frame_count={frame_count}, bytes={len(raw)}, expected={expected_bytes}"
            )
    valid_samples = payload.get("valid_samples")
    if not isinstance(valid_samples, int) or not 1 <= valid_samples <= frame_count * DUPLEXIO_FRAME_SIZE:
        raise ValueError("DuplexIO valid_samples is outside the framed PCM payload")
    if is_context_input and valid_samples != frame_count * DUPLEXIO_FRAME_SIZE:
        raise ValueError("DuplexIO context input must contain complete frames")
    return frame_count, raw


__all__ = [
    "DUPLEXIO_ROW_CELL_COUNT",
    "DuplexIORuntimeExtension",
    "build_duplexio_data_plane_prompt",
]
