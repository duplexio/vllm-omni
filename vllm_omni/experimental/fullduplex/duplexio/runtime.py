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


def duplexio_scheduler_token_budget(payload: object) -> int:
    """Return the exact flattened-cell count represented by one PCM append."""
    return _validated_frame_count(payload) * DUPLEXIO_ROW_CELL_COUNT


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
    frame_count = _validated_frame_count(payload)
    scheduler_token_budget = frame_count * DUPLEXIO_ROW_CELL_COUNT
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
                "final": final,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "frame_count": frame_count,
                "row_cell_count": DUPLEXIO_ROW_CELL_COUNT,
                "scheduler_token_budget": scheduler_token_budget,
                "scheduler_token_id": scheduler_token_id,
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
        configured: list[object] = []
        for default in defaults:
            clone = getattr(default, "clone", None)
            params = clone() if callable(clone) else copy.copy(default)
            if hasattr(params, "max_tokens"):
                setattr(params, "max_tokens", 1)
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
        del (
            stage_id,
            final_stage_id,
            segment_finished,
            segment_token_ids,
            segment_output_metadata,
            output,
        )
        return None


def _validated_frame_count(payload: object) -> int:
    if not isinstance(payload, dict):
        raise ValueError("DuplexIO append payload must be a dictionary")
    if payload.get("format") != "pcm_f32le":
        raise ValueError("DuplexIO data plane requires format='pcm_f32le'")
    if payload.get("sample_rate_hz") != DUPLEXIO_SAMPLE_RATE:
        raise ValueError("DuplexIO data plane requires 24000 Hz PCM")
    if payload.get("frame_size") != DUPLEXIO_FRAME_SIZE:
        raise ValueError(
            f"DuplexIO data plane requires frame_size={DUPLEXIO_FRAME_SIZE}"
        )
    frame_count = payload.get("frame_count")
    if not isinstance(frame_count, int) or frame_count != 1:
        raise ValueError("DuplexIO data plane requires exactly one frame per append")
    audio = payload.get("audio")
    if not isinstance(audio, str):
        raise ValueError("DuplexIO data plane requires base64 audio")
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
    if (
        not isinstance(valid_samples, int)
        or not 1 <= valid_samples <= DUPLEXIO_FRAME_SIZE
    ):
        raise ValueError("DuplexIO valid_samples is outside the framed PCM payload")
    return frame_count


__all__ = [
    "DUPLEXIO_ROW_CELL_COUNT",
    "DuplexIORuntimeExtension",
    "build_duplexio_data_plane_prompt",
    "duplexio_scheduler_token_budget",
]
