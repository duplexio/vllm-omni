# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.duplexio.data_plane import (
    DuplexIODataPlaneContext,
    DuplexIODataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_data_plane_projects_one_native_audio_frame_and_text() -> None:
    encoded: list[tuple[object, int, str, float | None]] = []

    def encode_audio(audio, sample_rate, response_format, speed):
        encoded.append((audio, sample_rate, response_format, speed))
        return "encoded-audio"

    session = DuplexIODataPlaneSession(encode_audio)
    output = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="hello")],
        multimodal_output={
            "audio": np.zeros(1_920, dtype=np.float32),
            "sample_rate_hz": 24_000,
            "duplex_epoch": 3,
            "duplex_turn_id": 7,
            "user_token_id": 101,
            "agent_token_id": 102,
            "tool_call_token_id": 103,
        },
    )
    result = session.project_output(
        output,
        context=DuplexIODataPlaneContext(
            epoch=3,
            response_format="pcm",
            speed=1.0,
            modalities=("text", "audio"),
        ),
    )

    assert result is not None
    assert result["text"] == "hello"
    assert result["audio_data"] == "encoded-audio"
    assert result["audio_duration_ms"] == 80
    assert result["model_turn_id"] == 7
    assert result["user_token_id"] == 101
    assert result["agent_token_id"] == 102
    assert result["tool_call_token_id"] == 103
    assert encoded[0][1:] == (24_000, "pcm", 1.0)


def test_data_plane_drops_stale_epoch_output() -> None:
    session = DuplexIODataPlaneSession(lambda *_args: "audio")
    output = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="stale")],
        multimodal_output={"duplex_epoch": 1},
    )

    assert (
        session.project_output(
            output,
            context=DuplexIODataPlaneContext(epoch=2),
        )
        is None
    )


def test_data_plane_projects_explicit_model_listen() -> None:
    session = DuplexIODataPlaneSession(lambda *_args: None)
    output = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="")],
        multimodal_output={"model_listen": True, "duplex_turn_id": 4},
    )

    result = session.project_output(
        output,
        context=DuplexIODataPlaneContext(),
    )

    assert result is not None
    assert result["is_listen"] is True
    assert result["model_listen"] is True
    assert result["model_turn_id"] == 4


def test_data_plane_does_not_encode_initial_delayed_audio() -> None:
    encoded = []
    session = DuplexIODataPlaneSession(lambda *_args: encoded.append(True))
    output = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="")],
        multimodal_output={
            "audio": np.empty(0, dtype=np.float32),
            "model_listen": True,
        },
    )

    result = session.project_output(
        output,
        context=DuplexIODataPlaneContext(),
    )

    assert result is not None
    assert result["is_listen"] is True
    assert not encoded
