# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
            "agent_audio_token_ids": [201, 202, 203],
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
    assert result["agent_audio_token_ids"] == [201, 202, 203]
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


def test_data_plane_drops_server_owned_prompt_prefill() -> None:
    encoded = []
    session = DuplexIODataPlaneSession(lambda *_args: encoded.append(True))
    output = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="should-not-leak")],
        multimodal_output={
            "audio": np.ones(1_920, dtype=np.float32),
            "sample_rate_hz": 24_000,
            "duplex_epoch": 0,
            "duplex_prefill": True,
            "model_listen": True,
        },
    )

    assert (
        session.project_output(
            output,
            context=DuplexIODataPlaneContext(),
        )
        is None
    )
    assert not encoded


def test_data_plane_marks_the_final_prompt_prefill_output() -> None:
    session = DuplexIODataPlaneSession(lambda *_args: None)
    output = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="")],
        multimodal_output={
            "duplex_prefill": False,
            "duplex_prefill_complete": True,
        },
    )

    result = session.project_output(
        output,
        context=DuplexIODataPlaneContext(),
    )

    assert result is not None
    assert result["initial_data_plane_complete"] is True


def test_data_plane_emits_only_new_samples_from_accumulated_audio() -> None:
    encoded: list[np.ndarray] = []

    def encode_audio(audio, *_args):
        encoded.append(np.asarray(audio).copy())
        return "encoded-audio"

    session = DuplexIODataPlaneSession(encode_audio)
    session.begin_request("request-1")
    context = DuplexIODataPlaneContext()
    first_frame = np.ones(1_920, dtype=np.float32)
    second_frame = np.full(1_920, 2, dtype=np.float32)

    first = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="")],
        multimodal_output={"audio": first_frame},
    )
    second = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="")],
        multimodal_output={
            "audio": np.concatenate((first_frame, second_frame)),
        },
    )

    assert session.project_output(first, context=context) is not None
    session.begin_request("request-1")
    assert session.project_output(second, context=context) is not None
    assert len(encoded) == 2
    np.testing.assert_array_equal(encoded[0], first_frame)
    np.testing.assert_array_equal(encoded[1], second_frame)


def test_data_plane_emits_agent_and_user_text_deltas() -> None:
    session = DuplexIODataPlaneSession(lambda *_args: None)
    decoded = {7: "you", 8: "you there"}
    session.configure_text_decoder(
        lambda token_ids: decoded[token_ids[-1]],
        silence_token_id=2,
    )
    session.begin_request("request-1")
    context = DuplexIODataPlaneContext(modalities=("text",))

    first = session.project_output(
        SimpleNamespace(
            request_id="request-1",
            outputs=[SimpleNamespace(text="Hel")],
            multimodal_output={"user_token_id": 7},
        ),
        context=context,
    )
    second = session.project_output(
        SimpleNamespace(
            request_id="request-1",
            outputs=[SimpleNamespace(text="Hello")],
            multimodal_output={"user_token_id": 8},
        ),
        context=context,
    )

    assert first is not None
    assert first["text"] == "Hel"
    assert first["input_text_delta"] == "you"
    assert second is not None
    assert second["text"] == "lo"
    assert second["input_text_delta"] == " there"


def test_data_plane_ignores_user_silence_tokens() -> None:
    session = DuplexIODataPlaneSession(lambda *_args: None)
    session.configure_text_decoder(
        lambda _token_ids: "must not decode",
        silence_token_id=2,
    )

    result = session.project_output(
        SimpleNamespace(
            request_id="request-1",
            outputs=[SimpleNamespace(text="")],
            multimodal_output={"user_token_id": 2, "model_listen": True},
        ),
        context=DuplexIODataPlaneContext(),
    )

    assert result is not None
    assert "input_text_delta" not in result


def test_data_plane_forwards_sampler_materialized_tool_call() -> None:
    session = DuplexIODataPlaneSession(lambda *_args: None)
    tool_call = {
        "name": "get_current_time",
        "arguments": {"timezone": "Europe/Copenhagen"},
    }
    payload = torch.tensor(
        list(json.dumps({"sequence": 1, **tool_call}).encode("utf-8")),
        dtype=torch.uint8,
    )

    result = session.project_output(
        SimpleNamespace(
            request_id="request-1",
            outputs=[SimpleNamespace(text="")],
            multimodal_output={"tool_call_json": [payload]},
        ),
        context=DuplexIODataPlaneContext(),
    )

    assert result is not None
    assert result["tool_call"] == tool_call


def test_data_plane_deduplicates_replayed_calls_by_model_sequence() -> None:
    session = DuplexIODataPlaneSession(lambda *_args: None)
    tool_call = {
        "name": "get_current_time",
        "arguments": {"timezone": "Europe/Copenhagen"},
    }
    payload = torch.tensor(
        list(json.dumps({"sequence": 1, **tool_call}).encode("utf-8")),
        dtype=torch.uint8,
    )
    output = SimpleNamespace(
        request_id="request-1",
        outputs=[SimpleNamespace(text="")],
        multimodal_output={"tool_call_json": [payload]},
    )
    context = DuplexIODataPlaneContext()

    first = session.project_output(output, context=context)
    duplicate = session.project_output(output, context=context)
    next_payload = torch.tensor(
        list(
            (
                json.dumps({"sequence": 1, **tool_call})
                + json.dumps({"sequence": 2, **tool_call})
            ).encode("utf-8")
        ),
        dtype=torch.uint8,
    )
    next_call = session.project_output(
        SimpleNamespace(
            request_id="request-1",
            outputs=[SimpleNamespace(text="")],
            multimodal_output={"tool_call_json": [next_payload]},
        ),
        context=context,
    )

    assert first is not None
    assert first["tool_call"] == tool_call
    assert duplicate is None
    assert next_call is not None
    assert next_call["tool_call"] == tool_call
