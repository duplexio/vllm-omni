# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.duplexio.input import (
    DuplexIOPcmAppendBuffer,
)
from vllm_omni.experimental.fullduplex.duplexio.runtime import (
    DuplexIORuntimeExtension,
    build_duplexio_data_plane_prompt,
)
from vllm_omni.experimental.fullduplex.engine.contracts import DuplexInputMode
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_data_plane_prompt_reserves_six_cells_for_one_frame() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    payload = buffer.append(
        {
            "format": "pcm_f32le",
            "sample_rate_hz": 24_000,
            "audio": base64.b64encode(
                np.zeros(2 * 1_920, dtype="<f4").tobytes()
            ).decode("ascii"),
        },
        chunk_period_ms=80,
    )
    assert payload is not None
    prompt = build_duplexio_data_plane_prompt(
        request_id="request-1",
        fence=DuplexFence("session-1", epoch=2, turn_id=3),
        session_config={},
        runtime_config={"duplexio_scheduler_token_id": 17},
        seq=4,
        turn_seq=5,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=payload,
        final=False,
    )

    assert prompt["prompt_token_ids"] == [17] * 6
    duplex = prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["frame_count"] == 1
    assert duplex["row_cell_count"] == 6
    assert duplex["scheduler_token_budget"] == 6


def test_runtime_forces_one_scheduler_proxy_token() -> None:
    class Params(SimpleNamespace):
        def clone(self):
            return Params(**vars(self))

    default = Params(max_tokens=99, temperature=0.7)
    configured = DuplexIORuntimeExtension().configure_sampling_params(
        runtime_config={},
        defaults=(default,),
    )

    assert configured[0].max_tokens == 1
    assert configured[0].temperature == 0.7
    assert default.max_tokens == 99


def test_data_plane_prompt_rejects_declared_frame_mismatch() -> None:
    payload = {
        "format": "pcm_f32le",
        "sample_rate_hz": 24_000,
        "frame_size": 1_920,
        "frame_count": 2,
        "valid_samples": 1_920,
        "audio": base64.b64encode(
            np.zeros(1_920, dtype="<f4").tobytes()
        ).decode("ascii"),
    }
    with pytest.raises(ValueError, match="exactly one frame"):
        build_duplexio_data_plane_prompt(
            request_id="request-1",
            fence=DuplexFence("session-1"),
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
        )


def test_data_plane_prompt_rejects_multiple_complete_frames() -> None:
    payload = {
        "format": "pcm_f32le",
        "sample_rate_hz": 24_000,
        "frame_size": 1_920,
        "frame_count": 2,
        "valid_samples": 3_840,
        "audio": base64.b64encode(
            np.zeros(3_840, dtype="<f4").tobytes()
        ).decode("ascii"),
    }

    with pytest.raises(ValueError, match="exactly one frame"):
        build_duplexio_data_plane_prompt(
            request_id="request-1",
            fence=DuplexFence("session-1"),
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload,
            final=False,
        )
