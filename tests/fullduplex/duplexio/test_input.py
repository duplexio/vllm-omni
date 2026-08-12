# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.duplexio.input import (
    DUPLEXIO_FRAME_SIZE,
    DUPLEXIO_SAMPLE_RATE,
    DuplexIOPcmAppendBuffer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(samples: np.ndarray, *, is_speech: bool = False) -> dict[str, object]:
    return {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": DUPLEXIO_SAMPLE_RATE,
        "audio": base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii"),
        "is_speech": is_speech,
    }


def _decoded(payload: dict[str, object]) -> np.ndarray:
    audio = payload["audio"]
    assert isinstance(audio, str)
    return np.frombuffer(base64.b64decode(audio), dtype="<f4")


def test_pcm_buffer_frames_across_client_chunk_boundaries() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    first = np.arange(400, dtype=np.float32)
    second = np.arange(400, DUPLEXIO_FRAME_SIZE, dtype=np.float32)

    assert buffer.append(_payload(first), chunk_period_ms=10) is None
    framed = buffer.append(_payload(second, is_speech=True), chunk_period_ms=70)

    assert framed is not None
    assert framed["frame_count"] == 1
    assert framed["frame_size"] == DUPLEXIO_FRAME_SIZE
    assert framed["valid_samples"] == DUPLEXIO_FRAME_SIZE
    assert framed["is_speech"] is True
    np.testing.assert_array_equal(_decoded(framed), np.concatenate((first, second)))
    assert not buffer.has_pending()


def test_pcm_buffer_reserves_complete_rows_one_at_a_time() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    samples = np.arange(2 * DUPLEXIO_FRAME_SIZE + 19, dtype=np.float32)

    first = buffer.append(_payload(samples), chunk_period_ms=1_000)

    assert first is not None
    assert first["frame_count"] == 1
    assert first["valid_samples"] == DUPLEXIO_FRAME_SIZE
    np.testing.assert_array_equal(
        _decoded(first),
        samples[:DUPLEXIO_FRAME_SIZE],
    )

    reservation = buffer.prepare_buffered_append(
        operation_id="drain-2",
        chunk_period_ms=80,
    )
    assert reservation is not None
    assert reservation.payload is not None
    assert reservation.payload["frame_count"] == 1
    assert reservation.payload["valid_samples"] == DUPLEXIO_FRAME_SIZE
    np.testing.assert_array_equal(
        _decoded(reservation.payload),
        samples[DUPLEXIO_FRAME_SIZE : 2 * DUPLEXIO_FRAME_SIZE],
    )
    reservation.commit()
    assert buffer.pending_byte_count == 19 * 4


def test_pcm_commit_zero_pads_only_the_real_residual() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    samples = np.arange(123, dtype=np.float32)
    assert buffer.append(_payload(samples), chunk_period_ms=80) is None

    framed = buffer.commit(chunk_period_ms=80)

    assert framed is not None
    assert framed["final"] is True
    assert framed["frame_count"] == 1
    assert framed["valid_samples"] == len(samples)
    decoded = _decoded(framed)
    np.testing.assert_array_equal(decoded[: len(samples)], samples)
    np.testing.assert_array_equal(
        decoded[len(samples) :],
        np.zeros(DUPLEXIO_FRAME_SIZE - len(samples), dtype=np.float32),
    )


def test_pcm_commit_requires_complete_frames_to_be_drained_first() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    buffer.append(
        _payload(np.zeros(2 * DUPLEXIO_FRAME_SIZE, dtype=np.float32)),
        chunk_period_ms=80,
    )

    with pytest.raises(RuntimeError, match="acknowledged"):
        buffer.commit(chunk_period_ms=80)


def test_pcm_flush_requires_complete_frames_to_be_drained_first() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    buffer.prepare_append(
        _payload(np.zeros(2 * DUPLEXIO_FRAME_SIZE, dtype=np.float32)),
        operation_id="first",
        chunk_period_ms=80,
        allow_emit=False,
    )

    with pytest.raises(RuntimeError, match="acknowledged"):
        buffer.flush(chunk_period_ms=80)


def test_pcm_reservation_rollback_restores_exact_bytes() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    samples = np.arange(DUPLEXIO_FRAME_SIZE, dtype=np.float32)
    reservation = buffer.prepare_append(
        _payload(samples, is_speech=True),
        operation_id="append-1",
        chunk_period_ms=80,
        allow_emit=True,
    )
    assert reservation is not None
    assert buffer.has_reserved()

    reservation.rollback()

    assert not buffer.has_reserved()
    assert buffer.pending_byte_count == DUPLEXIO_FRAME_SIZE * 4
    replay = buffer.append(_payload(np.empty(0, dtype=np.float32)), chunk_period_ms=80)
    assert replay is not None
    assert replay["is_speech"] is True
    np.testing.assert_array_equal(_decoded(replay), samples)


def test_pcm_reservations_commit_in_wire_order() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    first = buffer.prepare_append(
        _payload(np.zeros(DUPLEXIO_FRAME_SIZE, dtype=np.float32)),
        operation_id="first",
        chunk_period_ms=80,
        allow_emit=True,
    )
    second = buffer.prepare_append(
        _payload(np.ones(DUPLEXIO_FRAME_SIZE, dtype=np.float32)),
        operation_id="second",
        chunk_period_ms=80,
        allow_emit=True,
    )
    assert first is not None and second is not None

    with pytest.raises(RuntimeError, match="wire order"):
        second.commit()

    first.commit()
    second.commit()
    assert not buffer.has_reserved()


def test_buffered_drain_waits_for_earlier_wire_reservation() -> None:
    buffer = DuplexIOPcmAppendBuffer()
    samples = np.arange(3 * DUPLEXIO_FRAME_SIZE, dtype=np.float32)
    first = buffer.prepare_append(
        _payload(samples[: 2 * DUPLEXIO_FRAME_SIZE]),
        operation_id="first",
        chunk_period_ms=80,
        allow_emit=True,
    )
    second = buffer.prepare_append(
        _payload(samples[2 * DUPLEXIO_FRAME_SIZE :]),
        operation_id="second",
        chunk_period_ms=80,
        allow_emit=True,
    )
    assert first is not None and second is not None

    first.commit()

    assert (
        buffer.prepare_buffered_append(
            operation_id="third",
            chunk_period_ms=80,
        )
        is None
    )
    second.commit()
    third = buffer.prepare_buffered_append(
        operation_id="third",
        chunk_period_ms=80,
    )
    assert third is not None and third.payload is not None
    np.testing.assert_array_equal(
        _decoded(third.payload),
        samples[2 * DUPLEXIO_FRAME_SIZE :],
    )
    third.commit()
    assert not buffer.has_reserved()


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"format": "wav", "sample_rate_hz": 24_000, "audio": ""}, "format"),
        ({"format": "pcm_f32le", "sample_rate_hz": 16_000, "audio": ""}, "24000"),
        ({"format": "pcm_f32le", "sample_rate_hz": 24_000, "audio": "!"}, "base64"),
    ],
)
def test_pcm_buffer_rejects_non_native_audio(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DuplexIOPcmAppendBuffer().append(payload, chunk_period_ms=80)
