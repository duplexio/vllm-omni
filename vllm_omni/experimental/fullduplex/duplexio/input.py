# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Transactional PCM framing for the native DuplexIO data plane."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

DUPLEXIO_SAMPLE_RATE = 24_000
DUPLEXIO_FRAME_SIZE = 1_920
DUPLEXIO_FRAME_BYTES = DUPLEXIO_FRAME_SIZE * 4


@dataclass(frozen=True, slots=True)
class _PcmSpan:
    byte_count: int
    force_listen: bool
    is_speech: bool


class DuplexIOPcmAppendReservation:
    """One wire-ordered PCM reservation awaiting engine acknowledgement."""

    __slots__ = (
        "_active",
        "_owner",
        "_raw",
        "_sample_rate_hz",
        "_spans",
        "_turn_had_speech",
        "operation_id",
        "payload",
    )

    def __init__(
        self,
        *,
        owner: DuplexIOPcmAppendBuffer,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
        sample_rate_hz: int,
        spans: list[_PcmSpan],
        turn_had_speech: bool,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self._raw = raw
        self._sample_rate_hz = sample_rate_hz
        self._spans = spans
        self._turn_had_speech = turn_had_speech
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self._raw)

    def commit(self) -> None:
        self._owner._commit_reservation(self)

    def rollback(self) -> None:
        self._owner._rollback_reservation(self)


class DuplexIOPcmAppendBuffer:
    """Frame arbitrary client chunks into exact 80 ms DuplexIO appends.

    The vLLM-Omni duplex serving layer may receive PCM at any chunk boundary,
    while the model consumes only complete 1,920-sample rows. Bytes leave the
    buffer through a reservation and become permanent only after the engine
    acknowledges the matching append.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._spans: list[_PcmSpan] = []
        self._sample_rate_hz: int | None = None
        self._turn_had_speech = False
        self._reservation_seq = 0
        self._reservations: list[DuplexIOPcmAppendReservation] = []

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return bool(self._reservations)

    def clear(self) -> None:
        for reservation in self._reservations:
            reservation._active = False
        self._reservations.clear()
        self._buffer.clear()
        self._spans.clear()
        self._sample_rate_hz = None
        self._turn_had_speech = False

    def clear_force_listen(self) -> None:
        self._spans = [
            _PcmSpan(span.byte_count, False, span.is_speech)
            for span in self._spans
        ]

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> DuplexIOPcmAppendReservation | None:
        """Append client PCM and reserve the next complete model frame."""
        del chunk_period_ms  # DuplexIO's checkpoint owns the fixed 80 ms cadence.
        raw, sample_rate_hz = _decode_pcm_payload(payload)
        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError(
                "DuplexIO input sample rate changed within a session: "
                f"{self._sample_rate_hz} -> {sample_rate_hz}"
            )
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        self._append_span(
            _PcmSpan(
                len(raw),
                bool(payload.get("force_listen", False)),
                bool(payload.get("is_speech", False)),
            )
        )
        self._turn_had_speech |= bool(payload.get("is_speech", False))
        if not allow_emit:
            return None

        if len(self._buffer) < DUPLEXIO_FRAME_BYTES:
            return None
        return self._reserve(
            payload,
            operation_id=operation_id,
            emit_bytes=DUPLEXIO_FRAME_BYTES,
            pad_bytes=0,
            final=False,
        )

    def prepare_buffered_append(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> DuplexIOPcmAppendReservation | None:
        """Reserve one complete frame when no wire reservation is ahead of it."""
        del chunk_period_ms
        if self._reservations:
            return None
        if len(self._buffer) < DUPLEXIO_FRAME_BYTES:
            return None
        sample_rate_hz = self._sample_rate_hz or DUPLEXIO_SAMPLE_RATE
        return self._reserve(
            {
                "type": "audio",
                "format": "pcm_f32le",
                "sample_rate_hz": sample_rate_hz,
                "audio": "",
            },
            operation_id=operation_id,
            emit_bytes=DUPLEXIO_FRAME_BYTES,
            pad_bytes=0,
            final=False,
        )

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> DuplexIOPcmAppendReservation:
        """Reserve a zero-padded final frame, if a real residual exists."""
        del chunk_period_ms
        sample_rate_hz = self._sample_rate_hz or DUPLEXIO_SAMPLE_RATE
        if len(self._buffer) >= DUPLEXIO_FRAME_BYTES:
            raise RuntimeError(
                "DuplexIO complete frames must be acknowledged before turn commit"
            )
        if self._buffer:
            payload: dict[str, object] = {
                "type": "audio",
                "format": "pcm_f32le",
                "sample_rate_hz": sample_rate_hz,
                "audio": "",
            }
            reservation = self._reserve(
                payload,
                operation_id=operation_id,
                emit_bytes=len(self._buffer),
                pad_bytes=(-len(self._buffer)) % DUPLEXIO_FRAME_BYTES,
                final=True,
            )
        else:
            reservation = DuplexIOPcmAppendReservation(
                owner=self,
                operation_id=operation_id,
                payload=None,
                raw=b"",
                sample_rate_hz=sample_rate_hz,
                spans=[],
                turn_had_speech=self._turn_had_speech,
            )
            self._reservations.append(reservation)

        self._sample_rate_hz = None
        self._turn_had_speech = False
        return reservation

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        """Immediately commit a padded residual without ending the client turn."""
        del chunk_period_ms
        if not self._buffer:
            return None
        if len(self._buffer) >= DUPLEXIO_FRAME_BYTES:
            raise RuntimeError(
                "DuplexIO complete frames must be acknowledged before flush"
            )
        self._reservation_seq += 1
        sample_rate_hz = self._sample_rate_hz or DUPLEXIO_SAMPLE_RATE
        reservation = self._reserve(
            {
                "type": "audio",
                "format": "pcm_f32le",
                "sample_rate_hz": sample_rate_hz,
                "audio": "",
            },
            operation_id=f"immediate-flush-{self._reservation_seq}",
            emit_bytes=len(self._buffer),
            pad_bytes=(-len(self._buffer)) % DUPLEXIO_FRAME_BYTES,
            final=False,
        )
        reservation.commit()
        return reservation.payload

    def append(
        self,
        payload: dict[str, object],
        *,
        chunk_period_ms: int,
        allow_emit: bool = True,
    ) -> dict[str, object] | None:
        """Convenience path for non-transactional callers and focused tests."""
        self._reservation_seq += 1
        reservation = self.prepare_append(
            payload,
            operation_id=f"immediate-{self._reservation_seq}",
            chunk_period_ms=chunk_period_ms,
            allow_emit=allow_emit,
        )
        if reservation is None:
            return None
        reservation.commit()
        return reservation.payload

    def commit(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        self._reservation_seq += 1
        reservation = self.prepare_commit(
            operation_id=f"immediate-commit-{self._reservation_seq}",
            chunk_period_ms=chunk_period_ms,
        )
        reservation.commit()
        return reservation.payload

    def _reserve(
        self,
        source_payload: dict[str, object],
        *,
        operation_id: str,
        emit_bytes: int,
        pad_bytes: int,
        final: bool,
    ) -> DuplexIOPcmAppendReservation:
        raw = bytes(self._buffer[:emit_bytes])
        spans = self._consume_spans(emit_bytes)
        del self._buffer[:emit_bytes]
        encoded = raw + b"\x00" * pad_bytes
        frame_count, remainder = divmod(len(encoded), DUPLEXIO_FRAME_BYTES)
        assert frame_count > 0 and remainder == 0

        payload = dict(source_payload)
        payload["audio"] = base64.b64encode(encoded).decode("ascii")
        payload["format"] = "pcm_f32le"
        payload["sample_rate_hz"] = DUPLEXIO_SAMPLE_RATE
        payload["frame_size"] = DUPLEXIO_FRAME_SIZE
        payload["frame_count"] = frame_count
        payload["valid_samples"] = emit_bytes // 4
        payload["force_listen"] = any(span.force_listen for span in spans)
        payload["is_speech"] = any(span.is_speech for span in spans)
        if final:
            payload["final"] = True

        reservation = DuplexIOPcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=payload,
            raw=raw,
            sample_rate_hz=DUPLEXIO_SAMPLE_RATE,
            spans=spans,
            turn_had_speech=self._turn_had_speech,
        )
        self._reservations.append(reservation)
        return reservation

    def _append_span(self, span: _PcmSpan) -> None:
        if span.byte_count == 0:
            return
        if (
            self._spans
            and self._spans[-1].force_listen == span.force_listen
            and self._spans[-1].is_speech == span.is_speech
        ):
            previous = self._spans[-1]
            self._spans[-1] = _PcmSpan(
                previous.byte_count + span.byte_count,
                span.force_listen,
                span.is_speech,
            )
            return
        self._spans.append(span)

    def _consume_spans(self, byte_count: int) -> list[_PcmSpan]:
        consumed: list[_PcmSpan] = []
        remaining = byte_count
        while remaining:
            if not self._spans:
                raise RuntimeError("DuplexIO PCM metadata is shorter than buffered audio")
            span = self._spans.pop(0)
            take = min(remaining, span.byte_count)
            consumed.append(_PcmSpan(take, span.force_listen, span.is_speech))
            if take < span.byte_count:
                self._spans.insert(
                    0,
                    _PcmSpan(
                        span.byte_count - take,
                        span.force_listen,
                        span.is_speech,
                    ),
                )
            remaining -= take
        return consumed

    def _prepend_spans(self, spans: list[_PcmSpan]) -> None:
        current = self._spans
        self._spans = []
        for span in [*spans, *current]:
            self._append_span(span)

    def _commit_reservation(self, reservation: DuplexIOPcmAppendReservation) -> None:
        if not reservation._active:
            return
        if not self._reservations or self._reservations[0] is not reservation:
            raise RuntimeError("DuplexIO PCM reservations must commit in wire order")
        self._reservations.pop(0)
        reservation._active = False

    def _rollback_reservation(self, reservation: DuplexIOPcmAppendReservation) -> None:
        if not reservation._active:
            return
        try:
            index = self._reservations.index(reservation)
        except ValueError:
            reservation._active = False
            return
        rolled_back = self._reservations[index:]
        self._buffer[:0] = b"".join(item._raw for item in rolled_back)
        self._prepend_spans([span for item in rolled_back for span in item._spans])
        self._sample_rate_hz = self._sample_rate_hz or reservation._sample_rate_hz
        self._turn_had_speech |= any(item._turn_had_speech for item in rolled_back)
        for item in rolled_back:
            item._active = False
        del self._reservations[index:]


def _decode_pcm_payload(payload: dict[str, object]) -> tuple[bytes, int]:
    """Validate the external native-audio boundary and return raw PCM bytes."""
    if payload.get("format") != "pcm_f32le":
        raise ValueError("DuplexIO native input requires format='pcm_f32le'")
    sample_rate_hz = payload.get("sample_rate_hz")
    if sample_rate_hz != DUPLEXIO_SAMPLE_RATE:
        raise ValueError(
            "DuplexIO native input requires 24000 Hz PCM, got "
            f"{sample_rate_hz!r}"
        )
    audio = payload.get("audio")
    if not isinstance(audio, str):
        raise ValueError("DuplexIO native input requires base64 audio")
    try:
        raw = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("DuplexIO native input audio is not valid base64") from exc
    if len(raw) % 4:
        raise ValueError("DuplexIO pcm_f32le byte length must be divisible by four")
    return raw, DUPLEXIO_SAMPLE_RATE


__all__ = [
    "DUPLEXIO_FRAME_BYTES",
    "DUPLEXIO_FRAME_SIZE",
    "DUPLEXIO_SAMPLE_RATE",
    "DuplexIOPcmAppendBuffer",
    "DuplexIOPcmAppendReservation",
]
