"""Relay a browser's Realtime WebSocket to the DuplexIO engine, optionally as Opus.

Both relays use this: the page's own proxy (server.py) and, on Modal, the
model container, which the page reaches directly so its audio skips a hop.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import json
import time
from urllib.parse import urlencode

import av
import numpy as np
import websockets
from starlette.websockets import WebSocket, WebSocketDisconnect

INPUT_SAMPLE_RATE = 24_000
# Opus packets carry the browser leg; one packet holds one 80 ms model frame.
OPUS_BITRATE = 48_000
OPUS_FRAME_MS = 80
# A cold engine holds the connection until it has booted (up to ~15 minutes on
# Modal) before the relay sees the ticket, so a ticket outlives that.
TICKET_LIFETIME_SECONDS = 20 * 60


def ticket_key(password_hash: str) -> bytes:
    """The key tickets are signed with: whoever holds the page's secret can issue them."""
    return hmac.digest(password_hash.encode(), b"duplexio realtime ticket v1", "sha256")


def issue_ticket(key: bytes, now: float | None = None) -> str:
    """A ticket that lets a signed-in page open sessions on an engine relay directly."""
    expiry = str(int((time.time() if now is None else now) + TICKET_LIFETIME_SECONDS))
    return f"{expiry}.{hmac.digest(key, expiry.encode(), 'sha256').hex()}"


def ticket_is_valid(key: bytes, ticket: str | None, now: float | None = None) -> bool:
    expiry, _, signature = (ticket or "").partition(".")
    if not expiry.isdigit():
        return False
    expected = hmac.digest(key, expiry.encode(), "sha256").hex()
    return hmac.compare_digest(signature, expected) and int(expiry) > (time.time() if now is None else now)


def backend_query(params: list[tuple[str, str]]) -> tuple[str, OpusTranscoder | None]:
    """The engine's query string, and a transcoder if the page asked for Opus."""
    opus = OpusTranscoder() if ("codec", "opus") in params else None
    return urlencode([(key, value) for key, value in params if key not in {"codec", "ticket"}]), opus


class OpusTranscoder:
    """Carries audio as Opus between the page and this proxy.

    Base64 PCM costs about 1.5 Mbps per session, so a page that asks for
    ``codec=opus`` sends and receives ``format: "opus"`` packets instead (about
    100 kbps with the base64). The backend still sees PCM: microphone packets
    are decoded into ``pcm_f32le`` appends and model audio deltas are encoded.
    """

    def __init__(self) -> None:
        self._decoder = av.CodecContext.create("opus", "r")
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=INPUT_SAMPLE_RATE)
        self._encoder: av.AudioCodecContext | None = None
        self._pending = np.zeros(0, np.int16)
        self._encoded_samples = 0

    def to_backend(self, message: str) -> str:
        event = json.loads(message)
        if event.get("type") != "input_audio_buffer.append" or event.get("format") != "opus":
            return message
        samples = [
            frame.to_ndarray().reshape(-1)
            for decoded in self._decoder.decode(av.Packet(base64.b64decode(event["audio"])))
            for frame in self._resampler.resample(decoded)
        ]
        pcm = np.concatenate(samples) if samples else np.zeros(0, np.float32)
        event.update(
            audio=base64.b64encode(pcm.astype("<f4").tobytes()).decode(),
            format="pcm_f32le",
            sample_rate_hz=INPUT_SAMPLE_RATE,
        )
        return json.dumps(event)

    def to_client(self, message: str) -> list[str]:
        event = json.loads(message)
        kind = event.get("type")
        if kind == "response.output_audio.delta":
            pcm = base64.b64decode(event["delta"])
            samples = (
                (np.frombuffer(pcm, "<f4").clip(-1, 1) * 32767).astype(np.int16)
                if "f32" in str(event.get("format", "")).lower()
                else np.frombuffer(pcm, "<i2")
            )
            self._pending = np.concatenate([self._pending, samples])
            return [self._opus_delta(event, packet) for packet in self._encode(event, pad=False)]
        if kind == "response.output_audio.done" and self._pending.size:
            # A partial frame is padded with silence rather than held back.
            return [self._opus_delta(event, packet) for packet in self._encode(event, pad=True)] + [message]
        return [message]

    def _encode(self, event: dict, *, pad: bool) -> list[bytes]:
        if self._encoder is None:
            self._encoder = av.CodecContext.create("libopus", "w")
            self._encoder.sample_rate = int(event.get("sample_rate_hz") or INPUT_SAMPLE_RATE)
            self._encoder.layout = "mono"
            self._encoder.format = "s16"
            self._encoder.bit_rate = OPUS_BITRATE
            self._encoder.options = {"frame_duration": str(OPUS_FRAME_MS)}
            self._encoder.open()
        frame_size = self._encoder.frame_size
        if pad:
            self._pending = np.pad(self._pending, (0, -self._pending.size % frame_size))
        packets = []
        while self._pending.size >= frame_size:
            frame = av.AudioFrame.from_ndarray(self._pending[None, :frame_size], format="s16", layout="mono")
            frame.sample_rate = self._encoder.sample_rate
            frame.pts = self._encoded_samples
            self._encoded_samples += frame_size
            self._pending = self._pending[frame_size:]
            packets += [bytes(packet) for packet in self._encoder.encode(frame)]
        return packets

    @staticmethod
    def _opus_delta(event: dict, packet: bytes) -> str:
        # Per-frame backend metadata (token ids, playback counters) would be
        # larger than the audio itself, and the page does not read it.
        delta = {
            key: value
            for key, value in event.items()
            if key not in {"format", "sample_rate_hz", "metadata"}
        }
        delta.update(type="response.output_audio.delta", delta=base64.b64encode(packet).decode(), format="opus")
        return json.dumps(delta)


async def pump_client_to_backend(client: WebSocket, backend, opus: OpusTranscoder | None = None) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                await backend.close()
                return
            if message.get("text") is not None:
                await backend.send(message["text"] if opus is None else opus.to_backend(message["text"]))
            elif message.get("bytes") is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def pump_backend_to_client(client: WebSocket, backend, opus: OpusTranscoder | None = None) -> None:
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                for text in [message] if opus is None else opus.to_client(message):
                    await client.send_text(text)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        return
    except RuntimeError as exc:
        if "websocket.send" in str(exc) and "websocket.close" in str(exc):
            return
        raise


def expected_proxy_close(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed, asyncio.CancelledError)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.send" in str(exc) and "websocket.close" in str(exc)


async def relay(client: WebSocket, backend, opus: OpusTranscoder | None) -> None:
    """Pump both directions until either side closes."""
    tasks = {
        asyncio.create_task(pump_client_to_backend(client, backend, opus)),
        asyncio.create_task(pump_backend_to_client(client, backend, opus)),
    }
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for result in await asyncio.gather(*done, return_exceptions=True):
        if isinstance(result, BaseException) and not expected_proxy_close(result):
            raise result
