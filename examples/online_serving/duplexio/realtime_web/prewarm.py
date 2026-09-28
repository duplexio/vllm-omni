"""Warm the DuplexIO prefix and decode paths before accepting browser sessions.

A prefix alone leaves the first frames of a real session slow (and, after a
Modal snapshot restore, the first ~10 s of them), so the warmup also streams
user audio through the model faster than real time.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from urllib.parse import urlencode

import websockets

SAMPLE_RATE = 24_000
FRAME_SIZE = 1_920
STREAM_SECONDS = 20
# Frames are sent four times faster than real time, which the model outpaces.
SEND_INTERVAL_SECONDS = FRAME_SIZE / SAMPLE_RATE / 4


def realtime_url(backend: str, model: str) -> str:
    query = urlencode({"duplex": 1, "model": model, "autostart": 0})
    return f"{backend.rstrip('/')}/v1/realtime?{query}"


def session_update(model: str, reference_audio: str, tools: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "session.update",
        "session": {
            "model": model,
            "modalities": ["audio", "text"],
            "response_format": "pcm",
            "tools": tools,
            "tool_choice": "auto" if tools else "none",
            "extra_body": {
                "full_duplex": True,
                "auto_response": True,
                "start_role": "agent",
                "ref_audio_data": reference_audio,
                "ref_audio_format": "pcm_f32le",
                "ref_audio_sample_rate": SAMPLE_RATE,
            },
        },
    }


async def wait_for_event(websocket, event_types: set[str]) -> dict[str, object]:
    while True:
        event = json.loads(await websocket.recv())
        event_type = event.get("type")
        if event_type == "error":
            raise RuntimeError(f"DuplexIO prewarm failed: {event.get('error')!r}")
        if event_type in event_types:
            return event


async def prewarm(
    backend: str,
    model: str,
    reference_audio: str,
    tools: list[dict[str, object]],
    frames: list[str],
    *,
    timeout_seconds: float,
) -> None:
    async with asyncio.timeout(timeout_seconds):
        async with websockets.connect(
            realtime_url(backend, model),
            max_size=64 * 1024 * 1024,
        ) as websocket:
            await websocket.send(json.dumps(session_update(model, reference_audio, tools)))
            await wait_for_event(websocket, {"session.updated"})

            async def send_frames() -> None:
                for frame in frames:
                    await websocket.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": frame,
                        "format": "pcm_f32le",
                        "sample_rate_hz": SAMPLE_RATE,
                    }))
                    await asyncio.sleep(SEND_INTERVAL_SECONDS)

            # The model answers every user frame with one audio frame.
            sender = asyncio.create_task(send_frames())
            for _ in frames:
                await wait_for_event(websocket, {"response.audio.delta"})
            await sender
            await websocket.send(json.dumps({"type": "session.close"}))
            await wait_for_event(websocket, {"session.closed"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ref-audio", type=Path, required=True, help="Mono 24 kHz reference audio")
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    import soundfile as sf

    samples, rate = sf.read(args.ref_audio, dtype="float32")
    if rate != SAMPLE_RATE or samples.ndim != 1 or samples.size < FRAME_SIZE:
        parser.error("--ref-audio must contain at least 80 ms of mono 24 kHz audio")
    samples = samples.astype("<f4", copy=False)
    reference_audio = base64.b64encode(samples.tobytes()).decode()
    # The user says the reference clip after a second of silence.
    import numpy as np

    stream = np.zeros(STREAM_SECONDS * SAMPLE_RATE, "<f4")
    speech = samples[: stream.size - SAMPLE_RATE]
    stream[SAMPLE_RATE : SAMPLE_RATE + speech.size] = speech
    frames = [
        base64.b64encode(stream[offset : offset + FRAME_SIZE].tobytes()).decode()
        for offset in range(0, stream.size, FRAME_SIZE)
    ]

    tools = json.loads(args.tools.read_text(encoding="utf-8"))
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        parser.error("--tools must contain a JSON list of tool definitions")
    asyncio.run(
        prewarm(
            args.backend,
            args.model,
            reference_audio,
            tools,
            frames,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print("DuplexIO realtime path is warm")


if __name__ == "__main__":
    main()
