"""Warm the complete DuplexIO realtime path before accepting browser sessions."""

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
MODEL_OUTPUT_EVENTS = {
    "input.transcript.delta",
    "response.audio.delta",
    "response.audio_transcript.delta",
    "response.listen",
}


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


def silence_frame() -> dict[str, object]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(bytes(FRAME_SIZE * 4)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": SAMPLE_RATE,
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
            await websocket.send(json.dumps(silence_frame()))
            await wait_for_event(websocket, MODEL_OUTPUT_EVENTS)
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
    reference_audio = base64.b64encode(samples.astype("<f4", copy=False).tobytes()).decode()

    tools = json.loads(args.tools.read_text(encoding="utf-8"))
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        parser.error("--tools must contain a JSON list of tool definitions")
    asyncio.run(
        prewarm(
            args.backend,
            args.model,
            reference_audio,
            tools,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print("DuplexIO realtime path is warm")


if __name__ == "__main__":
    main()
