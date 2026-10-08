"""Stream mono 24 kHz audio through a native DuplexIO realtime session."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import soundfile as sf
import websockets

SAMPLE_RATE = 24_000
FRAME_SIZE = 1_920


def read_audio(path: Path) -> np.ndarray:
    """Parse the session's mono 24 kHz audio at the client boundary."""
    samples, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE or samples.ndim != 1 or samples.size < FRAME_SIZE:
        raise ValueError(f"{path} must contain at least 80 ms of mono 24 kHz audio")
    return samples.astype("<f4", copy=False)


async def run(args: argparse.Namespace) -> dict:
    reference = read_audio(args.ref_audio)
    speech = read_audio(args.user_audio) if args.user_audio else np.empty(0, dtype="<f4")
    frames = max(int(args.seconds * SAMPLE_RATE / FRAME_SIZE), (speech.size + FRAME_SIZE - 1) // FRAME_SIZE)
    query = urlencode({"duplex": 1, "model": args.model, "autostart": 0})
    audio, text, user_text = [], [], []
    counts = Counter()
    started = time.monotonic()
    async with asyncio.timeout(frames * FRAME_SIZE / SAMPLE_RATE + 180):
        async with websockets.connect(
            f"{args.url.rstrip('/')}/v1/realtime?{query}", max_size=64 * 1024 * 1024
        ) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "model": args.model,
                            "modalities": ["audio", "text"],
                            "instructions": args.instructions,
                            "response_format": "pcm",
                            "tools": [],
                            "tool_choice": "none",
                            "extra_body": {
                                "full_duplex": True,
                                "auto_response": True,
                                "start_role": "user" if args.user_audio else "agent",
                                "ref_audio_data": base64.b64encode(reference.tobytes()).decode(),
                                "ref_audio_format": "pcm_f32le",
                                "ref_audio_sample_rate": SAMPLE_RATE,
                                "duplexio_sampling": {"seed": args.seed},
                            },
                        },
                    }
                )
            )
            while True:
                event = json.loads(await socket.recv())
                if event["type"] == "error":
                    raise RuntimeError(event["error"])
                if event["type"] == "session.updated":
                    break

            async def receive() -> None:
                async for raw in socket:
                    event = json.loads(raw)
                    kind = event["type"]
                    counts[kind] += 1
                    if kind == "error":
                        raise RuntimeError(event["error"])
                    if kind == "session.closed":
                        return
                    if kind == "response.audio.delta":
                        pcm = base64.b64decode(event["delta"])
                        if "f32" in event.get("format", ""):
                            samples = np.frombuffer(pcm, dtype="<f4")
                        else:
                            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                        audio.append(samples)
                        played_ms = event["metadata"]["audio_duration_ms"]
                        await socket.send(
                            json.dumps(
                                {
                                    "type": "playback.ack",
                                    "response_id": event["response_id"],
                                    "item_id": event["item_id"],
                                    "played_ms": played_ms,
                                    "committed_ms": played_ms,
                                }
                            )
                        )
                    elif kind == "response.audio_transcript.delta":
                        text.append(event["delta"])
                    elif kind == "conversation.item.input_audio_transcription.delta":
                        user_text.append(event["delta"])

            receiver = asyncio.create_task(receive())
            try:
                stream_started = time.monotonic()
                for index in range(frames):
                    if receiver.done():
                        await receiver
                        raise RuntimeError("Session closed before all input frames were sent")
                    frame = np.zeros(FRAME_SIZE, dtype="<f4")
                    part = speech[index * FRAME_SIZE : (index + 1) * FRAME_SIZE]
                    frame[: part.size] = part
                    await socket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(frame.tobytes()).decode(),
                                "format": "pcm_f32le",
                                "sample_rate_hz": SAMPLE_RATE,
                            }
                        )
                    )
                    await asyncio.sleep(
                        max(0, stream_started + (index + 1) * FRAME_SIZE / SAMPLE_RATE - time.monotonic())
                    )
                await asyncio.sleep(2)
                await socket.send(json.dumps({"type": "session.close"}))
                await receiver
            finally:
                receiver.cancel()
    if not audio:
        raise RuntimeError("Session produced no agent audio")
    waveform = np.concatenate(audio)
    if not np.isfinite(waveform).all():
        raise RuntimeError("Agent audio contains non-finite samples")
    sf.write(args.output, waveform, SAMPLE_RATE, subtype="PCM_16")
    return {
        "agent_text": "".join(text).strip(),
        "user_text": "".join(user_text).strip(),
        "audio_seconds": waveform.size / SAMPLE_RATE,
        "input_seconds": frames * FRAME_SIZE / SAMPLE_RATE,
        "elapsed_seconds": time.monotonic() - started,
        "events": dict(counts),
        "output": str(args.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="duplexio/duo-4b")
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--ref-audio", required=True, type=Path)
    parser.add_argument("--user-audio", type=Path)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--instructions", default="You are a helpful voice assistant.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("agent.wav"))
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
