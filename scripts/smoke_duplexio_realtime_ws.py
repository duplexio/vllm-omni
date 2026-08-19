# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke: the FULL realtime websocket path, judged by independent ASR.

Boots the actual `vllm-omni serve` realtime endpoint, opens a browser-shaped
session (f32 24 kHz input frames, ``response_format: "pcm"``, agent-first),
streams silence so the model narrates, and reassembles the emitted
``response.audio.delta`` chunks EXACTLY the way the web client does
(format/rate fields per event, int16 default). The reassembled waveform is
then transcribed with offline Whisper (large-v3-turbo) and must fuzzily match
the session's own ``response.audio_transcript.delta`` text. A duration check
catches duplicated/overlapping chunks that a text-alignment gate cannot see.

Unlike smoke_duplexio_v3.py (which drives the duplex control plane
in-process), this exercises websocket serialization and the wire audio
encoding — everything a browser receives.

Invocation (1 GPU node):

  cd /dcai/users/thuand/vllm-omni-work && \
  PYTHONPATH=/dcai/users/thuand/vllm-omni-work \
  .venv/bin/python scripts/smoke_duplexio_realtime_ws.py \
    --model /dcai/users/thuand/duplexio/checkpoints/duplexio-489780-checkpoint-1-vllm-v3
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

FRAME_SIZE = 1_920
SAMPLE_RATE = 24_000


def fail(message: str) -> None:
    print(f"SMOKE FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def normalized_words(text: str) -> list[str]:
    cleaned = "".join(
        c if c.isalnum() or c.isspace() else " " for c in text.lower()
    )
    return cleaned.split()


def browser_decode(chunks: list[dict]) -> tuple[np.ndarray, int]:
    """Reassemble audio deltas exactly like the web client's app.js."""
    pcm_parts: list[np.ndarray] = []
    rate = SAMPLE_RATE
    for chunk in chunks:
        data = base64.b64decode(chunk["delta"])
        fmt = str(chunk.get("format", "")).lower()
        rate = int(chunk.get("sample_rate_hz") or rate)
        if "f32" in fmt:
            samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
        else:
            samples = (
                np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            )
        pcm_parts.append(samples)
    if not pcm_parts:
        return np.zeros(0, dtype=np.float32), rate
    return np.concatenate(pcm_parts), rate


async def run_client(args: argparse.Namespace) -> None:
    import websockets

    query = urlencode({"duplex": 1, "model": args.model, "autostart": 0})
    base = args.url.rstrip("/") if args.url else f"ws://127.0.0.1:{args.port}"
    url = f"{base}/v1/realtime?{query}"
    print(f"[ws] connecting: {url}")
    connect_started = time.monotonic()
    audio_chunks: list[dict] = []
    transcript_parts: list[str] = []
    listen_seen = asyncio.Event()
    done = asyncio.Event()

    async with websockets.connect(
        url,
        max_size=64 * 1024 * 1024,
        open_timeout=args.connect_timeout,
    ) as ws:
        print(f"[ws] connected after {time.monotonic() - connect_started:.1f}s "
              "(includes any cold start)")
        async def reader() -> None:
            try:
                while True:
                    event = json.loads(await ws.recv())
                    event_type = event.get("type")
                    if event_type == "error":
                        print(f"[ws] error event: {event}", file=sys.stderr)
                    elif event_type == "response.audio.delta":
                        audio_chunks.append(event)
                    elif event_type == "response.audio_transcript.delta":
                        transcript_parts.append(event.get("delta") or "")
                    elif event_type == "response.listen":
                        listen_seen.set()
            except websockets.ConnectionClosed:
                done.set()

        reader_task = asyncio.create_task(reader())
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "model": args.model,
                "modalities": ["audio", "text"],
                "voice": args.voice,
                "response_format": "pcm",
                "tools": [],
                "tool_choice": "none",
                "extra_body": {
                    "full_duplex": True,
                    "auto_response": True,
                    "start_role": "agent",
                },
            },
        }))
        # Stream silent f32 frames at the browser cadence (80 ms).
        silent = base64.b64encode(bytes(FRAME_SIZE * 4)).decode()
        frames = int(args.seconds / 0.08)
        start = time.monotonic()
        for index in range(frames):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": silent,
                "format": "pcm_f32le",
                "sample_rate_hz": SAMPLE_RATE,
            }))
            target = start + (index + 1) * 0.08
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        await asyncio.sleep(2.0)
        await ws.send(json.dumps({"type": "session.close"}))
        try:
            await asyncio.wait_for(done.wait(), timeout=10.0)
        except TimeoutError:
            pass
        reader_task.cancel()

    transcript = "".join(transcript_parts).strip()
    print(f"[ws] audio.delta chunks: {len(audio_chunks)}")
    print(f"[ws] transcript deltas: {transcript!r}")
    if not audio_chunks:
        fail("no response.audio.delta events received")
    if not transcript:
        fail("no response.audio_transcript.delta text received")

    waveform, rate = browser_decode(audio_chunks)
    duration = len(waveform) / rate
    print(f"[ws] reassembled audio: {len(waveform)} samples @ {rate} Hz "
          f"= {duration:.2f}s over {len(audio_chunks)} chunks")
    # Duplicated/overlapping chunks would inflate duration well beyond the
    # session length; the session streams args.seconds of frames total.
    if duration > args.seconds + 3.0:
        fail(
            f"reassembled audio ({duration:.1f}s) exceeds the session span "
            f"({args.seconds:.1f}s): duplicated or overlapping chunks"
        )
    if not np.isfinite(waveform).all():
        fail("reassembled audio contains non-finite samples")

    import wave

    wav_path = Path(args.out_dir) / "realtime_ws_agent.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
        )
    print(f"[ws] wav saved: {wav_path}")

    from difflib import SequenceMatcher

    import whisper
    from scipy.signal import resample_poly

    asr = whisper.load_model("large-v3-turbo")
    audio_16k = resample_poly(waveform.astype(np.float64), 16_000, rate)
    heard = asr.transcribe(
        audio_16k.astype(np.float32),
        language="en",
        fp16=True,
    )["text"].strip()
    print(f"[ws] whisper heard: {heard!r}")
    ratio = SequenceMatcher(
        None,
        normalized_words(transcript),
        normalized_words(heard),
    ).ratio()
    print(f"[ws] whisper/transcript match ratio: {ratio:.2f}")
    if ratio < 0.6:
        fail(
            f"browser-side audio does not transcribe to the agent text "
            f"(ratio {ratio:.2f} < 0.60)"
        )
    print("[ws] browser-side audio is intelligible and matches the text")
    print("SMOKE PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--url",
        default=None,
        help="Remote realtime endpoint base (e.g. "
        "wss://duplexio--model-snapshot-staging.modal.run); skips booting a "
        "local server",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=900.0,
        help="Websocket open timeout (covers remote cold starts)",
    )
    parser.add_argument("--deploy-config", type=Path, default=None)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from smoke_duplexio_v3 import pick_voice, write_deploy_yaml

    if args.url is not None:
        # Remote endpoint: the served model path is remote; --voice must be
        # given explicitly or resolvable from a local copy of the export.
        if args.voice is None:
            try:
                args.voice = pick_voice(Path(args.model), None)
            except SystemExit:
                fail("remote mode needs --voice when the export is not local")
        print(f"[ws] voice: {args.voice}")
        try:
            asyncio.run(run_client(args))
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            fail("websocket session raised (see traceback above)")
        return

    args.voice = pick_voice(Path(args.model), args.voice)
    print(f"[ws] voice: {args.voice}")

    with tempfile.TemporaryDirectory(prefix="duplexio-ws-smoke-") as tmp:
        if args.deploy_config is not None:
            deploy_yaml = args.deploy_config
        else:
            overlay = argparse.Namespace(
                max_model_len=32_768,
                gpu_memory_utilization=0.8,
            )
            deploy_yaml = write_deploy_yaml(Path(tmp), overlay)
        server = subprocess.Popen(
            [
                ".venv/bin/vllm-omni",
                "serve",
                args.model,
                "--omni",
                "--deploy-config",
                str(deploy_yaml),
                "--trust-remote-code",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
        )
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    fail(f"server exited early with {server.returncode}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health",
                        timeout=2,
                    ):
                        break
                except OSError:
                    time.sleep(2)
            else:
                fail("server did not become healthy within 900s")
            print("[ws] server healthy")
            asyncio.run(run_client(args))
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            fail("websocket session raised (see traceback above)")
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
