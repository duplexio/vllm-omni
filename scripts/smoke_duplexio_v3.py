# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke: load a DuplexIO v3 export and run a live session end-to-end.

Drives the same session machinery the realtime protocol uses — AsyncOmni's
duplex control plane (open / append_audio_chunk / close), the DuplexIO
serving adapter's runtime config, and the transactional PCM append buffer —
feeding real speech frame by frame and checking the mechanical contract:

  * DuplexIOConfig validates as export version 3 (client-side, this process)
  * the engine loads the checkpoint (vLLM's AutoWeightsLoader is strict:
    any missing or unexpected weight fails engine init)
  * every appended frame yields finite outputs with the expected audio
    shape/dtype and in-vocabulary text tokens

Decoded USER/AGENT/TOOL tokens and the agent's first-emission frame are
printed for the human eye; they are NOT asserted (checkpoint-dependent).

Invocation (1 GPU node):

  cd /dcai/users/thuand/vllm-omni-work && \
  PYTHONPATH=/dcai/users/thuand/vllm-omni-work \
  .venv/bin/python scripts/smoke_duplexio_v3.py \
    --model /dcai/users/thuand/duplexio/checkpoints/duplexio-487557-checkpoint-4-vllm-v3

Caveat: the login-profile exports TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR
under /home/anders/... — if that path is unwritable on the GPU node, override
them (and HF_HOME/XDG_CACHE_HOME) to a writable location first, e.g.:

  export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER-cache/torchinductor \
         TRITON_CACHE_DIR=/tmp/$USER-cache/triton
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

FRAME_SIZE = 1_920
SAMPLE_RATE = 24_000

# First sample of the Full-Duplex-Bench candor turn-taking split (v1.0),
# picked deterministically: sorted(listdir())[0] == "1".
DEFAULT_FDB_CLIP = Path(
    "/dcai/users/thuand/duplexio/datasets/full-duplex-bench/"
    "Full-Duplex-Bench-Data/v1.0/candor_turn_taking/1/input.wav"
)


def fail(message: str) -> None:
    print(f"SMOKE FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def validate_v3_config(model_path: Path) -> dict:
    """Run the serving config contract on the export in this process."""
    from vllm_omni.model_executor.models.duplexio.configuration_duplexio import (
        DuplexIOConfig,
    )

    raw = json.loads((model_path / "config.json").read_text())
    version = raw.get("duplexio_export_version")
    if version != 3:
        fail(f"export is duplexio_export_version={version!r}, expected 3")
    config = DuplexIOConfig.from_dict(raw)  # raises on any contract violation
    print(f"[smoke] config OK: export v{config.duplexio_export_version}, "
          f"window={config.audio_attention_window_frames} frames, "
          f"silence_token_id={config.silence_token_id}")
    return raw


def load_speech(audio_path: Path, seconds: float) -> np.ndarray:
    """Load real speech, downmix to mono, resample to 24 kHz float32."""
    import soundfile

    waveform, source_rate = soundfile.read(str(audio_path), dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if source_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        ratio = Fraction(SAMPLE_RATE, source_rate)
        waveform = resample_poly(waveform, ratio.numerator, ratio.denominator)
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = waveform[: int(seconds * SAMPLE_RATE)]
    frame_count = len(waveform) // FRAME_SIZE
    if frame_count < 2:
        fail(f"{audio_path} yields only {frame_count} frames at 24 kHz")
    print(f"[smoke] speech: {audio_path} ({source_rate} Hz -> {SAMPLE_RATE} Hz, "
          f"{frame_count} frames = {frame_count * 0.08:.1f}s)")
    return waveform[: frame_count * FRAME_SIZE]


def pick_voice(model_path: Path, requested: str | None) -> str:
    manifest = json.loads((model_path / "voices.json").read_text())
    voices = sorted(manifest["voices"])
    if not voices:
        fail("export has an empty voice manifest")
    if requested is not None:
        if requested not in voices:
            fail(f"voice {requested!r} not in export ({voices[:5]}...)")
        return requested
    return manifest.get("default_voice") or voices[0]


def frame_output_metadata(output: object) -> dict:
    """Locate the per-frame multimodal metadata on an OmniRequestOutput.

    Same attribute chain the DuplexIO data plane uses for projection. The
    output processor delivers a MultimodalPayload (a Mapping, not a dict).
    """
    from collections.abc import Mapping

    candidates = [output]
    inner = getattr(output, "request_output", None)
    if inner is not None and inner is not output:
        candidates.append(inner)
    for candidate in candidates:
        for holder in (
            candidate,
            *(getattr(candidate, "outputs", None) or [])[:1],
        ):
            metadata = getattr(holder, "multimodal_output", None)
            if isinstance(metadata, Mapping) and metadata:
                return dict(metadata)
    return {}


def scalar_int(value: object) -> int | None:
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1)
        return int(flat[-1].item()) if flat.numel() else None
    if isinstance(value, (list, tuple)):
        return scalar_int(value[-1]) if value else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def audio_samples(value: object) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(value, (list, tuple)):
        return audio_samples(value[-1]) if value else np.zeros(0, dtype=np.float32)
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.numel() and not tensor.is_floating_point():
            raise ValueError(f"audio tensor has non-float dtype {tensor.dtype}")
        # bfloat16 has no numpy dtype; widen for the finite/shape checks.
        return tensor.float().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def write_deploy_yaml(directory: Path, args: argparse.Namespace) -> Path:
    """Minimal duplexio deploy overlay honoring the model's runtime contract
    (no chunked prefill, no prefix caching, one live session)."""
    path = directory / "smoke_duplexio.yaml"
    path.write_text(
        "pipeline: duplexio\n"
        "session_mode: duplex\n"
        "async_chunk: false\n"
        "enable_prefix_caching: false\n"
        "enable_chunked_prefill: false\n"
        "duplex_session:\n"
        "  max_sessions: 1\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    max_num_seqs: 1\n"
        f"    max_model_len: {args.max_model_len}\n"
        f"    gpu_memory_utilization: {args.gpu_memory_utilization}\n"
        "    enforce_eager: true\n"
        "    async_scheduling: false\n"
    )
    return path


async def run_session(args: argparse.Namespace, speech: np.ndarray) -> None:
    import torch  # noqa: F401  (fail early if torch is broken)
    from transformers import AutoTokenizer

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.experimental.fullduplex.duplexio.input import (
        DuplexIOPcmAppendBuffer,
    )
    from vllm_omni.experimental.fullduplex.duplexio.serving_adapter import (
        DuplexIOServingRuntimeAdapter,
    )
    from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
    from vllm_omni.experimental.fullduplex.openai.protocol import (
        DuplexSessionConfig,
    )

    model_path = Path(args.model)
    config_raw = json.loads((model_path / "config.json").read_text())
    silence_token_id = config_raw["silence_token_id"]
    vocab_size = config_raw["text_config"]["vocab_size"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    voice = pick_voice(model_path, args.voice)
    print(f"[smoke] voice: {voice}")

    with tempfile.TemporaryDirectory(prefix="duplexio-smoke-") as tmp:
        if args.deploy_config is not None:
            deploy_yaml = args.deploy_config
            print(f"[smoke] deploy config: {deploy_yaml}")
        else:
            deploy_yaml = write_deploy_yaml(Path(tmp), args)
        # Engine init loads the checkpoint in the stage worker; vLLM's
        # AutoWeightsLoader raises on missing or unexpected weights, so
        # surviving this call is the zero-missing/zero-unexpected assertion.
        # Generous init timeout: CUDA-graph configs compile on first start.
        omni = AsyncOmni(
            model=str(model_path),
            stage_configs_path=str(deploy_yaml),
            init_timeout=1_800,
            stage_init_timeout=1_800,
        )
        print("[smoke] engine initialized (weights loaded strictly)")
        try:
            await drive_session(
                omni,
                args=args,
                speech=speech,
                voice=voice,
                tokenizer=tokenizer,
                silence_token_id=silence_token_id,
                vocab_size=vocab_size,
                adapter=DuplexIOServingRuntimeAdapter(
                    encode_audio=lambda *_args: None,
                ),
                fence_cls=DuplexFence,
                session_config_cls=DuplexSessionConfig,
                buffer_cls=DuplexIOPcmAppendBuffer,
            )
        finally:
            omni.shutdown()


async def drive_session(
    omni,
    *,
    args,
    speech: np.ndarray,
    voice: str,
    tokenizer,
    silence_token_id: int,
    vocab_size: int,
    adapter,
    fence_cls,
    session_config_cls,
    buffer_cls,
) -> None:
    async def open_session(session_id: str):
        fence = fence_cls(session_id)
        session_config = session_config_cls(
            model=args.model,
            voice=voice,
            # None: keep the export's sampling; 0.0 would divide top_p logits
            # by zero (the realtime protocol forbids it, gt=0).
            temperature=None,
            extra_body={"full_duplex": True, "start_role": args.start_role},
        )
        model_config = omni.engine.stage_vllm_configs[0].model_config
        runtime_config = await adapter.prepare_runtime_config(
            session_config,
            model_config=model_config,
        )
        runtime_config["duplexio_sampling_seed"] = 0  # reproducible smoke
        if args.argmax:
            runtime_config["duplexio_text_sampling"] = {
                "mode": "argmax",
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 1.0,
            }
            runtime_config["duplexio_emit_temperatures"] = {
                "user": 0.0,
                "agent": 0.0,
                "tool_call": 0.0,
            }
            runtime_config["duplexio_depth_sampling"] = {
                "temperature": 1.0,
                "top_k": 1,
            }
        if args.depth_top_k is not None or args.depth_temperature is not None:
            depth = dict(runtime_config["duplexio_depth_sampling"])
            if args.depth_top_k is not None:
                depth["top_k"] = args.depth_top_k
            if args.depth_temperature is not None:
                depth["temperature"] = args.depth_temperature
            runtime_config["duplexio_depth_sampling"] = depth
            print(f"[smoke] depth sampling override: {depth}")
        await omni.open_duplex_session_async(
            session_id,
            capabilities=adapter.capabilities(max_sessions=1).as_dict(),
            session_config=session_config.as_dict(),
            runtime_config=runtime_config,
            fence=fence,
            timeout=60.0,
        )
        print(f"[smoke] session open: {session_id}")
        return fence, runtime_config

    def token_text(token_id: int) -> str:
        if token_id == silence_token_id:
            return "·"
        return tokenizer.decode([token_id]).replace("\n", "\\n") or "?"

    async def stream_pcm(
        session_id: str,
        fence,
        pcm: np.ndarray,
        *,
        label: str,
    ) -> list[dict]:
        buffer = buffer_cls()
        frame_count = len(pcm) // FRAME_SIZE
        records: list[dict] = []
        for frame_index in range(frame_count):
            chunk = pcm[frame_index * FRAME_SIZE : (frame_index + 1) * FRAME_SIZE]
            framed = buffer.append(
                {
                    "type": "audio",
                    "format": "pcm_f32le",
                    "sample_rate_hz": SAMPLE_RATE,
                    "audio": base64.b64encode(chunk.tobytes()).decode("ascii"),
                    "is_speech": True,
                },
                chunk_period_ms=80,
            )
            if framed is None:
                fail(f"append buffer produced no frame payload at {frame_index}")
            result = await omni.append_duplex_input_async(
                session_id,
                mode="append_audio_chunk",
                payload=framed,
                final=frame_index == frame_count - 1,
                fence=fence,
                # First frame pays warmup/compile; be generous once.
                timeout=600.0 if frame_index == 0 else 60.0,
                collect_outputs=True,
            )

            outputs = result.get("data_plane_outputs")
            if not outputs:
                fail(f"{label} frame {frame_index} produced no data-plane output")
            metadata = frame_output_metadata(outputs[-1])
            if not metadata:
                fail(f"{label} frame {frame_index} has no multimodal metadata")

            user_id = scalar_int(metadata.get("user_token_id"))
            agent_id = scalar_int(metadata.get("agent_token_id"))
            tool_id = scalar_int(metadata.get("tool_call_token_id"))
            waveform = audio_samples(metadata.get("audio"))
            codes = depth_codes(metadata.get("agent_audio_token_ids"))

            for name, token_id in (
                ("user", user_id),
                ("agent", agent_id),
                ("tool", tool_id),
            ):
                if token_id is None or not 0 <= token_id < vocab_size:
                    fail(
                        f"{label} frame {frame_index} {name} token {token_id!r} "
                        f"outside vocab of {vocab_size}"
                    )
            if len(waveform) not in (0, FRAME_SIZE):
                fail(
                    f"{label} frame {frame_index} audio has {len(waveform)} "
                    f"samples, expected 0 (acoustic delay) or {FRAME_SIZE}"
                )
            if not np.isfinite(waveform).all():
                fail(f"{label} frame {frame_index} audio has non-finite samples")
            rms = math.sqrt(float(np.mean(waveform**2))) if len(waveform) else 0.0
            records.append(
                {
                    "user": user_id,
                    "agent": agent_id,
                    "tool": tool_id,
                    "codes": codes,
                    "audio": waveform,
                    "rms": rms,
                }
            )
            print(
                f"[{label} {frame_index:3d}] "
                f"user={token_text(user_id)!r} agent={token_text(agent_id)!r} "
                f"tool={token_text(tool_id)!r} "
                f"agent_audio={len(waveform)} samples (rms={rms:.4f})"
            )
        return records

    async def send_prefill(session_id: str, fence, runtime_config) -> list[dict]:
        from types import SimpleNamespace

        payloads = adapter.initial_data_plane_payloads(
            SimpleNamespace(runtime_config=runtime_config, turn_id=0)
        )
        prefill_records: list[dict] = []
        for payload_index, payload in enumerate(payloads):
            result = await omni.append_duplex_input_async(
                session_id,
                mode="append_audio_chunk",
                payload=dict(payload),
                final=False,
                fence=fence,
                timeout=600.0,
                collect_outputs=True,
            )
            outputs = result.get("data_plane_outputs")
            if not outputs:
                fail(f"prefill append {payload_index} produced no output")
            metadata = frame_output_metadata(outputs[-1])
            if payload.get("duplexio_prefill_final"):
                prefill_records.append(
                    {
                        "user": scalar_int(metadata.get("user_token_id")),
                        "agent": scalar_int(metadata.get("agent_token_id")),
                        "tool": scalar_int(metadata.get("tool_call_token_id")),
                        "codes": depth_codes(
                            metadata.get("agent_audio_token_ids")
                        ),
                        "audio": audio_samples(metadata.get("audio")),
                        "rms": 0.0,
                        "prefill": True,
                    }
                )
        print(f"[smoke] sent {len(payloads)} prefill appends "
              f"({len(prefill_records)} sampled)")
        return prefill_records

    session_id = "duplexio-v3-smoke"
    fence, runtime_config = await open_session(session_id)
    try:
        prefill_records: list[dict] = []
        if args.prefill:
            prefill_records = await send_prefill(session_id, fence, runtime_config)
        records = await stream_pcm(session_id, fence, speech, label="frame")
    finally:
        await omni.close_duplex_session_async(session_id, fence=fence, timeout=30.0)

    frame_count = len(speech) // FRAME_SIZE
    if len(records) != frame_count:
        fail(f"decoded {len(records)} frames, appended {frame_count}")
    print(f"\n[smoke] all {frame_count} frames returned finite, well-shaped outputs")

    sampled_records = [*prefill_records, *records]
    user_token_ids = [r["user"] for r in records if r["user"] != silence_token_id]
    agent_token_ids = [
        r["agent"] for r in sampled_records if r["agent"] != silence_token_id
    ]
    agent_frames = [
        index for index, r in enumerate(sampled_records)
        if r["agent"] != silence_token_id
    ]
    transcript = tokenizer.decode(user_token_ids) if user_token_ids else "(none)"
    print(f"[smoke] USER-stream transcription ({len(user_token_ids)} tokens): "
          f"{transcript}")
    agent_text = tokenizer.decode(agent_token_ids) if agent_token_ids else "(none)"
    print(f"[smoke] AGENT text ({len(agent_token_ids)} tokens): {agent_text}")

    if args.dump is not None:
        import torch

        torch.save(
            {
                "frames": [
                    {
                        "user": r["user"],
                        "agent": r["agent"],
                        "tool": r["tool"],
                        "codes": (
                            torch.as_tensor(r["codes"])
                            if r["codes"] is not None
                            else None
                        ),
                        "audio": torch.from_numpy(np.asarray(r["audio"]).copy()),
                        "prefill": bool(r.get("prefill", False)),
                    }
                    for r in [*prefill_records, *records]
                ],
                "silence_token_id": silence_token_id,
            },
            args.dump,
        )
        print(f"[smoke] per-frame dump written to {args.dump}")

    if args.verify_agent_speech:
        await verify_agent_speech(
            records=sampled_records,
            agent_frames=agent_frames,
            agent_text=agent_text,
            open_session=open_session,
            stream_pcm=stream_pcm,
            close_session=lambda sid, f: omni.close_duplex_session_async(
                sid, fence=f, timeout=30.0
            ),
            tokenizer=tokenizer,
            silence_token_id=silence_token_id,
        )
    elif agent_frames:
        print(f"[smoke] agent started emitting at frame {agent_frames[0]} "
              f"(t={agent_frames[0] * 0.08:.2f}s)")
    else:
        print("[smoke] agent never emitted (checkpoint-dependent; not asserted)")
    print("SMOKE PASS")


def depth_codes(value: object):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return depth_codes(value[-1]) if value else None
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1)
        return flat.tolist() if flat.numel() else None
    return None


def normalized_words(text: str) -> list[str]:
    cleaned = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in text.lower()
    )
    return cleaned.split()


async def verify_agent_speech(
    *,
    records: list[dict],
    agent_frames: list[int],
    agent_text: str,
    open_session,
    stream_pcm,
    close_session,
    tokenizer,
    silence_token_id: int,
) -> None:
    """Prove the agent's spoken audio matches its text stream.

    The collected agent audio is played back through a second session as
    user input; the USER stream is the model's own proven ASR, so its
    transcription must (fuzzily) match the agent text.
    """
    from difflib import SequenceMatcher

    if not agent_frames:
        fail("agent never spoke; cannot verify agent speech "
             "(increase --silence-seconds?)")

    # Audio content trails the text stream (acoustic delay plus the depth
    # sampler's own pacing); use a generous window for the loudness check.
    span_start = agent_frames[0]
    span_end = min(len(records), agent_frames[-1] + 25)
    speech_rms_values = [
        records[i]["rms"] for i in range(span_start, span_end)
        if len(records[i]["audio"])
    ]
    quiet_rms_values = [
        r["rms"] for i, r in enumerate(records)
        if len(r["audio"]) and not span_start <= i < span_end
    ]
    speech_rms = float(np.mean(speech_rms_values)) if speech_rms_values else 0.0
    quiet_rms = float(np.mean(quiet_rms_values)) if quiet_rms_values else 0.0
    print(f"[smoke] agent audio rms: speaking={speech_rms:.4f} "
          f"idle={quiet_rms:.4f} over frames [{span_start}, {span_end})")
    loud_rms = max(
        (records[i]["rms"] for i in range(span_start, span_end)
         if len(records[i]["audio"])),
        default=0.0,
    )
    if loud_rms < 1e-2:
        fail(f"agent audio is silent around the agent's turn "
             f"(peak frame rms={loud_rms:.5f})")

    # Play back the ENTIRE agent output track: the model's audio may trail
    # its text by several frames, and everything outside the agent's turn is
    # near-silence anyway.
    agent_audio = np.concatenate(
        [r["audio"] for r in records if len(r["audio"])]
        + [np.zeros(FRAME_SIZE, dtype=np.float32)] * 12
    ).astype(np.float32)
    session_id = "duplexio-v3-smoke-asr"
    fence, _ = await open_session(session_id)
    try:
        asr_records = await stream_pcm(session_id, fence, agent_audio, label="asr")
    finally:
        await close_session(session_id, fence)
    heard_ids = [
        r["user"] for r in asr_records if r["user"] != silence_token_id
    ]
    heard_text = tokenizer.decode(heard_ids) if heard_ids else "(none)"
    print(f"[smoke] playback USER-stream heard ({len(heard_ids)} tokens): "
          f"{heard_text}")

    expected = normalized_words(agent_text)
    heard = normalized_words(heard_text)
    ratio = SequenceMatcher(None, expected, heard).ratio()
    print(f"[smoke] agent speech/text match ratio: {ratio:.2f} "
          f"(expected words: {expected}) (heard words: {heard})")
    if ratio < 0.6:
        fail(
            f"agent speech does not match agent text (ratio {ratio:.2f} < 0.60)"
        )
    print("[smoke] agent speech matches agent text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="DuplexIO v3 export directory")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Any speech wav (default: first Full-Duplex-Bench candor clip)",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=0.0,
        help="Append this much silent PCM after the clip so the model takes "
        "its turn",
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="Force deterministic argmax sampling for text, emit, and depth",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Save per-frame records (token ids, depth codebook codes, audio) "
        "to this .pt file",
    )
    parser.add_argument(
        "--start-role",
        choices=("user", "agent"),
        default="user",
        help="agent: assistant-first session (model narrates over silent "
        "user input)",
    )
    parser.add_argument(
        "--silent-input",
        action="store_true",
        help="Feed pure silence instead of the speech clip",
    )
    parser.add_argument("--depth-top-k", type=int, default=None)
    parser.add_argument("--depth-temperature", type=float, default=None)
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="Send the adapter's system-prompt prefill appends before live "
        "audio (the deployed session flow) instead of interleaving system "
        "tokens over the first live frames",
    )
    parser.add_argument(
        "--verify-agent-speech",
        action="store_true",
        help="Require the agent to speak, and require its audio (fed back "
        "through a second session as user input) to transcribe to its own "
        "text stream",
    )
    parser.add_argument("--voice", default=None, help="Exported voice id")
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=None,
        help="Use an existing deploy YAML (e.g. vllm_omni/deploy/duplexio.yaml "
        "to exercise the CUDA-graph frame path) instead of the eager smoke "
        "overlay",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not (model_path / "config.json").is_file():
        fail(f"{model_path} is not a DuplexIO export (no config.json)")
    validate_v3_config(model_path)

    if args.silent_input:
        frames = int(args.seconds * SAMPLE_RATE) // FRAME_SIZE
        speech = np.zeros(frames * FRAME_SIZE, dtype=np.float32)
        print(f"[smoke] silent input: {frames} frames")
    else:
        audio_path = args.audio if args.audio is not None else DEFAULT_FDB_CLIP
        if not audio_path.is_file():
            fail(
                f"no speech input: {audio_path} does not exist "
                "(pass --audio <wav>; this smoke takes real speech only)"
            )
        speech = load_speech(audio_path, args.seconds)
    if args.silence_seconds > 0:
        silent_frames = int(args.silence_seconds * SAMPLE_RATE) // FRAME_SIZE
        speech = np.concatenate(
            (speech, np.zeros(silent_frames * FRAME_SIZE, dtype=np.float32))
        )
        print(f"[smoke] appended {silent_frames} silent frames "
              f"({silent_frames * 0.08:.1f}s) for the agent's turn")

    try:
        asyncio.run(run_session(args, speech))
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        fail("session run raised (see traceback above)")


if __name__ == "__main__":
    main()
