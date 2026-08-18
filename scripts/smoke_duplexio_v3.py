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

    Same attribute chain the DuplexIO data plane uses for projection.
    """
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
            if isinstance(metadata, dict) and metadata:
                return metadata
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
        # No cast: the float32 dtype assertion runs on the returned array.
        return value.detach().cpu().numpy().reshape(-1)
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
        deploy_yaml = write_deploy_yaml(Path(tmp), args)
        # Engine init loads the checkpoint in the stage worker; vLLM's
        # AutoWeightsLoader raises on missing or unexpected weights, so
        # surviving this call is the zero-missing/zero-unexpected assertion.
        omni = AsyncOmni(model=str(model_path), stage_configs_path=str(deploy_yaml))
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
                adapter=DuplexIOServingRuntimeAdapter,
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
    session_id = "duplexio-v3-smoke"
    fence = fence_cls(session_id)
    session_config = session_config_cls(
        model=args.model,
        voice=voice,
        temperature=0.0,
        extra_body={"full_duplex": True},
    )
    model_config = omni.engine.stage_vllm_configs[0].model_config
    runtime_config = await adapter.prepare_runtime_config(
        session_config,
        model_config=model_config,
    )
    runtime_config["duplexio_sampling_seed"] = 0  # reproducible smoke
    await omni.open_duplex_session_async(
        session_id,
        capabilities=adapter.capabilities(max_sessions=1).as_dict(),
        session_config=session_config.as_dict(),
        runtime_config=runtime_config,
        fence=fence,
        timeout=60.0,
    )
    print(f"[smoke] session open: {session_id}")

    buffer = buffer_cls()
    frame_count = len(speech) // FRAME_SIZE
    audio_position = 0  # mirrors the model's per-request counter: every
    # PCM append carries audio, so audio time advances by exactly one.
    first_agent_frame: int | None = None
    user_token_ids: list[int] = []
    decoded_frames = 0

    def token_text(token_id: int) -> str:
        if token_id == silence_token_id:
            return "·"
        return tokenizer.decode([token_id]).replace("\n", "\\n") or "?"

    try:
        for frame_index in range(frame_count):
            chunk = speech[frame_index * FRAME_SIZE : (frame_index + 1) * FRAME_SIZE]
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
            audio_position += 1

            outputs = result.get("data_plane_outputs")
            if not outputs:
                fail(f"frame {frame_index} produced no data-plane output")
            metadata = frame_output_metadata(outputs[-1])
            if not metadata:
                fail(f"frame {frame_index} output has no multimodal metadata")

            user_id = scalar_int(metadata.get("user_token_id"))
            agent_id = scalar_int(metadata.get("agent_token_id"))
            tool_id = scalar_int(metadata.get("tool_call_token_id"))
            waveform = audio_samples(metadata.get("audio"))

            for name, token_id in (
                ("user", user_id),
                ("agent", agent_id),
                ("tool", tool_id),
            ):
                if token_id is None or not 0 <= token_id < vocab_size:
                    fail(
                        f"frame {frame_index} {name} token {token_id!r} "
                        f"outside vocab of {vocab_size}"
                    )
            if waveform.dtype != np.float32:
                fail(f"frame {frame_index} audio dtype {waveform.dtype} != float32")
            if len(waveform) not in (0, FRAME_SIZE):
                fail(
                    f"frame {frame_index} audio has {len(waveform)} samples, "
                    f"expected 0 (acoustic delay) or {FRAME_SIZE}"
                )
            if not np.isfinite(waveform).all():
                fail(f"frame {frame_index} audio contains non-finite samples")
            rms = math.sqrt(float(np.mean(waveform**2))) if len(waveform) else 0.0

            if user_id != silence_token_id:
                user_token_ids.append(user_id)
            if agent_id != silence_token_id and first_agent_frame is None:
                first_agent_frame = frame_index
            decoded_frames += 1
            print(
                f"[frame {frame_index:3d}] audio_pos={audio_position:3d} "
                f"user={token_text(user_id)!r} agent={token_text(agent_id)!r} "
                f"tool={token_text(tool_id)!r} "
                f"agent_audio={len(waveform)} samples (rms={rms:.4f})"
            )
    finally:
        await omni.close_duplex_session_async(session_id, fence=fence, timeout=30.0)

    if decoded_frames != frame_count:
        fail(f"decoded {decoded_frames} frames, appended {frame_count}")
    print(f"\n[smoke] all {frame_count} frames returned finite, well-shaped outputs")
    transcript = tokenizer.decode(user_token_ids) if user_token_ids else "(none)"
    print(f"[smoke] USER-stream transcription ({len(user_token_ids)} tokens): "
          f"{transcript}")
    if first_agent_frame is None:
        print("[smoke] agent never emitted (checkpoint-dependent; not asserted)")
    else:
        print(f"[smoke] agent started emitting at frame {first_agent_frame} "
              f"(t={first_agent_frame * 0.08:.2f}s)")
    print("SMOKE PASS")


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
    parser.add_argument("--voice", default=None, help="Exported voice id")
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not (model_path / "config.json").is_file():
        fail(f"{model_path} is not a DuplexIO export (no config.json)")
    validate_v3_config(model_path)

    audio_path = args.audio if args.audio is not None else DEFAULT_FDB_CLIP
    if not audio_path.is_file():
        fail(
            f"no speech input: {audio_path} does not exist "
            "(pass --audio <wav>; this smoke takes real speech only)"
        )
    speech = load_speech(audio_path, args.seconds)

    try:
        asyncio.run(run_session(args, speech))
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        fail("session run raised (see traceback above)")


if __name__ == "__main__":
    main()
