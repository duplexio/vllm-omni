# SPDX-License-Identifier: Apache-2.0
"""Cross-decode: serving-sampled depth columns through the training codec.

Splits code-generation from codec-decode: decodes the SAME per-frame columns
the serving engine sampled (and streamed) through the training repo's proven
batch decode_audio, then judges BOTH waveforms with offline Whisper
(large-v3-turbo) against the agent's text. If Whisper reads the training
decode but not the serving stream, the serving streaming decode is the bug;
if both fail, the columns themselves are bad; if both pass, the wire/browser
is the bug.

Run (GPU node, training venv, cwd training repo):
  .venv/bin/python /dcai/users/thuand/vllm-omni-work/probe_dumps/cross_decode_probe.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

REPO = Path("/dcai/users/thuand/duplexio")
WORK = Path("/dcai/users/thuand/vllm-omni-work/probe_dumps")
CHECKPOINT = REPO / "runs/duplexio_train/489780/checkpoints/checkpoint_4"
EXPECTED_TEXT = (
    "Yeah, I'm here. I can help you. I'm ready to chat with you. "
    "What's on your mind?"
)


def save_wav(path: Path, waveform: np.ndarray) -> None:
    clipped = np.clip(waveform, -1, 1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes((clipped * 32767).astype("<i2").tobytes())


def transcribe(model, waveform_24k: np.ndarray, label: str) -> str:
    from scipy.signal import resample_poly

    audio_16k = resample_poly(waveform_24k.astype(np.float64), 2, 3)
    result = model.transcribe(
        audio_16k.astype(np.float32),
        language="en",
        fp16=True,
    )
    text = result["text"].strip()
    print(f"[whisper] {label}: {text!r}")
    return text


def word_ratio(expected: str, heard: str) -> float:
    from difflib import SequenceMatcher

    def words(text: str) -> list[str]:
        return "".join(
            c if c.isalnum() or c.isspace() else " " for c in text.lower()
        ).split()

    return SequenceMatcher(None, words(expected), words(heard)).ratio()


def main() -> None:
    from duplexio.config import load_config
    from duplexio.evaluate import load_replicated_fsdp_model
    from duplexio.full_duplex_bench import model_autocast

    dump = torch.load(WORK / "serving_codes_490031.pt", weights_only=False)
    codes = dump["codes"].long()  # (N, 8) serving-sampled delayed columns
    print(f"[probe] serving columns: {tuple(codes.shape)}")

    serving_dump = torch.load(WORK / "serving_490031.pt", weights_only=False)
    serving_wave = np.concatenate(
        [f["audio"].numpy() for f in serving_dump["frames"] if f["audio"].numel()]
    ).astype(np.float32)
    print(f"[probe] serving streamed waveform: {serving_wave.shape}")

    config = load_config(REPO / "configs/train.yaml")
    config.model.gradient_checkpointing = False
    model = load_replicated_fsdp_model(
        config.model,
        CHECKPOINT,
        torch.device("cuda"),
        flex_attention_dynamic=False,
    )
    device = next(model.parameters()).device
    with model_autocast(device):
        training_wave = (
            model.decode_audio(codes.to(device)).squeeze().float().cpu().numpy()
        )
    print(f"[probe] training batch-decoded waveform: {training_wave.shape}")
    save_wav(WORK / "cross_training_decode_490031.wav", training_wave)
    save_wav(WORK / "cross_serving_stream_490031.wav", serving_wave)

    # numeric comparison at frame lags 0..2 (streamed decode lags one frame)
    for lag_frames in (0, 1, 2):
        lag = lag_frames * 1920
        m = min(len(serving_wave) - 0, len(training_wave) - lag)
        if m <= 0:
            continue
        a = serving_wave[:m]
        b = training_wave[lag : lag + m]
        err = np.abs(a - b)
        ref = np.abs(b).mean() + 1e-9
        print(
            f"[probe] serving-vs-training lag={lag_frames}: "
            f"mean|err|={err.mean():.5f} rel={err.mean() / ref:.2f} "
            f"corr={np.corrcoef(a, b)[0, 1]:.3f}"
        )

    import whisper

    asr = whisper.load_model("large-v3-turbo", device="cuda")
    heard_training = transcribe(asr, training_wave, "training decode of serving codes")
    heard_serving = transcribe(asr, serving_wave, "serving streamed waveform")
    print(f"[probe] expected agent text: {EXPECTED_TEXT!r}")
    print(f"[probe] ratio training-decode: {word_ratio(EXPECTED_TEXT, heard_training):.2f}")
    print(f"[probe] ratio serving-stream:  {word_ratio(EXPECTED_TEXT, heard_serving):.2f}")


if __name__ == "__main__":
    main()
