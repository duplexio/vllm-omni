# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation and loading for native DuplexIO serving artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from safetensors.torch import load_file
from torch import Tensor

EXPORT_MANIFEST_FILENAME = "duplexio_export.json"


def resolve_checkpoint_directory(
    model_path: str,
    *,
    revision: str | None = None,
) -> Path:
    """Resolve a local checkpoint path or Hugging Face repository ID."""
    root = Path(model_path)
    if root.is_dir():
        return root

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, revision=revision))


def load_voice_clips(
    model_path: str,
    *,
    sample_rate: int,
    default_voice: str | None,
    revision: str | None = None,
) -> dict[str, tuple[Tensor, ...]]:
    """Validate one exported checkpoint and load its named voice-prompt clips.

    A voice is reference audio, not an embedding: the bundle holds one mono
    waveform per clip at the model's sample rate, and the agent's voice comes
    from pinning one of them ahead of the conversation.
    """
    root = resolve_checkpoint_directory(model_path, revision=revision)
    exported_voice_ids = validate_export_manifest(root)
    manifest_path = root / "voices.json"
    weights_path = root / "voices.safetensors"
    if manifest_path.is_file() != weights_path.is_file():
        raise ValueError(
            "DuplexIO checkpoint must contain both voices.json and "
            "voices.safetensors"
        )
    if not manifest_path.is_file():
        raise ValueError("Native DuplexIO requires at least one exported voice")

    manifest = json.loads(manifest_path.read_text())
    voices = manifest.get("voices") if isinstance(manifest, Mapping) else None
    if not isinstance(voices, Mapping):
        raise ValueError(f"Invalid DuplexIO voice manifest: {manifest_path}")
    bundle_rate = manifest.get("sample_rate") if isinstance(manifest, Mapping) else None
    if bundle_rate != sample_rate:
        raise ValueError(
            f"DuplexIO voice bundle is at {bundle_rate!r} Hz, model wants "
            f"{sample_rate}"
        )
    tensors = load_file(weights_path)
    clips: dict[str, tuple[Tensor, ...]] = {}
    for name, value in voices.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ValueError(f"Invalid DuplexIO voice entry in {manifest_path}")
        tensor_names = value.get("tensors")
        if not isinstance(tensor_names, list) or not tensor_names:
            raise ValueError(f"DuplexIO voice {name!r} lists no clips")
        waveforms: list[Tensor] = []
        for tensor_name in tensor_names:
            if not isinstance(tensor_name, str) or tensor_name not in tensors:
                raise ValueError(f"Missing tensor for DuplexIO voice {name!r}")
            waveform = tensors[tensor_name]
            if waveform.ndim != 1 or waveform.shape[0] < 1:
                raise ValueError(
                    f"Invalid DuplexIO voice clip {tensor_name!r}: shape="
                    f"{tuple(waveform.shape)}, expected one mono waveform"
                )
            waveforms.append(waveform)
        clips[name] = tuple(waveforms)
    if tuple(sorted(clips)) != exported_voice_ids:
        raise ValueError("DuplexIO export manifest and voice artifact disagree")
    if default_voice is not None and default_voice not in clips:
        raise ValueError(
            f"DuplexIO default voice {default_voice!r} is not exported"
        )
    return clips


def validate_export_manifest(root: Path) -> tuple[str, ...]:
    """Validate the serving manifest and return its canonical voice IDs."""
    manifest_path = root / EXPORT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(
            f"Native DuplexIO checkpoint is missing {EXPORT_MANIFEST_FILENAME}"
        )
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Invalid DuplexIO export manifest: {manifest_path}")
    if manifest.get("format") != "duplexio_vllm" or manifest.get("version") != 5:
        raise ValueError(f"Unsupported DuplexIO export manifest: {manifest_path}")
    weight_files = manifest.get("weight_files")
    if (
        not isinstance(weight_files, list)
        or not weight_files
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            for name in weight_files
        )
        or len(set(weight_files)) != len(weight_files)
    ):
        raise ValueError(f"Invalid DuplexIO weight file list: {manifest_path}")
    missing = [name for name in weight_files if not (root / name).is_file()]
    if missing:
        raise ValueError(f"DuplexIO export is missing weight files: {missing}")
    voice_ids = manifest.get("voice_ids")
    if (
        not isinstance(voice_ids, list)
        or not voice_ids
        or any(not isinstance(name, str) or not name for name in voice_ids)
        or len(set(voice_ids)) != len(voice_ids)
    ):
        raise ValueError(f"Invalid DuplexIO voice list: {manifest_path}")
    return tuple(sorted(voice_ids))


__all__ = [
    "load_voice_clips",
    "resolve_checkpoint_directory",
    "validate_export_manifest",
]
