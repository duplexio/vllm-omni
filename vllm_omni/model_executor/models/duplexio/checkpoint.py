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


def load_voice_pools(
    model_path: str,
    *,
    speaker_embed_dim: int,
    default_voice: str | None,
    revision: str | None = None,
) -> dict[str, Tensor]:
    """Validate one exported checkpoint and load its named speaker pools."""
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
    tensors = load_file(weights_path)
    pools: dict[str, Tensor] = {}
    for name, value in voices.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ValueError(f"Invalid DuplexIO voice entry in {manifest_path}")
        tensor_name = value.get("tensor")
        if not isinstance(tensor_name, str) or tensor_name not in tensors:
            raise ValueError(f"Missing tensor for DuplexIO voice {name!r}")
        pool = tensors[tensor_name]
        expected_count = value.get("num_embeddings")
        if (
            pool.ndim != 2
            or pool.shape[0] < 1
            or pool.shape[1] != speaker_embed_dim
            or expected_count != pool.shape[0]
        ):
            raise ValueError(
                f"Invalid DuplexIO speaker pool {name!r}: shape={tuple(pool.shape)}, "
                f"num_embeddings={expected_count!r}"
            )
        pools[name] = pool
    if tuple(sorted(pools)) != exported_voice_ids:
        raise ValueError("DuplexIO export manifest and voice artifact disagree")
    if default_voice is not None and default_voice not in pools:
        raise ValueError(
            f"DuplexIO default voice {default_voice!r} is not exported"
        )
    return pools


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
    if manifest.get("format") != "duplexio_vllm" or manifest.get("version") != 3:
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
    "load_voice_pools",
    "resolve_checkpoint_directory",
    "validate_export_manifest",
]
