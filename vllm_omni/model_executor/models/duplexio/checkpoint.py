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


def validate_export_manifest(root: Path) -> None:
    """Validate the serving manifest.

    The artifact carries no voices: a voice is reference audio the caller supplies
    per session, so any clip works without re-exporting the model.
    """
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


__all__ = [
    "resolve_checkpoint_directory",
    "validate_export_manifest",
]
