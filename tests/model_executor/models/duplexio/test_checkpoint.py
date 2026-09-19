# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm_omni.model_executor.models.duplexio.checkpoint import (
    resolve_checkpoint_directory,
    validate_export_manifest,
)


def _write_checkpoint(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "duplexio_export.json").write_text(
        json.dumps(
            {
                "format": "duplexio_vllm",
                "version": 7,
                "weight_files": ["model.safetensors"],
            }
        )
    )


def test_checkpoint_contract_carries_no_voices(tmp_path) -> None:
    _write_checkpoint(tmp_path)

    # A voice is reference audio the caller supplies per session, so the artifact
    # ships none and needs no voice bundle beside it.
    validate_export_manifest(tmp_path)
    assert not (tmp_path / "voices.json").exists()
    assert not (tmp_path / "voices.safetensors").exists()


def test_checkpoint_contract_rejects_missing_weight_file(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()

    with pytest.raises(ValueError, match="missing weight files"):
        validate_export_manifest(tmp_path)


def test_checkpoint_contract_rejects_old_export_version(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    manifest_path = tmp_path / "duplexio_export.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 5
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsupported DuplexIO export manifest"):
        validate_export_manifest(tmp_path)


def test_checkpoint_resolves_hugging_face_repository(
    tmp_path,
    monkeypatch,
) -> None:
    import huggingface_hub

    calls = []

    def snapshot_download(model_path: str, *, revision: str | None) -> str:
        calls.append((model_path, revision))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    root = resolve_checkpoint_directory(
        "owner/duplexio",
        revision="export-v2",
    )

    assert root == tmp_path
    assert calls == [("owner/duplexio", "export-v2")]
