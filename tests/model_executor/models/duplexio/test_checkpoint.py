# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.model_executor.models.duplexio.checkpoint import (
    load_voice_pools,
    resolve_checkpoint_directory,
    validate_export_manifest,
)


def _write_checkpoint(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "duplexio_export.json").write_text(
        json.dumps(
            {
                "format": "duplexio_vllm",
                "version": 1,
                "weight_files": ["model.safetensors"],
                "voice_ids": ["alice"],
            }
        )
    )
    (tmp_path / "voices.json").write_text(
        json.dumps(
            {
                "default_voice": "alice",
                "voices": {
                    "alice": {"tensor": "voice.0", "num_embeddings": 2}
                },
            }
        )
    )
    save_file({"voice.0": torch.ones(2, 3)}, tmp_path / "voices.safetensors")


def test_checkpoint_contract_loads_exact_voice_pool(tmp_path) -> None:
    _write_checkpoint(tmp_path)

    pools = load_voice_pools(
        str(tmp_path),
        speaker_embed_dim=3,
        default_voice="alice",
    )

    assert set(pools) == {"alice"}
    torch.testing.assert_close(pools["alice"], torch.ones(2, 3))


def test_checkpoint_contract_rejects_missing_weight_file(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()

    with pytest.raises(ValueError, match="missing weight files"):
        validate_export_manifest(tmp_path)


def test_checkpoint_contract_rejects_voice_manifest_disagreement(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    manifest = json.loads((tmp_path / "duplexio_export.json").read_text())
    manifest["voice_ids"] = ["bob"]
    (tmp_path / "duplexio_export.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="voice artifact disagree"):
        load_voice_pools(
            str(tmp_path),
            speaker_embed_dim=3,
            default_voice=None,
        )


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
        revision="export-v1",
    )

    assert root == tmp_path
    assert calls == [("owner/duplexio", "export-v1")]
