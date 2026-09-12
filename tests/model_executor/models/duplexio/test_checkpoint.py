# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.model_executor.models.duplexio.checkpoint import (
    load_voice_clips,
    resolve_checkpoint_directory,
    validate_export_manifest,
)


def _write_checkpoint(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "duplexio_export.json").write_text(
        json.dumps(
            {
                "format": "duplexio_vllm",
                "version": 5,
                "weight_files": ["model.safetensors"],
                "voice_ids": ["alice"],
            }
        )
    )
    (tmp_path / "voices.json").write_text(
        json.dumps(
            {
                "default_voice": "alice",
                "sample_rate": 24_000,
                "voices": {
                    "alice": {"tensors": ["voice.0.0", "voice.0.1"]}
                },
            }
        )
    )
    save_file(
        {"voice.0.0": torch.ones(4_800), "voice.0.1": torch.full((2_400,), 0.5)},
        tmp_path / "voices.safetensors",
    )


def test_checkpoint_contract_loads_every_voice_clip(tmp_path) -> None:
    _write_checkpoint(tmp_path)

    clips = load_voice_clips(
        str(tmp_path),
        sample_rate=24_000,
        default_voice="alice",
    )

    assert set(clips) == {"alice"}
    assert [clip.shape for clip in clips["alice"]] == [(4_800,), (2_400,)]
    torch.testing.assert_close(clips["alice"][0], torch.ones(4_800))


def test_checkpoint_contract_rejects_a_bundle_at_another_rate(tmp_path) -> None:
    _write_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="voice bundle is at"):
        load_voice_clips(
            str(tmp_path),
            sample_rate=16_000,
            default_voice=None,
        )


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
        load_voice_clips(
            str(tmp_path),
            sample_rate=24_000,
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
        revision="export-v2",
    )

    assert root == tmp_path
    assert calls == [("owner/duplexio", "export-v2")]
