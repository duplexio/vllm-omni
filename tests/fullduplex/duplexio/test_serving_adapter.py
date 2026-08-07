# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vllm_omni.experimental.fullduplex.duplexio.serving_adapter import (
    DuplexIOClientRuntimeConfigError,
    DuplexIOServingRuntimeAdapter,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSessionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_role", "expected_prefix_ids"),
    [
        (None, [5, 6]),
        ("user", [5, 6]),
        ("agent", [7, 8]),
    ],
)
async def test_serving_config_selects_exported_default_voice(
    tmp_path,
    monkeypatch,
    start_role: str | None,
    expected_prefix_ids: list[int],
) -> None:
    (tmp_path / "voices.json").write_text(
        json.dumps(
            {
                "default_voice": "voice-b",
                "voices": {"voice-b": {}, "voice-a": {}},
            }
        )
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(
            default_voice="voice-b",
            default_system_prompt="system",
            initial_agent_prefix="<|im_start|>assistant\n",
            initial_user_prefix="<|im_start|>user\n",
            pad_token_id=11,
            depth_transformer_config={
                "sampling_temperature": 0.9,
                "sampling_top_k": 32,
            },
        ),
    )
    encoded = {
        "system": [3, 4],
        "<|im_start|>user\n": [5, 6],
        "<|im_start|>assistant\n": [7, 8],
    }
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: encoded[text]
    )
    monkeypatch.setattr(
        "vllm_omni.experimental.fullduplex.duplexio.serving_adapter."
        "cached_tokenizer_from_config",
        lambda config: tokenizer,
    )

    extra_body: dict[str, object] = {"full_duplex": True}
    if start_role is not None:
        extra_body["start_role"] = start_role
    runtime = await DuplexIOServingRuntimeAdapter.prepare_runtime_config(
        DuplexSessionConfig(
            modalities=["text", "audio"],
            extra_body=extra_body,
        ),
        model_config=model_config,
    )

    assert runtime["duplexio_voice"] == "voice-b"
    assert runtime["duplexio_voice_ids"] == ["voice-a", "voice-b"]
    assert runtime["duplexio_scheduler_token_id"] == 11
    assert isinstance(runtime["duplexio_sampling_seed"], int)
    assert runtime["duplexio_system_token_ids"] == [
        3,
        4,
        *expected_prefix_ids,
    ]
    assert runtime["duplexio_start_role"] == (start_role or "user")
    assert runtime["duplexio_depth_sampling"] == {
        "temperature": 0.9,
        "top_k": 32,
    }


@pytest.mark.asyncio
async def test_serving_config_rejects_unknown_voice(tmp_path) -> None:
    (tmp_path / "voices.json").write_text(
        json.dumps({"default_voice": "voice-a", "voices": {"voice-a": {}}})
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(default_voice="voice-a", pad_token_id=0),
    )
    config = DuplexSessionConfig(
        modalities=["audio"],
        voice="missing",
        extra_body={"full_duplex": True},
    )

    with pytest.raises(DuplexIOClientRuntimeConfigError) as exc_info:
        await DuplexIOServingRuntimeAdapter.prepare_runtime_config(
            config,
            model_config=model_config,
        )

    assert exc_info.value.code == "voice_not_found"


@pytest.mark.asyncio
async def test_serving_config_rejects_invalid_manifest_default(tmp_path) -> None:
    (tmp_path / "voices.json").write_text(
        json.dumps(
            {
                "default_voice": "missing",
                "voices": {"voice-a": {}},
            }
        )
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(default_voice=None, pad_token_id=0),
    )

    with pytest.raises(ValueError, match="default voice"):
        await DuplexIOServingRuntimeAdapter.prepare_runtime_config(
            DuplexSessionConfig(
                modalities=["audio"],
                extra_body={"full_duplex": True},
            ),
            model_config=model_config,
        )


def test_duplexio_capabilities_use_native_scheduler_data_plane() -> None:
    capabilities = DuplexIOServingRuntimeAdapter.capabilities(max_sessions=2)

    assert capabilities.supports_model_native_turn_policy
    assert capabilities.supports_input_append
    assert capabilities.supports_core_resumable_request
    assert capabilities.supports_independent_io_streams
    assert capabilities.supports_multi_session_same_replica
    assert capabilities.input_modes == ["append_audio_chunk"]
    assert capabilities.chunk_period_ms == 80


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("instructions", "new prompt", "instructions_update_unsupported"),
        ("voice", "voice-b", "voice_update_unsupported"),
    ],
)
def test_duplexio_rejects_stateful_conditioning_updates(
    field: str,
    value: str,
    code: str,
) -> None:
    config = DuplexSessionConfig(
        modalities=["text", "audio"],
        instructions="original prompt",
        voice="voice-a",
        extra_body={"full_duplex": True},
    )
    setattr(config, field, value)
    current = {
        "instructions": "original prompt",
        "duplexio_voice": "voice-a",
        "duplexio_voice_ids": ["voice-a", "voice-b"],
    }

    with pytest.raises(DuplexIOClientRuntimeConfigError) as exc_info:
        DuplexIOServingRuntimeAdapter.validate_runtime_config_for_session(
            config,
            current,
        )

    assert exc_info.value.code == code


def test_duplexio_rejects_start_role_update() -> None:
    config = DuplexSessionConfig(
        modalities=["text", "audio"],
        extra_body={"full_duplex": True, "start_role": "agent"},
    )
    current = {
        "instructions": None,
        "duplexio_start_role": "user",
        "duplexio_voice": "voice-a",
        "duplexio_voice_ids": ["voice-a"],
    }

    with pytest.raises(DuplexIOClientRuntimeConfigError) as exc_info:
        DuplexIOServingRuntimeAdapter.validate_runtime_config_for_session(
            config,
            current,
        )

    assert exc_info.value.code == "start_role_update_unsupported"


@pytest.mark.asyncio
async def test_duplexio_rejects_invalid_start_role() -> None:
    with pytest.raises(DuplexIOClientRuntimeConfigError) as exc_info:
        await DuplexIOServingRuntimeAdapter.prepare_runtime_config(
            DuplexSessionConfig(
                modalities=["text", "audio"],
                extra_body={"full_duplex": True, "start_role": "assistant"},
            ),
            model_config=SimpleNamespace(),
        )

    assert exc_info.value.code == "start_role_invalid"


@pytest.mark.asyncio
async def test_duplexio_requires_full_duplex_mode(tmp_path) -> None:
    (tmp_path / "voices.json").write_text(
        json.dumps({"default_voice": "voice-a", "voices": {"voice-a": {}}})
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(default_voice="voice-a", pad_token_id=0),
    )

    with pytest.raises(DuplexIOClientRuntimeConfigError) as exc_info:
        await DuplexIOServingRuntimeAdapter.prepare_runtime_config(
            DuplexSessionConfig(modalities=["text", "audio"]),
            model_config=model_config,
        )

    assert exc_info.value.code == "full_duplex_required"


@pytest.mark.asyncio
async def test_duplexio_requires_voice_for_text_projection(
    tmp_path,
) -> None:
    (tmp_path / "voices.json").write_text(
        json.dumps({"default_voice": None, "voices": {"voice-a": {}}})
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(default_voice=None, pad_token_id=0),
    )

    with pytest.raises(DuplexIOClientRuntimeConfigError) as exc_info:
        await DuplexIOServingRuntimeAdapter.prepare_runtime_config(
            DuplexSessionConfig(
                modalities=["text"],
                extra_body={"full_duplex": True},
            ),
            model_config=model_config,
        )

    assert exc_info.value.code == "voice_required"
