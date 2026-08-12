# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vllm_omni.experimental.fullduplex.duplexio.input import (
    DUPLEXIO_FRAME_SIZE,
)
from vllm_omni.experimental.fullduplex.duplexio.serving_adapter import (
    DuplexIOClientRuntimeConfigError,
    DuplexIOServingRuntimeAdapter,
    parse_client_sampling_config,
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
            silence_token_id=13,
            rollout_sampling_config={
                "mode": "argmax",
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.95,
            },
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
        "<|im_start|>": [11],
        "<|im_end|>": [12],
        "<think>": [14, 15],
        "</think>": [16, 17],
        "<tool_call>": [18],
        "</tool_call>": [19],
    }
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: encoded[text],
        decode=lambda token_ids, skip_special_tokens: "decoded",
        all_special_ids=[11, 12],
    )
    monkeypatch.setattr(
        "vllm_omni.experimental.fullduplex.duplexio.serving_adapter."
        "cached_tokenizer_from_config",
        lambda config: tokenizer,
    )

    extra_body: dict[str, object] = {"full_duplex": True}
    if start_role is not None:
        extra_body["start_role"] = start_role
    adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
    runtime = await adapter.prepare_runtime_config(
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
        "temperature": 0.7,
        "top_k": 32,
    }
    assert runtime["duplexio_text_sampling"] == {
        "mode": "argmax",
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
    }
    assert runtime["duplexio_emit_temperatures"] == {
        "user": 0.0,
        "agent": 1.0,
        "tool_call": 1.0,
    }
    assert runtime["duplexio_suppressed_token_ids"] == [11, 12]
    assert runtime["duplexio_agent_suppressed_token_ids"] == [18, 19]


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
        adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
        await adapter.prepare_runtime_config(
            config,
            model_config=model_config,
        )

    assert exc_info.value.code == "voice_not_found"


@pytest.mark.asyncio
async def test_serving_config_applies_client_sampling_parameters(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "voices.json").write_text(
        json.dumps({"default_voice": "voice-a", "voices": {"voice-a": {}}})
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(
            default_voice="voice-a",
            default_system_prompt="system",
            initial_agent_prefix="agent",
            initial_user_prefix="user",
            pad_token_id=11,
            silence_token_id=13,
            rollout_sampling_config={
                "mode": "top_p",
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
            },
            depth_transformer_config={
                "sampling_temperature": 0.8,
                "sampling_top_k": 250,
            },
            quantized_audio_config={"codebook_size": 2_048},
        ),
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: [1],
        decode=lambda token_ids, skip_special_tokens: "decoded",
        all_special_ids=[11],
    )
    monkeypatch.setattr(
        "vllm_omni.experimental.fullduplex.duplexio.serving_adapter."
        "cached_tokenizer_from_config",
        lambda config: tokenizer,
    )
    adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
    runtime = await adapter.prepare_runtime_config(
        DuplexSessionConfig(
            modalities=["text", "audio"],
            extra_body={
                "full_duplex": True,
                "duplexio_sampling": {
                    "seed": 1234,
                    "text": {
                        "mode": "top_k",
                        "temperature": 0.7,
                        "top_k": 12,
                        "top_p": 0.9,
                    },
                    "audio": {"temperature": 0.75, "top_k": 64},
                    "emit": {
                        "user": 0.2,
                        "agent": 0.4,
                        "tool_call": 0.8,
                    },
                },
            },
        ),
        model_config=model_config,
    )

    assert runtime["duplexio_sampling_seed"] == 1234
    assert runtime["duplexio_text_sampling"] == {
        "mode": "top_k",
        "temperature": 0.7,
        "top_k": 12,
        "top_p": 0.9,
    }
    assert runtime["duplexio_depth_sampling"] == {
        "temperature": 0.75,
        "top_k": 64,
    }
    assert runtime["duplexio_emit_temperatures"] == {
        "user": 0.2,
        "agent": 0.4,
        "tool_call": 0.8,
    }


def test_serving_config_rejects_invalid_client_sampling() -> None:
    with pytest.raises(DuplexIOClientRuntimeConfigError) as exc_info:
        parse_client_sampling_config(
            {"duplexio_sampling": {"text": {"top_p": 0.0}}}
        )

    assert exc_info.value.code == "invalid_sampling"


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
        adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
        await adapter.prepare_runtime_config(
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


def test_duplexio_initial_payloads_seed_the_full_silent_prompt() -> None:
    adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
    payloads = adapter.initial_data_plane_payloads(
        SimpleNamespace(
            runtime_config={"duplexio_system_token_ids": [1, 2, 3]},
            turn_id=4,
        )
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["audio"] == ""
    assert payload["frame_size"] == DUPLEXIO_FRAME_SIZE
    assert payload["frame_count"] == 3
    assert payload["valid_samples"] == 3 * DUPLEXIO_FRAME_SIZE
    assert payload["duplexio_prefill_final"] is True
    assert payload["duplex_turn_id"] == 4


def test_duplexio_initial_payloads_chunk_long_system_prompts() -> None:
    adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
    payloads = adapter.initial_data_plane_payloads(
        SimpleNamespace(
            runtime_config={"duplexio_system_token_ids": list(range(260))},
            turn_id=4,
        )
    )

    assert [payload["frame_count"] for payload in payloads] == [128, 128, 4]
    assert [payload["duplexio_prefill_final"] for payload in payloads] == [
        False,
        False,
        True,
    ]
    assert sum(payload["valid_samples"] for payload in payloads) == (
        260 * DUPLEXIO_FRAME_SIZE
    )


def test_duplexio_tool_result_payloads_feed_consecutive_system_tokens() -> None:
    adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
    encoded_text = []

    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        encoded_text.append((text, add_special_tokens))
        return [41, 42, 43]

    adapter.tokenizer = SimpleNamespace(
        encode=encode,
    )
    adapter.data_plane.configure_text_decoder(
        lambda token_ids: "",
        silence_token_id=13,
    )

    payloads = adapter.tool_result_data_plane_payloads(
        SimpleNamespace(runtime_config={}, turn_id=5),
        '{"time":"18:30"}',
    )

    assert encoded_text == [
        ('<tool_response>\n{"time":"18:30"}\n</tool_response>', False),
    ]
    assert len(payloads) == 1
    assert payloads[0]["frame_count"] == 3
    assert payloads[0]["audio"] == ""
    assert payloads[0]["duplexio_system_token_ids"] == [41, 42, 43]
    assert payloads[0]["duplexio_system_input"] is True
    assert payloads[0]["duplexio_system_input_final"] is True


def test_duplexio_tool_result_payloads_chunk_long_results() -> None:
    adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
    adapter.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: list(range(260)),
    )
    adapter.data_plane.configure_text_decoder(
        lambda token_ids: "",
        silence_token_id=13,
    )

    payloads = adapter.tool_result_data_plane_payloads(
        SimpleNamespace(runtime_config={}, turn_id=5),
        "large result",
    )

    assert [payload["frame_count"] for payload in payloads] == [128, 128, 4]
    assert [payload["duplexio_system_input_final"] for payload in payloads] == [
        False,
        False,
        True,
    ]
    assert [
        token_id
        for payload in payloads
        for token_id in payload["duplexio_system_token_ids"]
    ] == list(range(260))


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
        adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
        await adapter.prepare_runtime_config(
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
        adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
        await adapter.prepare_runtime_config(
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
        adapter = DuplexIOServingRuntimeAdapter(lambda *_args: None)
        await adapter.prepare_runtime_config(
            DuplexSessionConfig(
                modalities=["text"],
                extra_body={"full_duplex": True},
            ),
            model_config=model_config,
        )

    assert exc_info.value.code == "voice_required"
