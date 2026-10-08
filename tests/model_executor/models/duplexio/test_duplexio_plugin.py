# SPDX-License-Identifier: Apache-2.0
"""DuplexIO's duplex plugin: session runtime config, append planning and output projection."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import AdditionalInformationPayload
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext, DuplexRuntimeConfigError
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.model_executor.models.duplexio.duplex import DuplexIODuplexPlugin
from vllm_omni.model_executor.models.duplexio.frame_output import frame_fields, pack_frame

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PLUGIN_MODULE = "vllm_omni.model_executor.models.duplexio.duplex.plugin"
REFERENCE_FRAMES = 3
REQUEST_ID = duplex_resource_request_id(DuplexFence("session", epoch=0, turn_id=0), "stage0")


def pcm(frames: float, value: float = 0.25) -> bytes:
    return torch.full((round(frames * 1920),), value).numpy().tobytes()


def reference_audio(frames: float = REFERENCE_FRAMES) -> str:
    """Base64 pcm_f32le reference audio: the voice, as the client supplies it."""
    return base64.b64encode(pcm(frames)).decode()


def frame_payload() -> dict[str, object]:
    return {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 24000,
        "audio": base64.b64encode(pcm(1, 0.5)).decode(),
    }


@pytest.fixture
def byte_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        models.BPE(vocab={f"<0x{byte:02X}>": byte for byte in range(256)}, merges=[], byte_fallback=True)
    )
    tokenizer.decoder = decoders.ByteFallback()
    tokenizer.add_special_tokens(["<special>"])
    return tokenizer


@pytest.fixture
def model_config(monkeypatch, byte_tokenizer):
    encoded = {
        "system": [3, 4],
        "<|im_start|>user\n": [5, 6],
        "<|im_start|>assistant\n": [7, 8],
        "<tool_response>\nsunny\n</tool_response>": [20, 21, 22],
        "<tool_response>\nrainy\n</tool_response>": [23, 24],
    }
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: encoded[text],
        backend_tokenizer=byte_tokenizer,
    )
    monkeypatch.setattr(f"{PLUGIN_MODULE}.cached_tokenizer_from_config", lambda config: tokenizer)
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            voice_prompt_max_frames=125,
            default_system_prompt="system",
            initial_agent_prefix="<|im_start|>assistant\n",
            initial_user_prefix="<|im_start|>user\n",
            pad_token_id=11,
            silence_token_id=257,
            depth_transformer_config={"sampling_temperature": 0.9, "sampling_top_k": 32},
            quantized_audio_config={"codebook_size": 64},
        ),
    )


def session_config(**extra_body: object) -> DuplexSessionConfig:
    return DuplexSessionConfig(extra_body={"auto_response": True, "ref_audio_data": reference_audio(), **extra_body})


async def open_session(model_config, **extra_body: object) -> tuple[DuplexIODuplexPlugin, dict[str, object]]:
    plugin = DuplexIODuplexPlugin(lambda audio, rate, fmt, speed: f"{fmt}:{rate}:{audio.numel()}")
    return plugin, await plugin.prepare_runtime_config(session_config(**extra_body), model_config=model_config)


def plan(plugin: DuplexIODuplexPlugin, runtime: dict[str, object], seq: int, request_id: str = REQUEST_ID) -> dict:
    return plugin.plan_append(
        request_id=request_id,
        fence=DuplexFence("session", epoch=0, turn_id=0),
        session_config={},
        runtime_config=runtime,
        seq=seq,
        turn_seq=seq,
        payload=frame_payload(),
        final=False,
        sampling_params=None,
    ).prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(("start_role", "prefix_ids"), [(None, [5, 6]), ("user", [5, 6]), ("agent", [7, 8])])
async def test_runtime_config_pins_the_reference_audio_and_prompt(model_config, start_role, prefix_ids) -> None:
    extra = {} if start_role is None else {"start_role": start_role}
    _, runtime = await open_session(model_config, **extra)
    assert runtime["duplexio_voice_prompt_pcm"] == pcm(REFERENCE_FRAMES)
    assert runtime["duplexio_voice_prompt_frames"] == REFERENCE_FRAMES
    assert runtime["duplexio_system_token_ids"] == [3, 4, *prefix_ids]
    assert runtime["duplexio_start_role"] == (start_role or "user")
    assert runtime["duplexio_scheduler_token_id"] == 11
    assert isinstance(runtime["duplexio_sampling_seed"], int)
    assert runtime["duplexio_depth_sampling"] == {"temperature": 0.7, "top_k": 32}
    assert runtime["duplexio_text_sampling"] == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    assert runtime["duplexio_emit_temperatures"] == {"user": 0.0, "agent": 1.0, "tool_call": 1.0}
    # The runtime config crosses the engine boundary with every append.
    wire = serialize_additional_information({"duplex": {"runtime_config": runtime}})
    decoded = MsgpackDecoder(AdditionalInformationPayload).decode(MsgpackEncoder().encode(wire))
    assert deserialize_additional_information(decoded)["duplex"]["runtime_config"] == runtime


@pytest.mark.asyncio
async def test_runtime_config_applies_client_sampling(model_config) -> None:
    _, runtime = await open_session(
        model_config,
        duplexio_sampling={
            "seed": 9,
            "agent": {"content": {"temperature": 0.2}},
            "audio": {"temperature": 0.5, "top_k": 4},
        },
    )
    assert runtime["duplexio_sampling_seed"] == 9
    assert runtime["duplexio_text_sampling"]["temperature"] == 0.2
    assert runtime["duplexio_depth_sampling"] == {"temperature": 0.5, "top_k": 4}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_body", "code"),
    [
        ({"ref_audio_data": None}, "ref_audio_required"),
        ({"ref_audio_data": reference_audio(0.5)}, "ref_audio_too_short"),
        ({"ref_audio_path": "/voices/a.wav"}, "ref_audio_path_rejected"),
        ({"auto_response": False}, "full_duplex_required"),
        ({"start_role": "narrator"}, "start_role_invalid"),
        ({"duplexio_sampling": {"seed": -1}}, "invalid_sampling"),
        ({"duplexio_sampling": {"audio": {"top_k": 65}}}, "invalid_sampling"),
    ],
)
async def test_runtime_config_rejects_invalid_sessions(model_config, extra_body, code) -> None:
    with pytest.raises(DuplexRuntimeConfigError) as error:
        await open_session(model_config, **extra_body)
    assert error.value.code == code


@pytest.mark.asyncio
async def test_flowmap_checkpoint_rejects_audio_sampling(model_config) -> None:
    model_config.hf_config.quantized_audio_config = {}
    model_config.hf_config.depth_transformer_config = {}
    with pytest.raises(DuplexRuntimeConfigError) as error:
        await open_session(model_config, duplexio_sampling={"audio": {"temperature": 0.5}})
    assert error.value.code == "invalid_sampling"


@pytest.mark.asyncio
async def test_runtime_config_rejects_server_owned_keys(model_config) -> None:
    with pytest.raises(DuplexRuntimeConfigError, match="duplexio_system_token_ids"):
        await open_session(model_config, duplexio_system_token_ids=[1])


@pytest.mark.asyncio
async def test_updates_change_only_the_text_temperature(model_config) -> None:
    plugin, runtime = await open_session(model_config, duplexio_sampling={"seed": 9})
    unchanged = plugin.runtime_config_for_update(session_config(duplexio_sampling={"seed": 9}), runtime)
    assert unchanged == runtime
    warmer = session_config(duplexio_sampling={"seed": 9})
    warmer.temperature = 0.9
    assert plugin.runtime_config_for_update(warmer, runtime)["duplexio_text_sampling"]["temperature"] == 0.9
    for config, code in [
        (session_config(duplexio_sampling={"seed": 10}), "sampling_update_unsupported"),
        (session_config(duplexio_sampling={"seed": 9}, start_role="agent"), "start_role_update_unsupported"),
        (session_config(duplexio_sampling={"seed": 9}, ref_audio_data=reference_audio(4)), "voice_update_unsupported"),
    ]:
        with pytest.raises(DuplexRuntimeConfigError) as error:
            plugin.runtime_config_for_update(config, runtime)
        assert error.value.code == code
    instructed = session_config(duplexio_sampling={"seed": 9})
    instructed.instructions = "be brief"
    with pytest.raises(DuplexRuntimeConfigError, match="instructions"):
        plugin.runtime_config_for_update(instructed, runtime)


def test_sampling_params_force_one_delta_token_per_append() -> None:
    plugin = DuplexIODuplexPlugin(lambda *args: None)
    default = SamplingParams(max_tokens=100)
    (configured,) = plugin.configure_sampling_params(runtime_config={}, defaults=(default,))
    assert configured.max_tokens == 1
    assert configured.output_kind == RequestOutputKind.DELTA
    assert default.max_tokens == 100


def test_capabilities_use_the_scheduler_data_plane() -> None:
    capabilities = DuplexIODuplexPlugin(lambda *args: None).capabilities(max_sessions=2)
    assert capabilities.supports_model_native_turn_policy
    assert capabilities.supports_core_resumable_request
    assert capabilities.requires_model_runner_kv
    assert capabilities.supports_multi_session
    assert capabilities.chunk_period_ms == 80
    assert not capabilities.supports_barge_in


@pytest.mark.asyncio
async def test_first_append_carries_the_prefix_before_its_live_frame(model_config) -> None:
    plugin, runtime = await open_session(model_config)
    first = plan(plugin, runtime, seq=1)
    duplex = first["model_intermediate_buffer"]["duplex"]
    prefix_frames = REFERENCE_FRAMES + 4
    assert first["prompt_token_ids"] == [11] * ((prefix_frames + 1) * 6)
    assert duplex["frame_count"] == prefix_frames + 1
    assert duplex["duplexio_prefix"]
    assert duplex["pcm"] == pcm(1, 0.5)
    assert "audio" not in duplex["payload"]
    later = plan(plugin, runtime, seq=2)
    assert later["prompt_token_ids"] == [11] * 6
    assert not later["model_intermediate_buffer"]["duplex"]["duplexio_prefix"]


@pytest.mark.asyncio
async def test_plan_rejects_anything_but_one_frame(model_config) -> None:
    plugin, runtime = await open_session(model_config)
    payload = {**frame_payload(), "audio": base64.b64encode(pcm(2)).decode()}
    with pytest.raises(ValueError, match="exactly 1920 samples"):
        plugin.plan_append(
            request_id="r",
            fence=DuplexFence("session"),
            session_config={},
            runtime_config=runtime,
            seq=2,
            turn_seq=2,
            payload=payload,
            final=False,
            sampling_params=None,
        )


@pytest.mark.asyncio
async def test_tool_results_follow_one_live_frame_once_then_retire(model_config) -> None:
    plugin, runtime = await open_session(model_config)
    for output in ("sunny", "rainy"):
        runtime = plugin.runtime_config_for_function_output(
            session_config(),
            runtime,
            {"type": "function_call_output", "call_id": "call_1", "output": output},
        )
    assert runtime["duplexio_tool_generation"] == 2
    first = plan(plugin, runtime, seq=2)
    duplex = first["model_intermediate_buffer"]["duplex"]
    assert duplex["duplexio_tool_token_ids"] == [20, 21, 22, 23, 24]
    assert duplex["duplexio_tool_generation"] == 2
    assert len(first["prompt_token_ids"]) == 6 * 6
    # A pipelined append planned before the worker reports them must not feed them again.
    assert plan(plugin, runtime, seq=3)["model_intermediate_buffer"]["duplex"]["duplexio_tool_token_ids"] == []
    # Another stage request (a new epoch) has not been fed them.
    assert (
        plan(
            plugin,
            runtime,
            seq=1,
            request_id=duplex_resource_request_id(DuplexFence("session", epoch=1, turn_id=0), "stage0"),
        )["model_intermediate_buffer"]["duplex"]["duplexio_tool_generation"]
        == 2
    )
    assert plugin.runtime_config_after_model_output(runtime, {"frame": pack_frame(tool_generation=0)}) is None
    retired = plugin.runtime_config_after_model_output(runtime, {"frame": pack_frame(tool_generation=2)})
    assert retired["duplexio_tool_results"] == []
    plugin.data_plane.close_session("session")
    assert not plugin.data_plane.planned_tool_generations


@pytest.mark.asyncio
async def test_tool_result_backlog_is_bounded(model_config) -> None:
    plugin, runtime = await open_session(model_config)
    for _ in range(8):
        runtime = plugin.runtime_config_for_function_output(
            session_config(), runtime, {"call_id": "c", "output": "sunny"}
        )
    with pytest.raises(DuplexRuntimeConfigError) as error:
        plugin.runtime_config_for_function_output(session_config(), runtime, {"call_id": "c", "output": "sunny"})
    assert error.value.code == "function_response_backlog"


def output(request_id: str = REQUEST_ID, *, text: str = "", audio: int = 0, tool_call: bytes = b"", **frame) -> object:
    metadata = {
        "frame": pack_frame(**{"predicted": True, "sample_rate_hz": 24000, "user_token_id": 257, **frame}),
        "audio": torch.zeros(audio),
        "tool_call_json": torch.tensor(list(tool_call), dtype=torch.uint8),
    }
    return SimpleNamespace(request_id=request_id, outputs=[SimpleNamespace(text=text)], multimodal_output=metadata)


def project(plugin: DuplexIODuplexPlugin, *outputs: object, epoch: int = 0) -> list[dict[str, object]]:
    context = DuplexDataPlaneContext(epoch=epoch, response_format="wav", modalities=("text", "audio"))
    return list(plugin.data_plane.project({"data_plane_outputs": list(outputs)}, context=context))


@pytest.mark.asyncio
async def test_data_plane_projects_one_frame_of_audio_and_text(model_config) -> None:
    plugin, _ = await open_session(model_config)
    (result,) = project(plugin, output(text="Hi", audio=1920, duplex_turn_id=3))
    assert result["text"] == "Hi"
    assert result["audio_data"] == "wav:24000:1920"
    assert result["audio_duration_ms"] == 80
    assert result["audio_text_mark"]
    assert result["model_turn_id"] == 3
    assert not result["is_listen"]
    assert project(plugin, output(text="Hi", audio=1920), epoch=1) == []
    context = DuplexDataPlaneContext(modalities=("text", "audio"))
    assert list(plugin.data_plane.project({"ok": True, "operation": "append"}, context=context)) == []
    assert project(plugin, output(text="Hi", predicted=False)) == []
    plugin.data_plane.mark_terminal(REQUEST_ID)
    assert project(plugin, output(text="Hi")) == []


@pytest.mark.asyncio
async def test_data_plane_reports_listen_and_silent_frames(model_config) -> None:
    plugin, _ = await open_session(model_config)
    (listen,) = project(plugin, output(model_listen=True))
    assert listen["is_listen"] and listen["model_listen"]
    # Delayed codec frames before the first waveform have nothing to show.
    assert project(plugin, output()) == []


@pytest.mark.parametrize("text", ["👋", "Ḥasan", "hello 👋 Ḥasan"])
@pytest.mark.asyncio
async def test_user_transcript_streams_whole_characters(model_config, text) -> None:
    plugin, _ = await open_session(model_config)
    deltas = [
        result["input_text_delta"]
        for token_id in text.encode()
        for result in project(plugin, output(user_token_id=token_id, model_listen=True))
    ]
    assert "".join(deltas) == text
    assert "�" not in "".join(deltas)
    plugin.data_plane.close_stream(REQUEST_ID)
    assert not plugin.data_plane.user_decoders


@pytest.mark.asyncio
async def test_data_plane_forwards_completed_tool_calls(model_config) -> None:
    plugin, _ = await open_session(model_config)
    call = json.dumps({"name": "weather", "arguments": {"city": "Aarhus"}, "sequence": 1}).encode()
    results = project(plugin, output(tool_call=call, audio=1920, tool_call_complete=True))
    assert results[0]["function_call"] is True
    assert results[0]["name"] == "weather"
    assert json.loads(results[0]["arguments"]) == {"city": "Aarhus"}
    assert results[0]["call_id"].startswith("call_")
    assert results[1]["audio_data"]


def test_frame_fields_round_trip_the_append_kind() -> None:
    fields = frame_fields({"frame": pack_frame(prefix=True, tool_result=True, tool_generation=4)})
    assert fields["prefix"] is True and fields["tool_result"] is True and fields["tool_generation"] == 4
