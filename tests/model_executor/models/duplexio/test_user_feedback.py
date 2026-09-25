# SPDX-License-Identifier: Apache-2.0
"""Follow real preprocess/sample/output feedback across staged policy commits."""

import base64
import copy
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

pytest.importorskip("duplexio")

from duplexio.rollout_policy import PolicyWeightReceiver
from duplexio.rollout_trajectory import TrajectoryRecorder

from tests.model_executor.models.duplexio.test_bulk_prefill import CountingCodec, model_fixture, request_state
from tests.model_executor.models.duplexio.test_user_heads import LocalVocabulary
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.experimental.fullduplex.duplexio.runtime import build_duplexio_data_plane_prompt
from vllm_omni.experimental.fullduplex.engine.contracts import DuplexInputMode
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.model_executor.models.duplexio.frame_output import frame_fields
from vllm_omni.model_executor.models.duplexio.text_sampling import content_distribution
from vllm_omni.outputs.mm_outputs import MultimodalPayload


class AudioSampler(nn.Module):
    def sample(self, conditions, text, **kwargs):
        return torch.tensor([[3, 4, 5]])


class Codec(CountingCodec):
    def decode(self, codes, state):
        raise AssertionError("decode_audio=False must not run the codec decoder")


def feedback_model():
    model = model_fixture()
    model.audio_codec = Codec()
    model.audio_sampler = AudioSampler()
    model.policy_version = 0
    model.llm.base_model.lm_head = nn.Linear(11, 32, bias=False)
    model.llm.base_model.lm_head.quant_method = UnquantizedEmbeddingMethod()
    model.llm.output_head_proj = nn.ModuleDict({name: nn.Linear(11, 11) for name in ("agent", "tool_call")})
    model.user_token_projection = nn.Linear(66, 11)
    model.user_emit_head = nn.Linear(66, 1)
    model.agent_emit_head = nn.Linear(66, 1)
    model.tool_call_emit_head = nn.Linear(66, 1)
    model.logits_processor = LocalVocabulary()
    model.content_distribution = content_distribution
    for stream in ("agent", "tool", "user"):
        setattr(model, f"{stream}_suppressed_token_ids", torch.tensor([2]))
    with torch.no_grad():
        model.llm.base_model.lm_head.weight.zero_()
        model.llm.base_model.lm_head.weight[7, 0] = 1
        model.user_token_projection.weight.zero_()
        model.user_token_projection.bias.zero_()
        model.user_token_projection.bias[0] = 1
        model.user_emit_head.weight.zero_()
        model.user_emit_head.bias.fill_(100)
    return model


def runtime():
    return {
        "duplexio_record_inputs": True,
        "duplexio_text_sampling": {"temperature": 1.0, "top_k": 1, "top_p": 1.0},
        "duplexio_user_sampling": {"content": {"temperature": 1.0, "top_k": 1, "top_p": 1.0}},
        "duplexio_emit_temperatures": {"user": 1.0, "agent": 0.0, "tool_call": 0.0},
        "duplexio_depth_sampling": {"temperature": 0.7, "top_k": 1},
    }


@torch.inference_mode()
def test_sampling_cache_reuses_settings_and_tracks_session_updates():
    model = feedback_model()
    state = request_state(model)
    settings = runtime()
    info = {"duplexio_working_state": state, "duplex": {"runtime_config": settings}}
    logits, emissions = torch.zeros(1, 3, 32), torch.zeros(1, 3)
    model.sample_text_batch(logits, emissions, [info])
    initial = state.sampling
    assert initial.agent.suppressed_token_ids is model.agent_suppressed_token_ids
    assert initial.tool.suppressed_token_ids is model.tool_suppressed_token_ids
    assert initial.user.suppressed_token_ids is model.user_suppressed_token_ids

    # Appends arrive with fresh wire dictionaries, including unrelated metadata.
    settings = copy.deepcopy(settings)
    settings["duplexio_record_hiddens"] = True
    info["duplex"]["runtime_config"] = settings
    model.sample_text_batch(logits, emissions, [info])
    assert state.sampling is initial

    working = state.fork()
    info["duplexio_working_state"] = working
    settings["duplexio_text_sampling"]["temperature"] = 0.3
    settings["duplexio_user_sampling"]["content"]["temperature"] = 0.0
    settings["duplexio_emit_temperatures"]["user"] = 0.0
    settings["duplexio_depth_sampling"]["top_k"] = 2
    model.sample_text_batch(logits, emissions, [info])
    updated = working.sampling
    assert updated is not initial
    assert updated.agent.temperature == updated.tool.temperature == 0.3
    assert updated.user.temperature == updated.emission.user == 0.0
    assert updated.depth.top_k == 2
    assert state.sampling is initial
    assert initial.agent.temperature == initial.user.temperature == initial.emission.user == 1.0
    model.sample_text_batch(logits, emissions, [info])
    assert working.sampling is updated
    settings["duplexio_text_sampling"]["temperature"] = 0.9
    model.sample_text_batch(logits, emissions, [info])
    assert working.sampling.agent.temperature == 0.9
    assert updated.agent.temperature == 0.3
    valid = working.sampling
    settings["duplexio_emit_temperatures"]["user"] = -1.0
    with pytest.raises(ValueError):
        model.sample_text_batch(logits, emissions, [info])
    assert working.sampling is valid


@torch.inference_mode()
def test_feedback_actions_and_versions_span_staging_then_commit(monkeypatch):
    model = feedback_model()
    state = request_state(model)
    recorder = TrajectoryRecorder()
    consumed_user_ids = []

    def step(*, prefix=False, final=False):
        nonlocal state
        frames = 5 if prefix else 1
        info = {
            "duplexio_model_state": state,
            "duplex_token_offset": state.frames_seen * 6,
            "duplex_prompt_len": (state.frames_seen + frames) * 6,
            "duplex": {
                "frame_count": frames,
                "runtime_config": runtime(),
                "pcm": torch.ones(1920).numpy().tobytes(),
                "decode_audio": False,
                "final": final,
                "duplexio_prefill": prefix,
            },
        }
        _, _, updates = model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info)
        info.update(updates)
        output = model.make_omni_output(
            torch.randn(frames * 6, 11),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, frames * 6)],
            request_sample_eligible=[True],
        )
        payload = MultimodalPayload.from_dict({name: values[0] for name, values in flatten_payload(model.finalize_omni_output(output.multimodal_outputs)).items()})
        assert {"user_action_eligible", "user_token_eligible", "user_action_logprob"}.isdisjoint(payload)
        recorder.append(payload, version=model.policy_version)
        consumed_user_ids.append(updates["duplexio_replay"]["text_ids"][-1, 1].item())
        state = model.postprocess(None, **info)["duplexio_model_state"]
        return payload

    first = step(prefix=True)
    assert frame_fields(first)["user_token_id"] == 7
    step()
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda _: None)
    receiver = PolicyWeightReceiver()
    receiver.device = torch.device("cpu")
    receiver.model_runner = SimpleNamespace(model=model, get_model=lambda: model)
    receiver._policy_stream = SimpleNamespace(synchronize=lambda: None)
    receiver._policy_pending = receiver._policy_plan = receiver._policy_version = None
    receiver._policy_buffer = []
    receiver._policy_commit_failed = False
    pushed = [(name, p.clone()) for name, p in model.named_parameters() if name.startswith("user_")]
    for name, value in pushed:
        if name == "user_emit_head.bias":
            value.fill_(-100)
    pointers = {name: model.get_parameter(name).data_ptr() for name, _ in pushed}
    source = iter(pushed)
    received, release = Event(), Event()

    def receive_views(views):
        def wait():
            received.set()
            assert release.wait(5)
            for tensor, _ in views:
                tensor.copy_(next(source)[1])

        return SimpleNamespace(wait=wait)

    receiver.policy_group = SimpleNamespace(rank=0, receive_views=receive_views)
    try:
        receiver.start_policy_weight_update(1, [
            [name, "float32", list(p.shape), [[0, 0, 0, p.shape[0] if p.ndim else 1]]]
            for name, p in pushed
        ])
        assert received.wait(5)
        assert frame_fields(step())["user_token_id"] == 7
    finally:
        release.set()
        receiver._policy_pending[1].result(timeout=5)
    receiver.commit_policy_weight_update(1)
    assert frame_fields(step())["user_token_id"] == 2
    step(final=True)
    trace = recorder.tensors()
    assert consumed_user_ids == [2, 7, 7, 7, 2]
    assert trace["prediction_rows"].tolist() == [4, 5, 6, 7, 8]
    assert trace["sampled_user_ids"].tolist() == [7, 7, 7, 2, 2]
    assert trace["sampled_user_emits"].tolist() == [True, True, True, False, False]
    assert trace["row_versions"].tolist() == [0, 0, 0, 1, 1]
    assert trace["user_action_eligible"].tolist() == [True, True, True, True, False]
    assert trace["user_token_eligible"].tolist() == [True, True, True, False, False]
    assert {name: model.get_parameter(name).data_ptr() for name, _ in pushed} == pointers


@torch.inference_mode()
def test_raw_audio_keeps_encoder_features_without_transcript_commits():
    model = feedback_model()
    state = request_state(model)
    state.text_input_ids[1] = 7
    prompt = build_duplexio_data_plane_prompt(
        request_id="raw-frame", fence=DuplexFence("session"),
        session_config={}, runtime_config=runtime(), seq=1, turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK, final=False,
        payload={
            "format": "pcm_f32le", "sample_rate_hz": 24000,
            "frame_size": 1920, "frame_count": 1, "valid_samples": 1920,
            "audio": base64.b64encode(torch.zeros(1920).numpy().tobytes()).decode(),
        },
    )
    _, _, updates = model.preprocess(
        torch.zeros(6, dtype=torch.long),
        None,
        duplexio_model_state=state,
        duplex_token_offset=0, duplex_prompt_len=6,
        **prompt["model_intermediate_buffer"],
    )
    # CountingASR has an encoder only: calling any RNN-T transcript code fails.
    assert updates["duplexio_replay"]["text_ids"][0, 1].item() == 7
    torch.testing.assert_close(updates["duplexio_replay"]["user_features"], torch.ones(1, 8))


@torch.inference_mode()
def test_chunked_context_records_all_rows_and_samples_only_at_boundary():
    model = feedback_model()
    state = request_state(model)
    recorder = TrajectoryRecorder()
    for context in ("prefill", "system_input"):
        prompt_len = (state.frames_seen + 5) * 6
        for index, frames in enumerate((1, 2, 2)):
            info = {
                "duplexio_model_state": state,
                "duplex_token_offset": state.frames_seen * 6,
                "duplex_prompt_len": prompt_len,
                "duplex": {"frame_count": 5, f"duplexio_{context}": True, "decode_audio": False,
                           "duplexio_system_token_ids": [6, 7, 8, 9, 10], "runtime_config": runtime()},
            }
            keys = state.persistent_keys
            _, _, updates = model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info)
            if context == "system_input":
                # System input sees the pinned prompt and all earlier text.
                assert keys >= 2
                assert updates["duplexio"]["persistent_last"][0].item() == keys
            info.update(updates)
            output = model.make_omni_output(
                torch.randn(frames * 6, 11), model_intermediate_buffer=[info],
                request_token_spans=[(0, frames * 6)], request_sample_eligible=[index == 2],
            )
            payload = MultimodalPayload.from_dict({name: values[0] for name, values in flatten_payload(model.finalize_omni_output(output.multimodal_outputs)).items()})
            recorder.append(payload, version=0)
            state = model.postprocess(None, **info)["duplexio_model_state"]
            assert frame_fields(payload)[f"duplex_{context}_complete"] == (index == 2)
            if index < 2:
                assert payload["agent_audio_token_ids"].numel() == 0
            else:
                assert payload["agent_audio_token_ids"].numel() > 0
    trace = recorder.tensors()
    assert trace["text_ids"].shape[0] == 10
    assert trace["text_ids"][5:, 0].tolist() == [6, 7, 8, 9, 10]
    assert trace["prediction_rows"].tolist() == [4, 9]
    assert trace["prompt_frames"].tolist() == [True, True] + [False] * 8
