# SPDX-License-Identifier: Apache-2.0
"""Follow real preprocess/sample/output feedback across staged policy commits."""

import base64
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import torch
from torch import nn

from tests.model_executor.models.duplexio.test_bulk_prefill import CountingCodec, model_fixture, request_state
from tests.model_executor.models.duplexio.test_user_heads import LocalVocabulary
from vllm_omni.experimental.fullduplex.duplexio.policy_receiver import PolicyWeightReceiver
from vllm_omni.experimental.fullduplex.duplexio.trajectory import TrajectoryRecorder
from vllm_omni.model_executor.models.duplexio.text_sampling import content_distribution


class AudioSampler(nn.Module):
    def sample(self, conditions, text, **kwargs):
        return torch.tensor([[3, 4, 5]])


class Codec(CountingCodec):
    def decode(self, codes, state):
        return torch.zeros(1, 1, 1920)


def feedback_model():
    model = model_fixture()
    model.audio_codec = Codec()
    model.audio_sampler = AudioSampler()
    model.policy_version = 0
    model.llm.base_model.lm_head = nn.Linear(11, 32, bias=False)
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
        "duplexio_text_sampling": {"mode": "top_k", "temperature": 1.0, "top_k": 1, "top_p": 1.0},
        "duplexio_emit_temperatures": {"user": 1.0, "agent": 0.0, "tool_call": 0.0},
        "duplexio_depth_sampling": {"temperature": 0.7, "top_k": 1},
    }


@torch.inference_mode()
def test_feedback_actions_and_versions_span_staging_then_commit(monkeypatch):
    model = feedback_model()
    state = request_state(model)
    recorder = TrajectoryRecorder()
    consumed_user_ids = []

    def step(*, prefix=False, final=False):
        nonlocal state
        frames = 3 if prefix else 1
        info = {
            "duplexio_model_state": state,
            "embed": {"speech_feat": torch.ones(1, 8)},
            "duplex": {
                "frame_count": frames,
                "runtime_config": runtime(),
                "payload": {"format": "duplexio_features"},
                "decode_audio": False,
                "final": final,
                "duplexio_prefill": prefix,
                "duplexio_prefill_final": prefix,
            },
        }
        model.preprocess_batch(req_ids=["one"], model_intermediate_buffer={"one": info}, device=torch.device("cpu"))
        _, _, updates = model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info)
        info.update(updates)
        output = model.make_omni_output(
            torch.randn(frames * 6, 11),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, frames * 6)],
            request_sample_eligible=[True],
        )
        payload = {name: values[0] for name, values in output.multimodal_outputs.items()}
        recorder.append(payload, version=model.policy_version)
        consumed_user_ids.append(updates["duplexio_replay"]["text_ids"][-1, 1].item())
        state = model.postprocess(None, **info)["duplexio_model_state"]
        return payload

    first = step(prefix=True)
    assert first["user_token_id"].item() == 7
    step()
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda _: None)
    receiver = PolicyWeightReceiver()
    receiver.device = torch.device("cpu")
    receiver.model_runner = SimpleNamespace(model=model)
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

    def broadcast(tensor):
        received.set()
        assert release.wait(5)
        tensor.copy_(next(source)[1])

    receiver.policy_group = SimpleNamespace(broadcast=broadcast)
    try:
        receiver.start_policy_weight_update(1, [[name, "float32", list(p.shape)] for name, p in pushed])
        assert received.wait(5)
        assert step()["user_token_id"].item() == 7
    finally:
        release.set()
        receiver._policy_pending[1].result(timeout=5)
    receiver.commit_policy_weight_update(1)
    assert step()["user_token_id"].item() == 2
    step(final=True)
    trace = recorder.tensors()
    assert consumed_user_ids == [2, 7, 7, 7, 2]
    assert trace["prediction_rows"].tolist() == [2, 3, 4, 5, 6]
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
    state.system_token_offset = 3
    state.text_input_ids[1] = 7
    payload = {"format": "pcm_f32le", "audio": base64.b64encode(torch.zeros(1920).numpy().tobytes()).decode()}
    _, _, updates = model.preprocess(
        torch.zeros(6, dtype=torch.long),
        None,
        duplexio_model_state=state,
        duplex={"frame_count": 1, "runtime_config": runtime(), "payload": payload},
    )
    # CountingASR has an encoder only: calling any RNN-T transcript code fails.
    assert updates["duplexio_replay"]["text_ids"][0, 1].item() == 7
    torch.testing.assert_close(updates["duplexio_replay"]["user_features"], torch.ones(1, 8))
