# SPDX-License-Identifier: Apache-2.0
"""Follow real preprocess/sample/output feedback across staged policy commits."""

import base64

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

pytest.importorskip("duplexio")

from duplexio.rollout_trajectory import TrajectoryRecorder

from tests.model_executor.models.duplexio.test_bulk_prefill import CountingCodec, model_fixture, request_state
from tests.model_executor.models.duplexio.test_user_heads import LocalVocabulary
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.duplexio.duplex import DuplexIODuplexPlugin
from vllm_omni.model_executor.models.duplexio.frame_output import frame_fields
from vllm_omni.outputs.mm_outputs import MultimodalPayload


class AudioSampler(nn.Module):
    def sample(self, conditioning, noise, temperature):
        return noise


class Codec(CountingCodec):
    def decode(self, codes, state):
        raise AssertionError("decode_audio=False must not run the codec decoder")


def feedback_model():
    model = model_fixture()
    model.audio_codec = Codec()
    model.audio_sampler = AudioSampler()
    model.policy_version = 0
    model.lm_head = nn.Linear(11, 32, bias=False)
    model.lm_head.quant_method = UnquantizedEmbeddingMethod()
    model.output_head_proj = nn.ModuleDict({name: nn.Linear(11, 11) for name in ("agent", "tool_call")})
    model.user_token_projection = nn.Linear(66, 11)
    model.user_emit_head = nn.Linear(66, 1)
    model.agent_emit_head = nn.Linear(66, 1)
    model.tool_call_emit_head = nn.Linear(66, 1)
    model.logits_processor = LocalVocabulary()
    model.text_config.vocab_size = 32
    for stream in ("agent", "tool", "user"):
        setattr(model, f"{stream}_suppressed_token_ids", torch.tensor([2]))
    model.init_text_sampling(32, 4)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[7, 0] = 1
        model.user_token_projection.weight.zero_()
        model.user_token_projection.bias.zero_()
        model.user_token_projection.bias[0] = 1
        model.user_emit_head.weight.zero_()
        model.user_emit_head.bias.fill_(100)
    return model


def runtime():
    return {
        "duplexio_record_inputs": True, "duplexio_scheduler_token_id": 1,
        "duplexio_tool_generation": 0, "duplexio_tool_results": [],
    }


def feedback_state(model):
    """Greedy agent and tool streams; the user stream draws its one top token."""
    state = request_state(model)
    content = {"temperature": 1.0, "top_k": 1, "top_p": 1.0}
    state.sampling = model.resolve_sampling({
        "agent": {"content": content}, "user": {"emission": {"temperature": 1.0}, "content": content},
    })
    return state


@torch.inference_mode()
def test_feedback_actions_and_versions_follow_a_weight_update():
    model = feedback_model()
    state = feedback_state(model)
    recorder = TrajectoryRecorder()
    consumed_user_ids = []

    def step(*, prefix=False, final=False):
        nonlocal state
        frames = 6 if prefix else 1
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
                "duplexio_prefix": prefix,
                "duplexio_tool_token_ids": [],
                "duplexio_tool_generation": 0,
                "duplexio_given_frame": None,
                "epoch": 0,
                "turn_id": 0,
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
        payload = MultimodalPayload.from_dict({name: values[0] for name, values in flatten_payload(model.finalize_multimodal_outputs_from_cpu_snapshot(output.multimodal_outputs)).items()})
        assert {"user_action_eligible", "user_token_eligible", "user_action_logprob"}.isdisjoint(payload)
        recorder.append(payload, version=model.policy_version)
        consumed_user_ids.append(updates["duplexio_replay"]["text_ids"][-1, 1].item())
        state = model.postprocess(None, **info)["duplexio_model_state"]
        return payload

    first = step(prefix=True)
    assert frame_fields(first)["user_token_id"] == 7
    step()
    pushed = [(name, p.clone()) for name, p in model.named_parameters() if name.startswith("user_")]
    for name, value in pushed:
        if name == "user_emit_head.bias":
            value.fill_(-100)
    pointers = {name: model.get_parameter(name).data_ptr() for name, _ in pushed}
    assert frame_fields(step())["user_token_id"] == 7
    model.load_weights(pushed)
    model.set_policy_version(1)
    assert frame_fields(step())["user_token_id"] == 2
    step(final=True)
    trace = recorder.tensors()
    assert consumed_user_ids == [2, 7, 7, 7, 2]
    assert trace["prediction_rows"].tolist() == [5, 6, 7, 8, 9]
    assert trace["sampled_user_ids"].tolist() == [7, 7, 7, 2, 2]
    assert trace["sampled_user_emits"].tolist() == [True, True, True, False, False]
    assert trace["row_versions"].tolist() == [0, 0, 0, 1, 1]
    assert trace["user_action_eligible"].tolist() == [True, True, True, True, False]
    assert trace["user_token_eligible"].tolist() == [True, True, True, False, False]
    assert {name: model.get_parameter(name).data_ptr() for name, _ in pushed} == pointers


@torch.inference_mode()
def test_raw_audio_keeps_encoder_features_without_transcript_commits():
    model = feedback_model()
    state = feedback_state(model)
    state.text_input_ids = (state.text_input_ids[0], 7, *state.text_input_ids[2:])
    plan = DuplexIODuplexPlugin(lambda *args: None).plan_append(
        request_id="raw-frame", fence=DuplexFence("session"),
        session_config={}, runtime_config=runtime(), seq=2, turn_seq=1, final=False, sampling_params=None,
        payload={
            "format": "pcm_f32le", "sample_rate_hz": 24000,
            "audio": base64.b64encode(torch.zeros(1920).numpy().tobytes()).decode(),
        },
    )
    _, _, updates = model.preprocess(
        torch.zeros(6, dtype=torch.long),
        None,
        duplexio_model_state=state,
        duplex_token_offset=0, duplex_prompt_len=6,
        **plan.prompt["model_intermediate_buffer"],
    )
    # CountingASR has an encoder only: calling any RNN-T transcript code fails.
    assert updates["duplexio_replay"]["text_ids"][0, 1].item() == 7
    torch.testing.assert_close(updates["duplexio_replay"]["user_features"], torch.ones(1, 8))


@torch.inference_mode()
def test_chunked_appends_record_all_rows_and_sample_only_at_their_end():
    model = feedback_model()
    state = feedback_state(model)
    recorder = TrajectoryRecorder()
    for prefix in (True, False):
        prompt_len = (state.frames_seen + 6) * 6
        for index, frames in enumerate((1, 2, 3)):
            info = {
                "duplexio_model_state": state,
                "duplex_token_offset": state.frames_seen * 6,
                "duplex_prompt_len": prompt_len,
                "duplex": {"frame_count": 6, "duplexio_prefix": prefix, "decode_audio": False,
                           "duplexio_tool_token_ids": [] if prefix else [6, 7, 8, 9, 10], "duplexio_tool_generation": 0,
                           "duplexio_given_frame": None, "epoch": 0, "turn_id": 0, "final": False,
                           "pcm": torch.zeros(1920).numpy().tobytes(), "runtime_config": runtime()},
            }
            keys = state.persistent_keys
            _, _, updates = model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info)
            if not prefix:
                # The live frame sees the pinned prompt and all earlier text.
                assert keys >= 2
                assert updates["duplexio"]["persistent_last"][0].item() == keys
            info.update(updates)
            output = model.make_omni_output(
                torch.randn(frames * 6, 11), model_intermediate_buffer=[info],
                request_token_spans=[(0, frames * 6)], request_sample_eligible=[index == 2],
            )
            payload = MultimodalPayload.from_dict({name: values[0] for name, values in flatten_payload(model.finalize_multimodal_outputs_from_cpu_snapshot(output.multimodal_outputs)).items()})
            recorder.append(payload, version=0)
            state = model.postprocess(None, **info)["duplexio_model_state"]
            fields = frame_fields(payload)
            assert fields["predicted"] == (index == 2)
            assert fields["prefix"] == prefix and fields["tool_result"] != prefix
            if index < 2:
                assert payload["agent_audio_token_ids"].numel() == 0
            else:
                assert payload["agent_audio_token_ids"].numel() > 0
    trace = recorder.tensors()
    assert trace["text_ids"].shape[0] == 12
    assert trace["text_ids"][7:, 0].tolist() == [6, 7, 8, 9, 10]
    assert trace["prediction_rows"].tolist() == [5, 11]
    assert trace["prompt_frames"].tolist() == [True, True] + [False] * 10
