# SPDX-License-Identifier: Apache-2.0
"""Replay retains prefix/tool rows and generated feedback without a guessed shift."""

import torch

from vllm_omni.experimental.fullduplex.duplexio.trajectory import TrajectoryRecorder
from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import drain_delta_payload, replace_snapshot_keys
from vllm_omni.outputs.output_modality import TensorAccumulationStrategy


def segment(frames: int, *, live: bool, predict: bool, prompt: bool = False) -> dict[str, torch.Tensor]:
    return {
        "replay_text_ids": torch.arange(4 * frames).view(frames, 4),
        "replay_user_features": torch.randn(frames, 8, dtype=torch.bfloat16),
        "replay_agent_audio": torch.randn(frames, 3),
        "replay_audio_mask": torch.full((frames,), live),
        "replay_prompt_frames": torch.full((frames,), prompt),
        "agent_token_id": torch.tensor([17]),
        "user_token_id": torch.tensor([1]),
        "user_emit": torch.tensor([True]),
        "user_emit_logprob": torch.tensor([-0.2]),
        "user_token_logprob": torch.tensor([-0.3]),
        "user_action_logprob": torch.tensor([-0.5]),
        "user_action_eligible": torch.tensor([True]),
        "policy_version": torch.tensor([0]),
        "tool_call_token_id": torch.tensor([2]),
        "agent_audio_token_ids": torch.randn(3) if predict else torch.empty(0),
        "predictor_hiddens": torch.randn(6, 8) if predict else torch.empty(0),
        "agent_token_logprob": -torch.rand(1) if predict else torch.empty(0),
        "agent_emit_logprob": -torch.rand(1) if predict else torch.empty(0),
        "tool_token_logprob": -torch.rand(1) if predict else torch.empty(0),
        "tool_emit_logprob": -torch.rand(1) if predict else torch.empty(0),
    }


def test_predictor_indices_cover_final_prefix_and_live_rows_and_tool_burst(tmp_path) -> None:
    recorder = TrajectoryRecorder()
    parts = [
        segment(3, live=False, predict=False),
        segment(2, live=False, predict=True),
        segment(1, live=True, predict=True),
        segment(4, live=False, predict=False),
        segment(2, live=False, predict=True),
        segment(1, live=True, predict=True),
    ]
    for part in parts:
        recorder.append(part)
    trace = recorder.tensors()
    assert trace["prediction_rows"].tolist() == [4, 5, 11, 12]
    assert trace["audio_mask"].tolist() == [False] * 5 + [True] + [False] * 6 + [True]
    for key in ("text_ids", "user_features", "agent_audio"):
        torch.testing.assert_close(trace[key], torch.cat([part[f"replay_{key}"] for part in parts]))
    assert trace["sampled_audio"].shape == (4, 3)
    assert trace["predictor_hiddens"].shape == (4, 6, 8)
    # One behaviour log-prob per prediction, matching row_versions.
    assert trace["sampled_agent_logprobs"].shape == trace["row_versions"].shape
    assert trace["agent_emit_logprobs"].shape == trace["row_versions"].shape
    for field, output_key in (("sampled_tool_logprobs", "tool_token_logprob"), ("tool_emit_logprobs", "tool_emit_logprob")):
        torch.testing.assert_close(trace[field], torch.cat([
            part[output_key] for part in parts if part["agent_audio_token_ids"].numel()
        ]))
    path = tmp_path / "trace.pt"
    torch.save(trace, path)
    loaded = torch.load(path, weights_only=True)
    for key in trace:
        torch.testing.assert_close(loaded[key], trace[key])


def test_voice_prompt_survives_recording_and_training_replay() -> None:
    import pytest

    training = pytest.importorskip("duplexio.opd")
    recorder = TrajectoryRecorder()
    recorder.append(segment(2, live=False, predict=False, prompt=True))
    recorder.append(segment(3, live=False, predict=True))
    recorder.append(segment(1, live=True, predict=True))
    trace = training.PolicyTrajectory(
        conversation_id="voice", runtime_config={}, metadata={}, elapsed_seconds=1.0,
        **recorder.tensors(),
    )
    batch = training.pack_policy_replay([trace], silence_token_id=2, device=torch.device("cpu"))
    assert batch.model_inputs["mask"].tolist() == [False] * 5 + [True]
    assert batch.model_inputs["prompt_frames"].tolist() == [True, True] + [False] * 4
    training.pad_policy_replay(batch, 8, pad_token_id=0, silence_token_id=2)
    assert batch.model_inputs["prompt_frames"].tolist() == [True, True] + [False] * 6


def test_recording_does_not_retain_other_batch_slots_or_mutable_input_storage() -> None:
    batched_features = torch.randn(4, 8, dtype=torch.bfloat16)
    output = segment(1, live=True, predict=True)
    output["replay_user_features"] = batched_features[2:3]
    recorder = TrajectoryRecorder()
    recorder.append(output)
    recorded = recorder.segments[0]["user_features"]
    expected = batched_features[2:3].clone()
    batched_features.zero_()
    torch.testing.assert_close(recorded, expected)
    assert recorded.untyped_storage().nbytes() == recorded.numel() * recorded.element_size()


def test_engine_delta_outputs_contain_only_the_current_append_inputs() -> None:
    accumulated = MultimodalPayload()
    recorder = TrajectoryRecorder()
    parts = [segment(5, live=False, predict=True), segment(1, live=True, predict=True)]
    for part in parts:
        incoming = MultimodalPayload.from_dict(part)
        replace_snapshot_keys(accumulated, incoming)
        accumulated = accumulated.merged_with(incoming)
        accumulated.consolidate_tensors(TensorAccumulationStrategy.CONCAT_LAST)
        for key, expected in part.items():
            torch.testing.assert_close(accumulated[key], expected)
        recorder.append(accumulated)
        drain_delta_payload(accumulated)
    assert recorder.tensors()["prediction_rows"].tolist() == [4, 5]


def test_user_action_eligibility_excludes_context_replacement_and_terminal_prediction():
    recorder = TrajectoryRecorder()
    parts = [
        segment(2, live=False, predict=True),
        segment(1, live=True, predict=True),
        segment(3, live=False, predict=True),
        segment(1, live=True, predict=True),
    ]
    for index, part in enumerate(parts):
        part["policy_version"] = torch.tensor([index // 2])
        recorder.append(part, version=index // 2)
    trace = recorder.tensors()
    assert trace["row_versions"].tolist() == [0, 0, 1, 1]
    assert trace["user_action_eligible"].tolist() == [True, False, True, False]
    assert trace["user_token_eligible"].tolist() == [True, False, True, False]
    torch.testing.assert_close(trace["user_action_logprobs"], torch.full((4,), -0.5))


def test_recorder_rejects_wrong_policy_version():
    import pytest

    with pytest.raises(ValueError, match="disagrees with rollout gate"):
        TrajectoryRecorder().append(segment(1, live=True, predict=True), version=1)
