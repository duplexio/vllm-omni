# SPDX-License-Identifier: Apache-2.0
"""Replay retains prefix/tool rows and generated feedback without a guessed shift."""

import torch

from vllm_omni.experimental.fullduplex.duplexio.trajectory import TrajectoryRecorder
from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import drain_delta_payload, replace_snapshot_keys
from vllm_omni.outputs.output_modality import TensorAccumulationStrategy


def segment(frames: int, *, live: bool, predict: bool) -> dict[str, torch.Tensor]:
    return {
        "replay_text_ids": torch.arange(4 * frames).view(frames, 4),
        "replay_user_features": torch.randn(frames, 8, dtype=torch.bfloat16),
        "replay_agent_audio": torch.randn(frames, 3),
        "replay_audio_mask": torch.full((frames,), live),
        "agent_token_id": torch.tensor([17]),
        "tool_call_token_id": torch.tensor([2]),
        "agent_audio_token_ids": torch.randn(3) if predict else torch.empty(0),
        "predictor_hiddens": torch.randn(6, 8) if predict else torch.empty(0),
        "agent_token_logprob": -torch.rand(1) if predict else torch.empty(0),
        "agent_emit_logprob": -torch.rand(1) if predict else torch.empty(0),
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
    path = tmp_path / "trace.pt"
    torch.save(trace, path)
    loaded = torch.load(path, weights_only=True)
    for key in trace:
        torch.testing.assert_close(loaded[key], trace[key])


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
