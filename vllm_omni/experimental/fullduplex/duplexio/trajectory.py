# SPDX-License-Identifier: Apache-2.0
"""Exact model inputs and sampled outputs for offline policy replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class TrajectoryRecorder:
    """Collect completed segments; predictor rows refer to consumed input rows.

    A final prefix row can predict audio before any live user frame is consumed.
    Intermediate prefix/tool-result segments have inputs but no predictions.
    Keeping explicit row indices avoids manufacturing a next-frame target shift.
    Each prediction records the policy version whose weights produced it.
    """

    segments: list[dict[str, Tensor]] = field(default_factory=list)
    prediction_rows: list[int] = field(default_factory=list)
    row_versions: list[int] = field(default_factory=list)
    frames: int = 0

    def append(self, output: Mapping[str, Tensor], version: int | None = None) -> None:
        """Own the tensors crossing the output boundary, without shared storage."""
        segment = {
            name: output[f"replay_{name}"].detach().cpu().clone()
            for name in ("text_ids", "user_features", "agent_audio", "audio_mask", "prompt_frames")
        }
        frames = segment["text_ids"].shape[0]
        assert frames > 0, "Rollout must enable duplexio_record_inputs"
        self.frames += frames
        if output["agent_audio_token_ids"].numel():
            self.prediction_rows.append(self.frames - 1)
            active_version = output["policy_version"].item()
            if version is not None and active_version != version:
                raise ValueError(f"Prediction version {active_version} disagrees with rollout gate {version}")
            self.row_versions.append(active_version)
            segment["sampled_user_ids"] = output["user_token_id"].detach().cpu().clone().reshape(1)
            segment["sampled_user_emits"] = output["user_emit"].detach().cpu().clone().reshape(1)
            segment["user_action_eligible"] = output["user_action_eligible"].detach().cpu().clone().reshape(1)
            segment["sampled_agent_ids"] = output["agent_token_id"].detach().cpu().clone()
            segment["sampled_tool_ids"] = output["tool_call_token_id"].detach().cpu().clone()
            segment["sampled_audio"] = output["agent_audio_token_ids"].detach().cpu().clone().unsqueeze(0)
            # Scores at the rollout's weights. Tool emit stores the raw head's score
            # for the chosen action, including forced decisions; the other fields
            # store probabilities under the actual sampling distribution.
            for name, key in (
                ("sampled_agent_logprobs", "agent_token_logprob"),
                ("agent_emit_logprobs", "agent_emit_logprob"),
                ("sampled_tool_logprobs", "tool_token_logprob"),
                ("tool_emit_logprobs", "tool_emit_logprob"),
                ("user_emit_logprobs", "user_emit_logprob"),
                ("sampled_user_logprobs", "user_token_logprob"),
                ("user_action_logprobs", "user_action_logprob"),
            ):
                segment[name] = output[key].detach().cpu().clone().float().reshape(1)
            if output["predictor_hiddens"].numel():
                segment["predictor_hiddens"] = output["predictor_hiddens"].detach().cpu().clone().unsqueeze(0)
        self.segments.append(segment)

    def tensors(self) -> dict[str, Tensor]:
        """Return a weights-only-loadable tensor dictionary for one conversation."""
        predictions = [segment for segment in self.segments if "sampled_audio" in segment]
        assert predictions, "A trajectory must contain at least one model prediction"
        prediction_rows = torch.tensor(self.prediction_rows, dtype=torch.long)
        text_ids = torch.cat([segment["text_ids"] for segment in self.segments])
        audio_mask = torch.cat([segment["audio_mask"] for segment in self.segments])
        user_ids = torch.cat([segment["sampled_user_ids"] for segment in predictions])
        user_emits = torch.cat([segment["sampled_user_emits"] for segment in predictions])
        # A decision at t is eligible only if t+1 actually consumed its feedback.
        # A following context burst replaces that prediction; a terminal one has
        # no target row. Keep both in the record, including their behavior probs.
        targets = prediction_rows + 1
        consumed = targets < self.frames
        eligible = torch.zeros_like(consumed)
        eligible[consumed] = audio_mask[targets[consumed]]
        eligible &= torch.cat([segment["user_action_eligible"] for segment in predictions])
        if not torch.equal(text_ids[targets[eligible], 1], user_ids[eligible]):
            raise ValueError("User decisions do not match next-frame feedback")
        return {
            "user_action_eligible": eligible,
            "user_token_eligible": eligible & user_emits,
            **{
                name: torch.cat([segment[name] for segment in self.segments])
                for name in ("text_ids", "user_features", "agent_audio", "audio_mask", "prompt_frames")
            },
            **{
                name: torch.cat([segment[name] for segment in predictions])
                for name in (
                    "sampled_agent_ids",
                    "sampled_tool_ids",
                    "sampled_audio",
                    "sampled_agent_logprobs",
                    "agent_emit_logprobs",
                    "sampled_tool_logprobs",
                    "tool_emit_logprobs",
                    "sampled_user_ids",
                    "sampled_user_emits",
                    "sampled_user_logprobs",
                    "user_emit_logprobs",
                    "user_action_logprobs",
                )
            },
            "prediction_rows": prediction_rows,
            "row_versions": torch.tensor(self.row_versions, dtype=torch.long),
            **(
                {"predictor_hiddens": torch.cat([part["predictor_hiddens"] for part in predictions])}
                if "predictor_hiddens" in predictions[0] else {}
            ),
        }
