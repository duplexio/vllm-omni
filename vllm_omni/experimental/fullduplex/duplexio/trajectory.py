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

    def append(self, output: Mapping[str, Tensor], version: int = 0) -> None:
        """Own the tensors crossing the output boundary, without shared storage."""
        segment = {
            name: output[f"replay_{name}"].detach().cpu().clone()
            for name in ("text_ids", "user_features", "agent_audio", "audio_mask")
        }
        frames = segment["text_ids"].shape[0]
        assert frames > 0, "Rollout must enable duplexio_record_inputs"
        self.frames += frames
        if output["agent_audio_token_ids"].numel():
            self.prediction_rows.append(self.frames - 1)
            self.row_versions.append(version)
            segment["sampled_agent_ids"] = output["agent_token_id"].detach().cpu().clone()
            segment["sampled_tool_ids"] = output["tool_call_token_id"].detach().cpu().clone()
            segment["sampled_audio"] = output["agent_audio_token_ids"].detach().cpu().clone().unsqueeze(0)
            if output["predictor_hiddens"].numel():
                segment["predictor_hiddens"] = output["predictor_hiddens"].detach().cpu().clone().unsqueeze(0)
        self.segments.append(segment)

    def tensors(self) -> dict[str, Tensor]:
        """Return a weights-only-loadable tensor dictionary for one conversation."""
        predictions = [segment for segment in self.segments if "sampled_audio" in segment]
        assert predictions, "A trajectory must contain at least one model prediction"
        return {
            **{
                name: torch.cat([segment[name] for segment in self.segments])
                for name in ("text_ids", "user_features", "agent_audio", "audio_mask")
            },
            **{
                name: torch.cat([segment[name] for segment in predictions])
                for name in ("sampled_agent_ids", "sampled_tool_ids", "sampled_audio")
            },
            "prediction_rows": torch.tensor(self.prediction_rows, dtype=torch.long),
            "row_versions": torch.tensor(self.row_versions, dtype=torch.long),
            **(
                {"predictor_hiddens": torch.cat([part["predictor_hiddens"] for part in predictions])}
                if "predictor_hiddens" in predictions[0] else {}
            ),
        }
