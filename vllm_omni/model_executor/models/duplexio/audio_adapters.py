# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MLP audio adapters used by DuplexIO."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AudioInputAdapter(nn.Module):
    """Map one frame of user-audio features into Qwen and retain its local skip."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, audio_features: Tensor) -> tuple[Tensor, Tensor]:
        gate, value = self.gate_up_proj(audio_features).chunk(2, dim=-1)
        skip = F.silu(gate) * value
        return self.output_proj(skip), skip


class AgentAudioInputAdapter(nn.Module):
    """Inject speaker-conditioned agent audio into Qwen."""

    def __init__(
        self,
        input_dim: int,
        speaker_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.speaker_modulation = nn.Linear(speaker_dim, 2 * hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(
        self,
        audio_features: Tensor,
        speaker_embeddings: Tensor,
        request_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate, value = self.gate_up_proj(audio_features).chunk(2, dim=-1)
        skip = F.silu(gate) * value
        scale, shift = self.speaker_modulation(speaker_embeddings).index_select(
            0,
            request_indices,
        ).chunk(2, dim=-1)
        conditioned = self.norm(skip) * (1 + scale) + shift
        return self.output_proj(conditioned), skip


class AgentAudioOutputAdapter(nn.Module):
    """Map Qwen's agent-audio state and local conditioning to the depth head."""

    def __init__(
        self,
        backbone_dim: int,
        skip_dim: int,
        speaker_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        gate_up_dim = 2 * hidden_dim
        self.backbone_gate_up_proj = nn.Linear(
            backbone_dim,
            gate_up_dim,
            bias=False,
        )
        self.skip_gate_up_proj = nn.Linear(skip_dim, gate_up_dim, bias=False)
        self.speaker_gate_up_proj = nn.Linear(
            speaker_dim,
            gate_up_dim,
            bias=False,
        )
        self.output_proj = nn.Linear(hidden_dim, backbone_dim, bias=False)

    def forward(
        self,
        backbone_hidden: Tensor,
        audio_skip: Tensor,
        speaker_embeddings: Tensor,
        request_indices: Tensor,
    ) -> Tensor:
        speaker_gate_up = self.speaker_gate_up_proj(
            speaker_embeddings
        ).index_select(0, request_indices)
        gate_up = (
            self.backbone_gate_up_proj(backbone_hidden)
            + self.skip_gate_up_proj(audio_skip)
            + speaker_gate_up
        )
        gate, value = gate_up.chunk(2, dim=-1)
        return self.output_proj(F.silu(gate) * value)
