# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only MLP audio adapters used by DuplexIO."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AudioInputAdapter(nn.Module):
    """Map one frame of audio features into Qwen with a SwiGLU MLP; each audio cell has its own."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, audio_features: Tensor) -> Tensor:
        # The projections compute in the weight dtype either way (autocast or
        # matching inputs); casting first keeps prompt-only FP32 and sampled
        # BF16 batches on one compiled graph.
        gate, value = self.gate_up_proj(audio_features.to(self.gate_up_proj.weight.dtype)).chunk(2, dim=-1)
        return self.output_proj(F.silu(gate) * value)
