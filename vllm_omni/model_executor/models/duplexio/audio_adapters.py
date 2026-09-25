# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MLP audio adapters used by DuplexIO."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def audio_adapter_hidden(gate_up: Tensor) -> Tensor:
    """Keep gate and normalization identical in train/decode."""
    gate, value = gate_up.chunk(2, dim=-1)
    hidden = F.silu(gate) * value
    # Normalize in FP32 with a row-local reduction independent of batch size.
    hidden = hidden.float()
    eps = torch.finfo(torch.float32).eps
    return F.rms_norm(hidden, (hidden.shape[-1],), eps=eps)


@torch.compile(dynamic=True, fullgraph=True, options={"triton.cudagraphs": False})
def audio_adapter(audio_features: Tensor, gate_up_weight: Tensor, output_weight: Tensor) -> Tensor:
    """Compile both projections and the intervening activation/normalization together."""
    return F.linear(audio_adapter_hidden(F.linear(audio_features, gate_up_weight)), output_weight)


class AudioInputAdapter(nn.Module):
    """Map one frame of audio features into Qwen.

    Both audio cells use it: the agent's voice comes from the pinned prompt in
    its own cell, so nothing is injected here.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, audio_features: Tensor) -> Tensor:
        # The first projection computes in the weight dtype either way (autocast
        # or matching inputs); casting first keeps prompt-only FP32 and sampled
        # BF16 batches on one compiled graph.
        weight = self.gate_up_proj.weight
        return audio_adapter(audio_features.to(weight.dtype), weight, self.output_proj.weight)
