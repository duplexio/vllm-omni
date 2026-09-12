# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MLP audio adapters used by DuplexIO."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vllm_omni.model_executor.models.duplexio.numerics import call_compiled_function


@torch.compile(dynamic=True, fullgraph=True)
def audio_adapter_hidden(gate_up: Tensor) -> Tensor:
    """Keep gate and normalization identical in train/decode."""
    gate, value = gate_up.chunk(2, dim=-1)
    hidden = F.silu(gate) * value
    # Normalize in FP32 with a row-local reduction independent of batch size.
    hidden = hidden.float()
    eps = torch.finfo(torch.float32).eps
    if hidden.is_cuda:
        from quack import rmsnorm

        hidden = rmsnorm(hidden, eps=eps)
    else:
        hidden = F.rms_norm(hidden, (hidden.shape[-1],), eps=eps)
    return hidden


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
        hidden = call_compiled_function(
            audio_adapter_hidden, self.gate_up_proj(audio_features)
        )
        return self.output_proj(hidden)
