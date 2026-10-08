# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Normalization between Pocket Mimi latents and the model's agent-audio space."""

import torch
import torch.nn as nn
from torch import Tensor


class ContinuousAudioRepresentation(nn.Module):
    """Per-channel affine map between codec latents and the model's agent-audio space."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.register_buffer("embedding_mean", torch.zeros(embedding_dim, dtype=torch.float32))
        self.register_buffer("embedding_scale", torch.ones(embedding_dim, dtype=torch.float32))

    def normalize(self, latent: Tensor) -> Tensor:
        return (latent - self.embedding_mean) / self.embedding_scale

    def denormalize(self, latent: Tensor) -> Tensor:
        return latent * self.embedding_scale + self.embedding_mean
