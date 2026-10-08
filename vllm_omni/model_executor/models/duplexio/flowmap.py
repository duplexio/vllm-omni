# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) Kyutai, all rights reserved.
# See PocketTTSLicense.txt in this directory.
"""Batched inference for DuplexIO's Pocket-TTS FlowMap checkpoints."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PocketRMSNorm(nn.Module):
    """Pocket uses sample variance, not mean square, for time embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, hidden: Tensor) -> Tensor:
        variance = hidden.var(dim=-1, keepdim=True) + 1e-5
        # Match Pocket's AMP arithmetic: alpha is an FP32 parameter, but
        # the normalization and its result retain the activation dtype.
        scale = self.alpha.to(variance) * torch.rsqrt(variance)
        return (hidden * scale).to(hidden.dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.exp(
                -math.log(10_000.0) * torch.arange(128, dtype=torch.float32) / 128
            ),
            persistent=False,
        )
        self.input_projection = nn.Linear(256, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.norm = PocketRMSNorm(dim)

    def forward(self, time: Tensor) -> Tensor:
        angles = time.unsqueeze(-1) * self.frequencies
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return self.norm(
            self.output_projection(F.silu(self.input_projection(embedding)))
        )


class PocketLayerNorm(nn.Module):
    """LayerNorm computed in the activation dtype, without an FP32 upcast."""

    def __init__(self, dim: int, *, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if elementwise_affine else None

    def forward(self, hidden: Tensor) -> Tensor:
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (hidden - mean) / torch.sqrt(variance + 1e-6)
        if self.weight is not None:
            normalized = normalized * self.weight + self.bias
        return normalized


class AdaLNResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = PocketLayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.adaln_projection = nn.Linear(dim, 3 * dim)

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale, gate = self.adaln_projection(F.silu(conditioning)).chunk(
            3, dim=-1
        )
        residual = self.norm(hidden) * (1 + scale) + shift
        return hidden + gate * self.linear2(F.silu(self.linear1(residual)))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = PocketLayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, output_dim)
        self.adaln_projection = nn.Linear(dim, 2 * dim)

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.adaln_projection(F.silu(conditioning)).chunk(2, dim=-1)
        return self.linear(self.norm(hidden) * (1 + scale) + shift)


class FlowMapSampler(nn.Module):
    """The checkpoint's ``audio_sampler.flow`` module, optionally compiled for serving."""

    def __init__(
        self,
        latent_dim: int,
        conditioning_dim: int,
        mlp_dim: int,
        mlp_depth: int,
        *,
        inference_steps: int,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.flow = FlowMap(
            latent_dim,
            mlp_dim,
            conditioning_dim,
            mlp_depth,
            inference_steps=inference_steps,
        )
        # Compiled for serving, where the model captures it with text sampling.
        self.sample_function = (
            torch.compile(
                self.flow.sample, fullgraph=True, dynamic=True,
                options={"emulate_precision_casts": True},
            )
            if compile else self.flow.sample
        )

    def sample(self, conditioning: Tensor, noise: Tensor, temperature: Tensor) -> Tensor:
        """Sample with explicit request-owned noise, at each row's temperature."""
        return self.sample_function(conditioning, noise, temperature)


class FlowMap(nn.Module):
    """A one-to-few-step flow map from noise to the agent's next audio latent, with FP32 parameters.

    Rows are independent conversations. The caller supplies standard-normal
    noise per row; linear layers follow the ambient autocast.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        *,
        inference_steps: int = 1,
    ) -> None:
        super().__init__()
        self.inference_steps = inference_steps
        self.input_projection = nn.Linear(in_channels, model_channels)
        self.start_time_embedding = TimestepEmbedder(model_channels)
        self.target_time_embedding = TimestepEmbedder(model_channels)
        self.conditioning_embedding = nn.Linear(cond_channels, model_channels)
        self.blocks = nn.ModuleList(
            [AdaLNResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.final_layer = FinalLayer(model_channels, in_channels)
        # vLLM constructs the backbone under a BF16 default; FlowMap parameters
        # and integration stay FP32, and its linears follow the ambient autocast.
        self.float()

    def forward(self, x: Tensor, cond: Tensor, s: Tensor, t: Tensor) -> Tensor:
        modulation = (
            self.start_time_embedding(s) + self.target_time_embedding(t)
        ) / 2 + self.conditioning_embedding(cond)
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden, modulation)
        return self.final_layer(hidden, modulation)

    def sample(self, conditioning: Tensor, noise: Tensor, temperature: Tensor) -> Tensor:
        """Integrate from noise (time zero) to a normalized latent (time one).

        ``temperature`` holds one value per row: the variance of its noise.
        """
        current = temperature.sqrt().unsqueeze(-1) * noise
        for step in range(self.inference_steps):
            s = conditioning.new_full(
                conditioning.shape[:-1], step / self.inference_steps
            )
            t = conditioning.new_full(
                conditioning.shape[:-1], (step + 1) / self.inference_steps
            )
            current = current + self(current, conditioning, s, t) / self.inference_steps
        return current
