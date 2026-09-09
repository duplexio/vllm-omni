# Copyright (c) Kyutai, all rights reserved.
# See PocketTTSLicense.txt in this directory.
"""Batched inference for DuplexIO's Pocket-TTS FlowMap checkpoints."""

from __future__ import annotations

import math
from collections.abc import Callable

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


class AdaLNResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
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
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(dim, output_dim)
        self.adaln_projection = nn.Linear(dim, 2 * dim)

    def forward(self, hidden: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.adaln_projection(F.silu(conditioning)).chunk(2, dim=-1)
        return self.linear(self.norm(hidden) * (1 + scale) + shift)


class FlowMapSampler(nn.Module):
    """Keep the training checkpoint's audio_sampler.flow parameter namespace."""

    def __init__(
        self,
        latent_dim: int,
        conditioning_dim: int,
        mlp_dim: int,
        mlp_depth: int,
        *,
        inference_steps: int,
        sampling_temperature: float,
        use_cuda_graph: bool = False,
    ) -> None:
        super().__init__()
        self.flow = FlowMap(
            latent_dim,
            mlp_dim,
            conditioning_dim,
            mlp_depth,
            inference_steps=inference_steps,
            sampling_temperature=sampling_temperature,
        )
        self.use_cuda_graph = use_cuda_graph
        self.sample_function = (
            torch.compile(
                self.flow.sample, fullgraph=True, dynamic=True,
                options={"emulate_precision_casts": True},
            )
            if use_cuda_graph else self.flow.sample
        )
        self.graphs: dict[int, FlowMapGraph] = {}

    def sample(self, conditioning: Tensor, noise: Tensor) -> Tensor:
        """Sample with explicit request-owned noise; return owned latent storage."""
        if not self.use_cuda_graph:
            return self.sample_function(conditioning, noise)
        batch = conditioning.shape[0]
        if batch not in self.graphs:
            self.graphs[batch] = FlowMapGraph(self.sample_function, conditioning, noise)
        return self.graphs[batch](conditioning, noise)


class FlowMap(nn.Module):
    """FP32 parameters and integration with the training checkpoint's names.

    Rows are independent conversations. The caller supplies standard-normal
    noise explicitly, so parity checks can replay identical stochastic inputs.
    Match the reference rollout's autocast context for linear-layer arithmetic.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        *,
        inference_steps: int = 1,
        sampling_temperature: float = 0.3,
    ) -> None:
        super().__init__()
        self.inference_steps = inference_steps
        self.sampling_temperature = sampling_temperature
        self.input_projection = nn.Linear(in_channels, model_channels)
        self.start_time_embedding = TimestepEmbedder(model_channels)
        self.target_time_embedding = TimestepEmbedder(model_channels)
        self.conditioning_embedding = nn.Linear(cond_channels, model_channels)
        self.blocks = nn.ModuleList(
            [AdaLNResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.final_layer = FinalLayer(model_channels, in_channels)
        # vLLM constructs the backbone under a BF16 default; the trained
        # FlowMap parameters and integration are FP32; linears obey autocast.
        self.float()

    def forward(self, x: Tensor, cond: Tensor, s: Tensor, t: Tensor) -> Tensor:
        modulation = (
            self.start_time_embedding(s) + self.target_time_embedding(t)
        ) / 2 + self.conditioning_embedding(cond)
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden, modulation)
        return self.final_layer(hidden, modulation)

    def sample(self, conditioning: Tensor, noise: Tensor) -> Tensor:
        """Integrate from noise (time zero) to a normalized latent (time one)."""
        current = self.sampling_temperature**0.5 * noise
        for step in range(self.inference_steps):
            s = conditioning.new_full(
                conditioning.shape[:-1], step / self.inference_steps
            )
            t = conditioning.new_full(
                conditioning.shape[:-1], (step + 1) / self.inference_steps
            )
            current = current + self(current, conditioning, s, t) / self.inference_steps
        return current


class FlowMapGraph:
    """Capture the compiled deterministic sampler, leaving RNG with each request."""

    def __init__(
        self,
        sample: Callable[[Tensor, Tensor], Tensor],
        conditioning: Tensor,
        noise: Tensor,
    ) -> None:
        self.conditioning = conditioning.clone()
        self.noise = noise.clone()
        stream = torch.cuda.Stream(device=conditioning.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sample(self.conditioning, self.noise)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = sample(self.conditioning, self.noise)

    def __call__(self, conditioning: Tensor, noise: Tensor) -> Tensor:
        self.conditioning.copy_(conditioning)
        self.noise.copy_(noise)
        self.graph.replay()
        # Request state and asynchronous output consumers outlive this replay.
        return self.output.clone()
