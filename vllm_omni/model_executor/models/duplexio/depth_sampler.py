# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under MoshiLicense.txt in this directory.
"""Inference implementation of DuplexIO's speaker-conditioned depth sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledEmbedding(nn.Embedding):
    """Moshi embedding with DuplexIO's serving-checkpoint module names."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        low_rank: int | None,
        zero_token_id: int = -1,
    ) -> None:
        super().__init__(num_embeddings, low_rank or embedding_dim)
        self.zero_token_id = zero_token_id
        self.output_projection = (
            nn.Linear(low_rank, embedding_dim, bias=False)
            if low_rank is not None
            else None
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        zero_mask = token_ids == self.zero_token_id
        hidden = super().forward(token_ids.clamp_min(0))
        hidden = torch.where(zero_mask.unsqueeze(-1), hidden.new_zeros(()), hidden)
        if self.output_projection is not None:
            hidden = self.output_projection(hidden)
        return hidden


class SpeakerAdaptiveLayerNorm(nn.Module):
    """LayerNorm modulated by the selected voice embedding."""

    def __init__(self, dim: int, speaker_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(speaker_dim, 2 * dim),
        )

    def forward(self, hidden: Tensor, speaker: Tensor) -> Tensor:
        shift, scale = self.modulation(speaker).chunk(2, dim=-1)
        return self.norm(hidden) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GatedMLP(nn.Module):
    def __init__(self, dim: int, feedforward_dim: int) -> None:
        super().__init__()
        hidden_dim = (
            (21 * dim) // 8
            if feedforward_dim == 4 * dim
            else (2 * feedforward_dim) // 3
        )
        self.input = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.input(hidden).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


@dataclass(frozen=True)
class AttentionCache:
    keys: Tensor | None = None
    values: Tensor | None = None


class DepthAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_codebooks: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.input_projections = nn.ModuleList(
            [nn.Linear(dim, 3 * dim, bias=False) for _ in range(num_codebooks)]
        )
        self.output_projections = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_codebooks)]
        )

    def step(
        self,
        hidden: Tensor,
        codebook: int,
        cache: AttentionCache,
    ) -> tuple[Tensor, AttentionCache]:
        batch_size, sequence_length, dim = hidden.shape
        assert sequence_length == 1
        projected = self.input_projections[codebook](hidden)
        query, key, value = (
            projected.view(
                batch_size,
                1,
                3,
                self.num_heads,
                self.head_dim,
            ).unbind(dim=2)
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if cache.keys is not None:
            assert cache.values is not None
            key = torch.cat((cache.keys, key), dim=2)
            value = torch.cat((cache.values, value), dim=2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch_size, 1, dim)
        return self.output_projections[codebook](attended), AttentionCache(
            keys=key,
            values=value,
        )


class DepthFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        feedforward_dim: int,
        num_codebooks: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GatedMLP(dim, feedforward_dim) for _ in range(num_codebooks)]
        )

    def step(self, hidden: Tensor, codebook: int) -> Tensor:
        return self.layers[codebook](hidden)


class DepthTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_codebooks: int,
        speaker_dim: int,
    ) -> None:
        super().__init__()
        self.attention_norm = SpeakerAdaptiveLayerNorm(dim, speaker_dim)
        self.attention = DepthAttention(dim, num_heads, num_codebooks)
        self.feedforward_norm = SpeakerAdaptiveLayerNorm(dim, speaker_dim)
        self.feedforward = DepthFeedForward(
            dim,
            feedforward_dim,
            num_codebooks,
        )

    def step(
        self,
        hidden: Tensor,
        speaker: Tensor,
        codebook: int,
        cache: AttentionCache,
    ) -> tuple[Tensor, AttentionCache]:
        update, cache = self.attention.step(
            self.attention_norm(hidden, speaker),
            codebook,
            cache,
        )
        hidden = hidden + update
        return (
            hidden
            + self.feedforward.step(
                self.feedforward_norm(hidden, speaker),
                codebook,
            ),
            cache,
        )


class DepthTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        num_codebooks: int,
        speaker_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DepthTransformerLayer(
                    dim,
                    num_heads,
                    feedforward_dim,
                    num_codebooks,
                    speaker_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def step(
        self,
        hidden: Tensor,
        speaker: Tensor,
        codebook: int,
        caches: tuple[AttentionCache, ...],
    ) -> tuple[Tensor, tuple[AttentionCache, ...]]:
        next_caches = []
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden, cache = layer.step(hidden, speaker, codebook, cache)
            next_caches.append(cache)
        return hidden, tuple(next_caches)


@dataclass(frozen=True)
class DepthSamplerConfig:
    conditioning_dim: int
    speaker_embedding_dim: int
    text_vocab_size: int
    codebook_size: int = 2_048
    num_codebooks: int = 8
    low_rank_embeddings: int | None = 128
    dim: int = 1_024
    num_layers: int = 6
    num_heads: int = 16
    feedforward_dim: int = 4_224
    sampling_temperature: float = 0.8
    sampling_top_k: int = 250
    semantic_sampling_top_k: int | None = None


class DepthAutoregressiveSampler(nn.Module):
    """Generate one delayed Mimi column from text, voice, and backbone state."""

    def __init__(self, config: DepthSamplerConfig) -> None:
        super().__init__()
        self.config = config
        self.conditioning_projections = nn.ModuleList(
            [
                nn.Linear(config.conditioning_dim, config.dim, bias=False)
                for _ in range(config.num_codebooks)
            ]
        )
        self.speaker_projection = nn.Linear(
            config.speaker_embedding_dim,
            config.dim,
            bias=False,
        )
        self.previous_codebook_embeddings = nn.ModuleList(
            [
                ScaledEmbedding(
                    config.codebook_size + 1,
                    config.dim,
                    low_rank=config.low_rank_embeddings,
                )
                for _ in range(config.num_codebooks - 1)
            ]
        )
        self.text_embedding = ScaledEmbedding(
            config.text_vocab_size + 1,
            config.dim,
            low_rank=config.low_rank_embeddings,
        )
        self.transformer = DepthTransformer(
            config.dim,
            config.num_layers,
            config.num_heads,
            config.feedforward_dim,
            config.num_codebooks,
            config.dim,
        )
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.dim, config.codebook_size, bias=False)
                for _ in range(config.num_codebooks)
            ]
        )

    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        text_tokens: Tensor,
        speaker_embeddings: Tensor,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        semantic_top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        temperature = (
            self.config.sampling_temperature
            if temperature is None
            else temperature
        )
        top_k = self.config.sampling_top_k if top_k is None else top_k
        semantic_top_k = (
            self.config.semantic_sampling_top_k
            if semantic_top_k is None
            else semantic_top_k
        )
        speaker = self.speaker_projection(speaker_embeddings)
        caches = tuple(AttentionCache() for _ in self.transformer.layers)
        sampled_codes = []
        last_token = text_tokens
        for codebook, head in enumerate(self.heads):
            token = (
                self.text_embedding(last_token)
                if codebook == 0
                else self.previous_codebook_embeddings[codebook - 1](last_token)
            )
            hidden = token + self.conditioning_projections[codebook](conditioning)
            hidden, caches = self.transformer.step(
                hidden.unsqueeze(1),
                speaker,
                codebook,
                caches,
            )
            logits = head(hidden[:, 0])
            codebook_top_k = semantic_top_k if codebook == 0 else top_k
            codebook_top_k = top_k if codebook_top_k is None else codebook_top_k
            if codebook_top_k == 1:
                last_token = logits.argmax(dim=-1)
            else:
                probabilities = F.softmax(logits.float() / temperature, dim=-1)
                top_probabilities, top_indices = probabilities.topk(
                    codebook_top_k,
                    dim=-1,
                )
                selected = torch.multinomial(
                    top_probabilities,
                    num_samples=1,
                    generator=generator,
                )
                last_token = top_indices.gather(-1, selected).squeeze(-1)
            sampled_codes.append(last_token)
        return torch.stack(sampled_codes, dim=-1)


__all__ = [
    "DepthAutoregressiveSampler",
    "DepthSamplerConfig",
]
