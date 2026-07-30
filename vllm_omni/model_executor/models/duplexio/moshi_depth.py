# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under MoshiLicense.txt in this directory.
"""Native inference implementation of Moshi's original depth transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MoshiScaledEmbedding(nn.Embedding):
    """Moshi embedding, including its negative zero token and low-rank path."""

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
        self.low_rank = (
            nn.Linear(low_rank, embedding_dim, bias=False)
            if low_rank is not None
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        zero_mask = input == self.zero_token_id
        output = super().forward(input.clamp_min(0))
        output = torch.where(
            zero_mask.unsqueeze(-1),
            output.new_zeros(()),
            output,
        )
        if self.low_rank is not None:
            output = self.low_rank(output)
        return output


class MoshiRMSNorm(nn.Module):
    """Moshi's float32 RMSNorm with its original ``alpha`` parameter layout."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, hidden: Tensor) -> Tensor:
        input_dtype = hidden.dtype
        hidden_f32 = hidden.float()
        variance = self.eps + hidden_f32.square().mean(dim=-1, keepdim=True)
        return (
            hidden_f32 * self.alpha.float() * torch.rsqrt(variance)
        ).to(input_dtype)


class MoshiActivationGating(nn.Module):
    """Moshi's SiLU-gated feed-forward projection."""

    def __init__(self, dim: int, feedforward_dim: int) -> None:
        super().__init__()
        hidden_dim = (
            (21 * dim) // 8
            if feedforward_dim == 4 * dim
            else (2 * feedforward_dim) // 3
        )
        self.linear_in = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.linear_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.linear_in(hidden).chunk(2, dim=-1)
        return self.linear_out(F.silu(gate) * value)


class MoshiDepthAttention(nn.Module):
    """Original Moshi causal attention with one projection set per depth step."""

    def __init__(self, dim: int, num_heads: int, num_codebooks: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_projs = nn.ModuleList(
            [nn.Linear(dim, 3 * dim, bias=False) for _ in range(num_codebooks)]
        )
        self.out_projs = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_codebooks)]
        )

    def forward(self, hidden: Tensor) -> Tensor:
        batch_size, depth, dim = hidden.shape
        projected = torch.stack(
            [self.in_projs[step](hidden[:, step]) for step in range(depth)],
            dim=1,
        )
        query, key, value = (
            projected.view(
                batch_size,
                depth,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, depth, dim)
        return torch.stack(
            [self.out_projs[step](attended[:, step]) for step in range(depth)],
            dim=1,
        )


class MoshiDepthLayer(nn.Module):
    """One original pre-norm Moshi depth-transformer layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_codebooks: int,
    ) -> None:
        super().__init__()
        self.self_attn = MoshiDepthAttention(dim, num_heads, num_codebooks)
        self.norm1 = MoshiRMSNorm(dim)
        self.norm2 = MoshiRMSNorm(dim)
        self.gating = nn.ModuleList(
            [
                MoshiActivationGating(dim, feedforward_dim)
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, hidden: Tensor) -> Tensor:
        attention_update = self.self_attn(self.norm1(hidden))
        hidden = hidden.to(attention_update) + attention_update
        normalized = self.norm2(hidden)
        feedforward_update = torch.stack(
            [
                self.gating[step](normalized[:, step])
                for step in range(hidden.shape[1])
            ],
            dim=1,
        )
        return hidden.to(feedforward_update) + feedforward_update


class MoshiDepthStack(nn.Module):
    """Moshi depformer stack reset to depth position zero for every audio frame."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        num_codebooks: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MoshiDepthLayer(
                    dim,
                    num_heads,
                    feedforward_dim,
                    num_codebooks,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden: Tensor) -> Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


@dataclass(frozen=True)
class MoshiDepthConfig:
    """The original Moshi 7B depformer shape used by DuplexIO."""

    conditioning_dim: int
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


class MoshiDepthTransformer(nn.Module):
    """Original Moshi depformer modules and weight names, isolated per frame."""

    def __init__(self, config: MoshiDepthConfig) -> None:
        super().__init__()
        self.config = config
        self.depformer_in = nn.ModuleList(
            [
                nn.Linear(config.conditioning_dim, config.dim, bias=False)
                for _ in range(config.num_codebooks)
            ]
        )
        self.depformer_emb = nn.ModuleList(
            [
                MoshiScaledEmbedding(
                    config.codebook_size + 1,
                    config.dim,
                    low_rank=config.low_rank_embeddings,
                )
                for _ in range(config.num_codebooks - 1)
            ]
        )
        self.depformer_text_emb = MoshiScaledEmbedding(
            config.text_vocab_size + 1,
            config.dim,
            low_rank=config.low_rank_embeddings,
        )
        self.depformer_norms = nn.ModuleList(
            [nn.Identity() for _ in range(config.num_codebooks)]
        )
        self.depformer = MoshiDepthStack(
            config.dim,
            config.num_layers,
            config.num_heads,
            config.feedforward_dim,
            config.num_codebooks,
        )
        self.linears = nn.ModuleList(
            [
                nn.Linear(config.dim, config.codebook_size, bias=False)
                for _ in range(config.num_codebooks)
            ]
        )

    def forward(
        self,
        conditioning: Tensor,
        text_tokens: Tensor,
        target_codes: Tensor,
    ) -> Tensor:
        """Return teacher-forced logits shaped ``(batch, codebooks, vocab)``."""
        if target_codes.shape != (
            conditioning.shape[0],
            self.config.num_codebooks,
        ):
            raise ValueError(
                "target_codes must be shaped (batch, num_codebooks), got "
                f"{tuple(target_codes.shape)}"
            )
        inputs = self._inputs(
            conditioning,
            text_tokens,
            target_codes[:, :-1],
        )
        hidden = self.depformer(inputs)
        return torch.stack(
            [
                self.linears[step](self.depformer_norms[step](hidden[:, step]))
                for step in range(self.config.num_codebooks)
            ],
            dim=1,
        )

    def sample(
        self,
        conditioning: Tensor,
        text_tokens: Tensor,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample one complete Mimi column autoregressively across codebooks."""
        temperature = (
            self.config.sampling_temperature
            if temperature is None
            else temperature
        )
        top_k = self.config.sampling_top_k if top_k is None else top_k
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 1 <= top_k <= self.config.codebook_size:
            raise ValueError(
                f"top_k must be in [1, {self.config.codebook_size}], got {top_k}"
            )

        sampled: list[Tensor] = []
        for step in range(self.config.num_codebooks):
            previous_codes = (
                torch.stack(sampled, dim=1)
                if sampled
                else text_tokens.new_empty((text_tokens.shape[0], 0))
            )
            hidden = self.depformer(
                self._inputs(conditioning, text_tokens, previous_codes)
            )[:, -1]
            logits = self.linears[step](self.depformer_norms[step](hidden))
            if temperature == 0:
                token = logits.argmax(dim=-1)
            else:
                top_logits, top_indices = logits.topk(top_k, dim=-1)
                probabilities = (top_logits / temperature).softmax(dim=-1)
                selected = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                token = top_indices.gather(1, selected).squeeze(1)
            sampled.append(token)
        return torch.stack(sampled, dim=1)

    def _inputs(
        self,
        conditioning: Tensor,
        text_tokens: Tensor,
        previous_codes: Tensor,
    ) -> Tensor:
        depth = previous_codes.shape[1] + 1
        if depth > self.config.num_codebooks:
            raise ValueError(f"Depth prefix exceeds {self.config.num_codebooks}")
        inputs: list[Tensor] = []
        for step in range(depth):
            token_embedding = (
                self.depformer_text_emb(text_tokens)
                if step == 0
                else self.depformer_emb[step - 1](previous_codes[:, step - 1])
            )
            inputs.append(self.depformer_in[step](conditioning) + token_embedding)
        return torch.stack(inputs, dim=1)
