# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantized Mimi representation used by native DuplexIO serving."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


class MimiEmbedding(nn.Module):
    """Sum one learned embedding table per delayed Mimi codebook."""

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(codebook_size + 1, embedding_dim)
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, codes: Tensor) -> Tensor:
        if codes.shape[-1] != self.num_codebooks:
            raise ValueError(
                "Mimi codes must end in the codebook dimension, got "
                f"{tuple(codes.shape)}"
            )
        output = self.embeddings[0](codes[..., 0])
        for codebook in range(1, self.num_codebooks):
            output = output + self.embeddings[codebook](codes[..., codebook])
        return output


@dataclass
class DelayedMimiState:
    """One request's one-frame acoustic-delay history."""

    previous_acoustic_codes: Tensor
    pending_semantic_code: Tensor | None = None


class DelayedMimiRepresentation:
    """Streaming form of Moshi's semantic-now/acoustics-previous layout."""

    def __init__(
        self,
        *,
        num_codebooks: int,
        codebook_size: int,
        acoustic_delay_frames: int,
    ) -> None:
        if acoustic_delay_frames != 1:
            raise ValueError(
                "Native DuplexIO currently requires a one-frame acoustic delay"
            )
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    def new_state(self, *, device: torch.device) -> DelayedMimiState:
        return DelayedMimiState(
            previous_acoustic_codes=torch.full(
                (self.num_codebooks - 1,),
                self.codebook_size,
                dtype=torch.long,
                device=device,
            )
        )

    def encode_column(
        self,
        raw_codes: Tensor,
        state: DelayedMimiState,
    ) -> Tensor:
        """Delay one raw Mimi column and advance the request-local history."""
        if raw_codes.shape != (self.num_codebooks,):
            raise ValueError(
                f"Expected one Mimi column with {self.num_codebooks} codes, "
                f"got {tuple(raw_codes.shape)}"
            )
        delayed = torch.cat(
            (raw_codes[:1], state.previous_acoustic_codes),
            dim=0,
        )
        state.previous_acoustic_codes = raw_codes[1:]
        return delayed

    def encode_sequence(
        self,
        raw_codes: Tensor,
        state: DelayedMimiState,
    ) -> Tensor:
        """Delay a complete ``(frames, codebooks)`` Mimi sequence."""
        if raw_codes.ndim != 2 or raw_codes.shape[1] != self.num_codebooks:
            raise ValueError(
                "Expected Mimi sequence with shape (frames, "
                f"{self.num_codebooks}), got {tuple(raw_codes.shape)}"
            )
        if raw_codes.shape[0] == 0:
            return raw_codes
        acoustic_codes = torch.cat(
            (
                state.previous_acoustic_codes.unsqueeze(0),
                raw_codes[:-1, 1:],
            ),
            dim=0,
        )
        state.previous_acoustic_codes = raw_codes[-1, 1:]
        return torch.cat((raw_codes[:, :1], acoustic_codes), dim=1)

    def decode_column(
        self,
        delayed_codes: Tensor,
        state: DelayedMimiState,
    ) -> Tensor | None:
        """Undelay a generated column once its matching acoustics arrive."""
        if delayed_codes.shape != (self.num_codebooks,):
            raise ValueError(
                f"Expected one delayed Mimi column with {self.num_codebooks} "
                f"codes, got {tuple(delayed_codes.shape)}"
            )
        semantic = state.pending_semantic_code
        state.pending_semantic_code = delayed_codes[:1]
        if semantic is None:
            return None
        return torch.cat((semantic, delayed_codes[1:]), dim=0)

    def initial_column(self, *, device: torch.device) -> Tensor:
        return torch.full(
            (self.num_codebooks,),
            self.codebook_size,
            dtype=torch.long,
            device=device,
        )


__all__ = [
    "DelayedMimiRepresentation",
    "DelayedMimiState",
    "MimiEmbedding",
]
