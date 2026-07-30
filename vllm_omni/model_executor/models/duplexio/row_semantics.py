# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact six-cell row semantics shared by DuplexIO's native Qwen layers."""

from __future__ import annotations

import torch
from torch import Tensor

DUPLEXIO_NUM_TEXT_CELLS = 4
DUPLEXIO_NUM_CELLS = 6


def duplexio_attention_visible(
    query_positions: Tensor,
    key_positions: Tensor,
    key_active: Tensor,
    *,
    audio_attention_window_frames: int,
) -> Tensor:
    """Return the exact DuplexIO visibility relation for flattened cells.

    Positions are logical vLLM token positions, with six consecutive cells per
    frame. A query sees itself and active keys from prior frames. Audio keys
    additionally expire after ``audio_attention_window_frames``.
    """
    query_frames = torch.div(
        query_positions,
        DUPLEXIO_NUM_CELLS,
        rounding_mode="floor",
    )
    key_frames = torch.div(
        key_positions,
        DUPLEXIO_NUM_CELLS,
        rounding_mode="floor",
    )
    key_cells = torch.remainder(key_positions, DUPLEXIO_NUM_CELLS)
    prior_frame = query_frames > key_frames
    key_is_audio = key_cells >= DUPLEXIO_NUM_TEXT_CELLS
    audio_in_window = (
        query_frames - key_frames
    ) <= audio_attention_window_frames
    key_visible = key_active & (~key_is_audio | audio_in_window)
    return (prior_frame & key_visible) | (query_positions == key_positions)


def expand_stream_conv_weight(weight: Tensor, *, num_cells: int) -> Tensor:
    """Convert a causal kernel into an equivalent row-major dilated kernel."""
    if weight.ndim != 3:
        raise ValueError(
            "DuplexIO GDN convolution weight must have shape "
            f"(channels, 1, kernel), got {tuple(weight.shape)}"
        )
    kernel_size = weight.shape[-1]
    expanded_size = (kernel_size - 1) * num_cells + 1
    expanded = weight.new_zeros((*weight.shape[:-1], expanded_size))
    expanded[..., ::num_cells] = weight
    return expanded


def mask_inactive_gdn_gates(
    beta_logits: Tensor,
    decay_logits: Tensor,
    key_active: Tensor,
) -> tuple[Tensor, Tensor]:
    """Make inactive cells exact no-ops in Qwen's GDN recurrence.

    Qwen computes ``beta = sigmoid(beta_logits)`` and a softplus-derived decay.
    Negative infinity therefore produces exactly zero for both update terms.
    """
    inactive = ~key_active.unsqueeze(-1)
    return (
        beta_logits.masked_fill(inactive, -torch.inf),
        decay_logits.masked_fill(inactive, -torch.inf),
    )


def duplexio_frame_positions(token_positions: Tensor) -> Tensor:
    """Map vLLM's flattened cell positions to shared per-frame RoPE positions."""
    return torch.div(token_positions, DUPLEXIO_NUM_CELLS, rounding_mode="floor")


def duplexio_cell_ids(token_positions: Tensor) -> Tensor:
    """Return the cell column for each flattened vLLM token position."""
    return torch.remainder(token_positions, DUPLEXIO_NUM_CELLS)


__all__ = [
    "DUPLEXIO_NUM_CELLS",
    "DUPLEXIO_NUM_TEXT_CELLS",
    "duplexio_attention_visible",
    "duplexio_cell_ids",
    "duplexio_frame_positions",
    "expand_stream_conv_weight",
    "mask_inactive_gdn_gates",
]
