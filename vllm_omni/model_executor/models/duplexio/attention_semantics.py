# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention-window predicates vendored from the DuplexIO training repo.

``key_in_window`` and ``key_visible`` are copied verbatim from
``duplexio/models/duplexio.py`` and MUST stay in sync with training. The
window distance runs over AUDIO TIME: a non-decreasing per-conversation
cumsum that advances only on frames carrying real audio, so token-only burst
frames (for example spliced tool results) neither advance it nor consume
window budget, and text keys are never windowed.

Eviction bound: audio position is a non-decreasing cumsum, so the last
frame's audio position is a lower bound for every future query; ``+ 1``
would wrongly assume the next frame advances audio time and evict keys
still visible to frozen-audio-time frames.
"""

from __future__ import annotations

from torch import Tensor

# Number of frame-aligned text streams (system, user, agent, tool_call).
# Mirrors duplexio.multistream.streams.NUM_STREAMS; cells at or above this
# index are audio cells. Equal to row_semantics.DUPLEXIO_NUM_TEXT_CELLS.
NUM_STREAMS = 4


def key_in_window(
    q_window_pos: Tensor,
    k_window_pos: Tensor,
    k_cell_ids: Tensor,
    window_frames: int,
    window_all_keys: bool,
) -> Tensor:
    """Return whether keys fall inside the attention window of the queries.

    This predicate is the single source of truth for window semantics; the
    flex mask_mods, the cached attention mask, and cache eviction all call it.
    Window distance runs over ``*_window_pos``: audio time for the backbone
    (``window_all_keys=False`` — audio time is a non-decreasing cumsum over
    frames, so inserted non-audio frames such as tool-result token bursts do
    not consume audio history, and text keys are never windowed) and frame
    positions for the adapter (``window_all_keys=True`` — its streaming KV
    cache evicts by cell count, so a positional window keeps cached and batch
    forwards identical).

    It doubles as the eviction keep-predicate: window positions never
    decrease, so a key outside the window of position ``p`` is invisible to
    every query at ``p`` or later.
    """
    in_window = (q_window_pos - k_window_pos) <= window_frames
    if window_all_keys:
        return in_window
    return (k_cell_ids < NUM_STREAMS) | in_window


def key_visible(
    q_frame_pos: Tensor,
    k_frame_pos: Tensor,
    q_window_pos: Tensor,
    k_window_pos: Tensor,
    k_cell_ids: Tensor,
    k_active: Tensor,
    window_frames: int,
    window_all_keys: bool,
) -> Tensor:
    """Cross-row key visibility: strict row causality, active keys only, and
    the shared attention window. Same-token self visibility is the caller's
    responsibility (flex adds ``q_idx == kv_idx``; the cached mask adds an
    identity block over the current cells)."""
    prior_row = k_frame_pos < q_frame_pos
    return (
        prior_row
        & k_active
        & key_in_window(
            q_window_pos,
            k_window_pos,
            k_cell_ids,
            window_frames,
            window_all_keys,
        )
    )


__all__ = ["NUM_STREAMS", "key_in_window", "key_visible"]
