# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The dense attention reference the paged kernels are checked against."""

import pytest
import torch

from tests.model_executor.models.duplexio.reference_attention import attention_visible, key_in_window, key_visible
from vllm_omni.model_executor.models.duplexio.frame_layout import NUM_CELLS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WINDOW = 2

# One conversation, window = 2, text cells 0-3, audio cells 4-5. Frames 3 and
# 4 are token-only bursts, so audio time freezes at 3 while frames advance:
# frame:      0  1  2  3  4  5
# audio time: 1  2  3  3  3  4
# Each row: (q_frame, k_frame, q_audio, k_audio, k_cell, k_active, k_pinned,
#            in_window, visible). Queries sit at frame 5 (audio 4) unless the
# case needs another vantage point.
TRUTH_TABLE = [
    # Text keys are never windowed, however far back they are.
    (5, 0, 4, 1, 0, True, False, True, True),
    (5, 0, 4, 1, 3, True, False, True, True),
    # Inactive keys stay in window (text) but are never cross-frame visible.
    (5, 0, 4, 1, 2, False, False, True, False),
    # Audio key inside the window.
    (5, 4, 4, 3, 4, True, False, True, True),
    # Boundary: audio key exactly at window distance is still visible.
    (5, 1, 4, 2, 5, True, False, True, True),
    # One past the window: evicted for every future query.
    (5, 0, 4, 1, 4, True, False, False, False),
    # The same key pinned as the voice prompt: no window expires it.
    (5, 0, 4, 1, 4, True, True, True, True),
    # Pinning does not make an inactive or same-frame key visible.
    (5, 0, 4, 1, 4, False, True, True, False),
    (5, 5, 4, 4, 4, True, True, True, False),
    # Burst case: frame distance 3 exceeds the window, but the frozen audio
    # time keeps the key visible (audio distance 1).
    (5, 2, 4, 3, 4, True, False, True, True),
    # Same for a query on a burst frame itself (frame 4, audio 3 vs frame 1,
    # audio 2): frame distance 3, audio distance 1.
    (4, 1, 3, 2, 5, True, False, True, True),
    # Same-frame keys are never cross-frame visible (self visibility is the
    # caller's q_idx == kv_idx term), though they are inside the window.
    (5, 5, 4, 4, 4, True, False, True, False),
    (5, 5, 4, 4, 1, True, False, True, False),
]


def test_reference_matches_literal_truth_table() -> None:
    for q_frame, k_frame, q_audio, k_audio, k_cell, k_active, k_pinned, in_window, visible in TRUTH_TABLE:
        case = (q_frame, k_frame, q_audio, k_audio, k_cell, k_active, k_pinned)
        q_audio, k_audio, k_cell, k_pinned = map(torch.tensor, (q_audio, k_audio, k_cell, k_pinned))
        assert bool(key_in_window(q_audio, k_audio, k_cell, k_pinned, WINDOW)) is in_window, case
        assert (
            bool(
                key_visible(
                    torch.tensor(q_frame),
                    torch.tensor(k_frame),
                    q_audio,
                    k_audio,
                    k_cell,
                    torch.tensor(k_active),
                    k_pinned,
                    WINDOW,
                )
            )
            is visible
        ), case


def test_reference_visibility_freezes_the_window_over_frames_without_audio() -> None:
    positions = torch.arange(30)
    # Frame 3 is a token-only burst: audio time freezes, so the window is
    # measured over audio positions [1, 2, 3, 3, 4], not frame indices.
    frame_audio_positions = torch.tensor([1, 1, 1, 0, 1]).cumsum(0)
    audio_positions = frame_audio_positions.repeat_interleave(NUM_CELLS)
    query = positions[:, None]
    key = positions[None, :]
    key_active = torch.ones_like(key, dtype=torch.bool)
    key_active[:, 1] = False

    key_pinned = torch.zeros_like(key, dtype=torch.bool)

    visible = attention_visible(
        query,
        key,
        audio_positions[:, None],
        audio_positions[None, :],
        key_active,
        key_pinned,
        window_frames=2,
    )

    expected = torch.zeros_like(visible)
    for query_position in positions.tolist():
        for key_position in positions.tolist():
            query_frame, _ = divmod(query_position, NUM_CELLS)
            key_frame, key_cell = divmod(key_position, NUM_CELLS)
            active = key_position != 1
            audio_distance = int(frame_audio_positions[query_frame] - frame_audio_positions[key_frame])
            expected[query_position, key_position] = query_position == key_position or (
                key_frame < query_frame and active and (key_cell < 4 or audio_distance <= 2)
            )

    torch.testing.assert_close(visible, expected)
    # Frame 4 sits 3 audio steps from frame 0 (out of the window of 2) but
    # frame 3 sits 4 frames later than frame 0 at audio distance 2 (inside).
    assert not bool(visible[4 * NUM_CELLS + 4, 4])
    assert bool(visible[3 * NUM_CELLS + 4, 4])


def test_pinned_voice_prompt_cells_outlive_the_audio_window() -> None:
    # Two frames of audio a full window apart: frame 0's audio cells expire for
    # frame 1's query, unless they are the pinned voice prompt.
    positions = torch.arange(2 * NUM_CELLS)
    audio_positions = torch.tensor([1, 9]).repeat_interleave(NUM_CELLS)
    query = positions[:, None]
    key = positions[None, :]
    key_active = torch.ones_like(key, dtype=torch.bool)
    audio_cells = (positions % NUM_CELLS) >= 4
    last_row = positions >= NUM_CELLS

    expired = attention_visible(
        query,
        key,
        audio_positions[:, None],
        audio_positions[None, :],
        key_active,
        torch.zeros_like(key, dtype=torch.bool),
        window_frames=2,
    )
    pinned = attention_visible(
        query,
        key,
        audio_positions[:, None],
        audio_positions[None, :],
        key_active,
        (audio_cells & ~last_row)[None, :].expand_as(key),
        window_frames=2,
    )

    first_row_audio = (audio_cells & ~last_row)[None, :].expand_as(expired)
    assert not expired[last_row][:, (audio_cells & ~last_row)].any()
    assert pinned[last_row][:, (audio_cells & ~last_row)].all()
    # Nothing else moves: the two relations differ only on those keys.
    torch.testing.assert_close(pinned & ~first_row_audio, expired & ~first_row_audio)
