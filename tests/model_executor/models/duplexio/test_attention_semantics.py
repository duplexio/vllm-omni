# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the window predicates vendored from the training repo."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.attention_semantics import (
    key_in_window,
    key_visible,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WINDOW = 2

# One conversation, window = 2, text cells 0-3, audio cells 4-5. Frames 3 and
# 4 are token-only bursts, so audio time freezes at 3 while frames advance:
# frame:      0  1  2  3  4  5
# audio time: 1  2  3  3  3  4
# Each row: (q_frame, k_frame, q_audio, k_audio, k_cell, k_active,
#            in_window, visible). Queries sit at frame 5 (audio 4) unless the
# case needs another vantage point.
TRUTH_TABLE = [
    # Text keys are never windowed, however far back they are.
    (5, 0, 4, 1, 0, True, True, True),
    (5, 0, 4, 1, 3, True, True, True),
    # Inactive keys stay in window (text) but are never cross-frame visible.
    (5, 0, 4, 1, 2, False, True, False),
    # Audio key inside the window.
    (5, 4, 4, 3, 4, True, True, True),
    # Boundary: audio key exactly at window distance is still visible.
    (5, 1, 4, 2, 5, True, True, True),
    # One past the window: evicted for every future query.
    (5, 0, 4, 1, 4, True, False, False),
    # Burst case: frame distance 3 exceeds the window, but the frozen audio
    # time keeps the key visible (audio distance 1).
    (5, 2, 4, 3, 4, True, True, True),
    # Same for a query on a burst frame itself (frame 4, audio 3 vs frame 1,
    # audio 2): frame distance 3, audio distance 1.
    (4, 1, 3, 2, 5, True, True, True),
    # Same-frame keys are never cross-frame visible (self visibility is the
    # caller's q_idx == kv_idx term), though they are inside the window.
    (5, 5, 4, 4, 4, True, True, False),
    (5, 5, 4, 4, 1, True, True, False),
]


def test_vendored_predicates_match_literal_truth_table() -> None:
    for case in TRUTH_TABLE:
        (
            q_frame,
            k_frame,
            q_audio,
            k_audio,
            k_cell,
            k_active,
            in_window,
            visible,
        ) = case
        assert bool(
            key_in_window(
                torch.tensor(q_audio),
                torch.tensor(k_audio),
                torch.tensor(k_cell),
                WINDOW,
                False,
            )
        ) is in_window, case
        assert bool(
            key_visible(
                torch.tensor(q_frame),
                torch.tensor(k_frame),
                torch.tensor(q_audio),
                torch.tensor(k_audio),
                torch.tensor(k_cell),
                torch.tensor(k_active),
                WINDOW,
                False,
            )
        ) is visible, case


def test_window_all_keys_windows_text_keys_too() -> None:
    # The adapter variant windows every key by positional distance.
    assert not bool(
        key_in_window(
            torch.tensor(5),
            torch.tensor(1),
            torch.tensor(0),
            WINDOW,
            True,
        )
    )
    assert bool(
        key_in_window(
            torch.tensor(5),
            torch.tensor(3),
            torch.tensor(0),
            WINDOW,
            True,
        )
    )


def test_vendored_predicates_match_training_repo() -> None:
    try:
        from duplexio.models.duplexio import key_in_window as training_in_window
        from duplexio.models.duplexio import key_visible as training_visible
    except ImportError:
        pytest.skip("duplexio training package is not importable")

    generator = torch.Generator().manual_seed(23)
    size = (4_096,)
    k_frame = torch.randint(0, 40, size, generator=generator)
    q_frame = k_frame + torch.randint(-2, 8, size, generator=generator)
    k_audio = torch.randint(0, 30, size, generator=generator)
    q_audio = k_audio + torch.randint(-2, 8, size, generator=generator)
    k_cell = torch.randint(0, 6, size, generator=generator)
    k_active = torch.rand(size, generator=generator) < 0.7
    for window_frames in (0, 1, 3, 4_096):
        for window_all_keys in (False, True):
            torch.testing.assert_close(
                key_in_window(
                    q_audio,
                    k_audio,
                    k_cell,
                    window_frames,
                    window_all_keys,
                ),
                training_in_window(
                    q_audio,
                    k_audio,
                    k_cell,
                    window_frames,
                    window_all_keys,
                ),
            )
            torch.testing.assert_close(
                key_visible(
                    q_frame,
                    k_frame,
                    q_audio,
                    k_audio,
                    k_cell,
                    k_active,
                    window_frames,
                    window_all_keys,
                ),
                training_visible(
                    q_frame,
                    k_frame,
                    q_audio,
                    k_audio,
                    k_cell,
                    k_active,
                    window_frames,
                    window_all_keys,
                ),
            )
