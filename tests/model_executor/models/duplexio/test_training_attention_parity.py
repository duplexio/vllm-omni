# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The attention reference agrees with the training implementation."""

import pytest
import torch

from tests.model_executor.models.duplexio.reference_attention import key_in_window, key_visible

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reference_matches_training_repo() -> None:
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
    k_pinned = torch.rand(size, generator=generator) < 0.2
    for window_frames in (0, 1, 3, 4_096):
        torch.testing.assert_close(
            key_in_window(
                q_audio,
                k_audio,
                k_cell,
                k_pinned,
                window_frames,
            ),
            training_in_window(
                q_audio,
                k_audio,
                k_cell,
                k_pinned,
                window_frames,
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
                k_pinned,
                window_frames,
            ),
            training_visible(
                q_frame,
                k_frame,
                q_audio,
                k_audio,
                k_cell,
                k_active,
                k_pinned,
                window_frames,
            ),
        )
