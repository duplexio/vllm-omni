# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dense reference for which cached DuplexIO keys a query may attend to.

A query sees its own cell and active keys of earlier frames. Text keys never
expire. Audio keys expire once the query is more than ``window_frames`` audio
frames past them, measured in audio time, which frames without audio (such as
tool-result bursts) do not advance. Pinned voice-prompt keys never expire.
"""

from torch import Tensor

from vllm_omni.model_executor.models.duplexio.frame_layout import NUM_CELLS, NUM_TEXT_CELLS


def key_in_window(q_audio: Tensor, k_audio: Tensor, k_cells: Tensor, k_pinned: Tensor, window_frames: int) -> Tensor:
    return (k_cells < NUM_TEXT_CELLS) | k_pinned | ((q_audio - k_audio) <= window_frames)


def key_visible(
    q_frames: Tensor,
    k_frames: Tensor,
    q_audio: Tensor,
    k_audio: Tensor,
    k_cells: Tensor,
    k_active: Tensor,
    k_pinned: Tensor,
    window_frames: int,
) -> Tensor:
    """Visibility of keys in earlier frames; a query's own cell is the caller's to add."""
    return (k_frames < q_frames) & k_active & key_in_window(q_audio, k_audio, k_cells, k_pinned, window_frames)


def attention_visible(
    query_positions: Tensor,
    key_positions: Tensor,
    query_audio: Tensor,
    key_audio: Tensor,
    key_active: Tensor,
    key_pinned: Tensor,
    *,
    window_frames: int,
) -> Tensor:
    """Visibility between flattened cell positions, six cells per frame."""
    return key_visible(
        query_positions // NUM_CELLS,
        key_positions // NUM_CELLS,
        query_audio,
        key_audio,
        key_positions % NUM_CELLS,
        key_active,
        key_pinned,
        window_frames,
    ) | (query_positions == key_positions)
