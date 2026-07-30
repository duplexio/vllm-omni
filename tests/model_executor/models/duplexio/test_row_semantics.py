# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    duplexio_attention_visible,
    duplexio_cell_ids,
    duplexio_frame_positions,
    expand_stream_conv_weight,
    mask_inactive_gdn_gates,
)


def test_duplexio_attention_visibility_matches_row_contract() -> None:
    positions = torch.arange(30)
    query = positions[:, None]
    key = positions[None, :]
    key_active = torch.ones_like(key, dtype=torch.bool)
    key_active[:, 1] = False

    visible = duplexio_attention_visible(
        query,
        key,
        key_active,
        audio_attention_window_frames=2,
    )

    expected = torch.zeros_like(visible)
    for query_position in positions.tolist():
        for key_position in positions.tolist():
            query_frame, _ = divmod(query_position, DUPLEXIO_NUM_CELLS)
            key_frame, key_cell = divmod(key_position, DUPLEXIO_NUM_CELLS)
            active = key_position != 1
            expected[query_position, key_position] = (
                query_position == key_position
                or (
                    key_frame < query_frame
                    and active
                    and (key_cell < 4 or query_frame - key_frame <= 2)
                )
            )

    torch.testing.assert_close(visible, expected)


def test_expanded_gdn_kernel_matches_independent_cell_convolutions() -> None:
    torch.manual_seed(0)
    rows = 9
    channels = 3
    kernel_size = 4
    values = torch.randn(rows, DUPLEXIO_NUM_CELLS, channels)
    weight = torch.randn(channels, 1, kernel_size)

    flattened = values.reshape(rows * DUPLEXIO_NUM_CELLS, channels).T.unsqueeze(0)
    expanded = expand_stream_conv_weight(weight, num_cells=DUPLEXIO_NUM_CELLS)
    row_major = F.conv1d(
        F.pad(flattened, (expanded.shape[-1] - 1, 0)),
        expanded,
        groups=channels,
    )

    per_cell = []
    for cell in range(DUPLEXIO_NUM_CELLS):
        column = values[:, cell].T.unsqueeze(0)
        per_cell.append(
            F.conv1d(
                F.pad(column, (kernel_size - 1, 0)),
                weight,
                groups=channels,
            )
        )
    expected = torch.stack(per_cell, dim=3).reshape_as(row_major)

    torch.testing.assert_close(row_major, expected)


def test_inactive_gdn_gates_are_exact_recurrent_noops() -> None:
    beta_logits = torch.tensor([[0.0, 2.0], [1.0, -1.0]])
    decay_logits = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    active = torch.tensor([True, False])

    beta_logits, decay_logits = mask_inactive_gdn_gates(
        beta_logits,
        decay_logits,
        active,
    )

    assert torch.equal(beta_logits[0], torch.tensor([0.0, 2.0]))
    assert torch.equal(decay_logits[0], torch.tensor([3.0, 4.0]))
    assert torch.equal(torch.sigmoid(beta_logits[1]), torch.zeros(2))
    assert torch.equal(F.softplus(decay_logits[1]), torch.zeros(2))


def test_cell_and_frame_positions_share_one_rope_position_per_row() -> None:
    positions = torch.arange(18)

    assert torch.equal(
        duplexio_frame_positions(positions),
        torch.arange(3).repeat_interleave(DUPLEXIO_NUM_CELLS),
    )
    assert torch.equal(
        duplexio_cell_ids(positions),
        torch.arange(DUPLEXIO_NUM_CELLS).repeat(3),
    )
