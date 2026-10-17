# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.duplexio.frame_layout import NUM_CELLS
from vllm_omni.model_executor.models.duplexio.stream_conv import expand_stream_conv_weight


def test_expanded_gdn_kernel_matches_independent_cell_convolutions() -> None:
    torch.manual_seed(0)
    rows = 9
    channels = 3
    kernel_size = 4
    values = torch.randn(rows, NUM_CELLS, channels)
    weight = torch.randn(channels, 1, kernel_size)

    flattened = values.reshape(rows * NUM_CELLS, channels).T.unsqueeze(0)
    expanded = expand_stream_conv_weight(weight)
    row_major = F.conv1d(
        F.pad(flattened, (expanded.shape[-1] - 1, 0)),
        expanded,
        groups=channels,
    )

    per_cell = []
    for cell in range(NUM_CELLS):
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
