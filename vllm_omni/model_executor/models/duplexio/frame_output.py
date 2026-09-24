# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed per-frame DuplexIO outputs.

Each request's per-frame scalars travel as one int64 ``frame`` tensor and, when
the frame predicted, one float32 ``frame_logprobs`` tensor. Every tensor in an
output is encoded, sent, decoded and copied separately, so thirty scalar
tensors per request made the driver's host work scale with fields, not bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

FRAME_FIELDS = (
    "duplex_epoch",
    "duplex_turn_id",
    "duplex_prefill",
    "duplex_prefill_complete",
    "duplex_system_input",
    "duplex_system_input_complete",
    "end_of_turn",
    "predicted",
    "model_listen",
    "tool_call_complete",
    "user_emit",
    "user_token_id",
    "agent_token_id",
    "tool_call_token_id",
    "policy_version",
    "sample_rate_hz",
)
FRAME_INDEX = {name: index for index, name in enumerate(FRAME_FIELDS)}
FRAME_FLAGS = frozenset({
    "duplex_prefill", "duplex_prefill_complete", "duplex_system_input", "duplex_system_input_complete",
    "end_of_turn", "predicted", "model_listen", "tool_call_complete", "user_emit",
})
# Columns of ``frame_logprobs``.
FRAME_LOGPROBS = (
    "agent_emit_logprob", "agent_token_logprob", "tool_emit_logprob", "tool_token_logprob",
    "user_emit_logprob", "user_token_logprob",
)


def frame_fields(output: Mapping[str, Any]) -> dict[str, int | bool]:
    """Unpack an output's ``frame`` into named Python scalars with one read."""
    values = output["frame"].tolist()
    return {
        name: bool(value) if name in FRAME_FLAGS else value
        for name, value in zip(FRAME_FIELDS, values, strict=True)
    }


def pack_frame(**fields: int | bool) -> Tensor:
    """Build a ``frame`` from named fields; unnamed fields are zero."""
    unknown = fields.keys() - FRAME_INDEX.keys()
    if unknown:
        raise KeyError(f"Unknown DuplexIO frame fields: {sorted(unknown)}")
    return torch.tensor([int(fields.get(name, 0)) for name in FRAME_FIELDS], dtype=torch.long)
