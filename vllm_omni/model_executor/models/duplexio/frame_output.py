# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed per-frame DuplexIO outputs.

Each request's per-frame scalars travel as one int64 ``frame`` tensor and, when
the frame predicted, one float32 ``frame_logprobs`` tensor with the columns
``FRAME_LOGPROBS``. Every tensor in an output is encoded, sent, decoded and copied
separately, so thirty scalar tensors per request made the driver's host work
scale with fields, not bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FRAME_FIELDS = (
    "duplex_epoch",
    "duplex_turn_id",
    "prefix",  # The append carried the voice prompt and system tokens.
    "tool_result",  # The append carried tool-result rows after its live frame,
    "tool_generation",  # the newest of which had this generation.
    "given",  # The live frame heard given history instead of the last prediction.
    "end_of_turn",
    "predicted",
    "model_listen",
    "tool_call_complete",
    "tool_emit_sampled",
    "user_emit",
    "user_token_id",
    "agent_token_id",
    "tool_call_token_id",
    "policy_version",
    "sample_rate_hz",
)
FRAME_INDEX = {name: index for index, name in enumerate(FRAME_FIELDS)}
FRAME_FLAGS = frozenset({
    "prefix", "tool_result", "given", "end_of_turn", "predicted", "model_listen", "tool_call_complete",
    "tool_emit_sampled", "user_emit",
})
# ``tool_emit_sampled`` marks frames whose tool emit was drawn; inside a call or
# without a grammar it is forced, and its logprob is only the raw head's score.

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


def frame_row(**fields: int | bool) -> list[int]:
    """A ``frame``'s values in field order; every field must be named."""
    if fields.keys() != FRAME_INDEX.keys():
        raise KeyError(f"DuplexIO frame fields differ: {sorted(fields.keys() ^ FRAME_INDEX.keys())}")
    return [int(fields[name]) for name in FRAME_FIELDS]
