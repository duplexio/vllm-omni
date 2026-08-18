# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
    replace_snapshot_keys,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunk_accumulation_policy_replaces_snapshots_and_drains_delta_state():
    accumulated = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([1.0]),
            "meta.segment_end": torch.tensor([0]),
            "meta.tts_is_last_chunk": torch.tensor([0]),
            "meta.turn_end": torch.tensor([0]),
            "meta.stable_request_value": "keep",
        }
    )
    incoming = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([2.0]),
            "meta.segment_end": torch.tensor([1]),
            "meta.tts_is_last_chunk": torch.tensor([1]),
            "meta.turn_end": torch.tensor([1]),
        }
    )
    assert accumulated is not None
    assert incoming is not None

    replace_snapshot_keys(accumulated, incoming)
    merged = accumulated.merged_with(incoming)

    assert not is_non_final_delta_audio_chunk(merged, "audio")

    drain_delta_payload(merged)

    assert "audio" not in merged
    assert "meta.segment_end" not in merged
    assert "meta.tts_is_last_chunk" not in merged
    assert "meta.turn_end" not in merged
    assert merged.metadata["meta.stable_request_value"] == "keep"


def test_duplexio_frame_keys_replace_instead_of_accumulating():
    """Per-frame duplex snapshot keys (token ids, flags, sample rate) must be
    replaced each append; only the audio chunk itself is drainable content."""
    accumulated = MultimodalPayload()
    for frame, token_id in enumerate((11, 22)):
        incoming = MultimodalPayload.from_dict(
            {
                "audio": torch.full((1920,), float(frame)),
                "user_token_id": torch.tensor(token_id),
                "agent_token_id": torch.tensor(token_id + 1),
                "tool_call_token_id": torch.tensor(token_id + 2),
                "model_listen": torch.tensor(True),
                "end_of_turn": torch.tensor(False),
                "sample_rate_hz": torch.tensor(24_000),
            }
        )
        assert incoming is not None
        replace_snapshot_keys(accumulated, incoming)
        accumulated = accumulated.merged_with(incoming)
        drain_delta_payload(accumulated)

        assert "audio" not in accumulated

    assert int(accumulated.get("user_token_id")) == 22
    assert int(accumulated.get("agent_token_id")) == 23
    assert int(accumulated.get("tool_call_token_id")) == 24
    assert int(accumulated.get("sample_rate_hz")) == 24_000
    for key in ("user_token_id", "agent_token_id", "model_listen"):
        assert not isinstance(accumulated.get(key), list), key
