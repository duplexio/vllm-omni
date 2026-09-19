# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunk_accumulation_policy_replaces_snapshots_and_drains_delta_state():
    accumulated = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([1.0]),
            "chunk.meta.segment_end": torch.tensor([0]),
            "chunk.meta.tts_is_last_chunk": torch.tensor([0]),
            "chunk.meta.turn_end": torch.tensor([0]),
            "meta.stable_request_value": "keep",
        }
    )
    incoming = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([2.0]),
            "chunk.meta.segment_end": torch.tensor([1]),
            "chunk.meta.tts_is_last_chunk": torch.tensor([1]),
            "chunk.meta.turn_end": torch.tensor([1]),
        }
    )
    assert accumulated is not None
    assert incoming is not None

    merged = accumulated.merged_with(incoming)

    assert not is_non_final_delta_audio_chunk(merged, "audio")

    drain_delta_payload(merged)

    assert "audio" not in merged
    assert "meta.segment_end" not in merged
    assert "meta.tts_is_last_chunk" not in merged
    assert "meta.turn_end" not in merged
    assert merged.metadata["meta.stable_request_value"] == "keep"


def test_arbitrary_chunk_metadata_replaces_and_does_not_leak_into_next_delta():
    accumulated = MultimodalPayload()
    first_output = None
    for token_id in (11, 22):
        incoming = MultimodalPayload.from_dict({
            "audio": torch.ones(1920),
            "chunk": {"new_model_field": torch.tensor(token_id)},
            "meta.request_label": "keep",
        })
        accumulated = accumulated.merged_with(incoming)
        assert accumulated["new_model_field"].item() == token_id
        if first_output is None:
            first_output = accumulated.to_dict()
        drain_delta_payload(accumulated)
        assert "audio" not in accumulated
        assert "new_model_field" not in accumulated
        assert accumulated["meta.request_label"] == "keep"
    assert first_output["new_model_field"].item() == 11


def test_chunk_snapshot_removes_fields_absent_from_next_chunk():
    accumulated = MultimodalPayload.from_dict({
        "chunk.optional_event": torch.tensor(1),
        "chunk.current_value": torch.tensor(11),
    })
    incoming = MultimodalPayload.from_dict({"chunk.current_value": torch.tensor(22)})
    accumulated = accumulated.merged_with(incoming)
    assert "optional_event" not in accumulated
    assert accumulated["current_value"].item() == 22
    accumulated = accumulated.merged_with(MultimodalPayload.from_dict({"audio": torch.ones(1)}))
    assert "current_value" not in accumulated
