# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.session import MossTTSRealtimeSessionStore

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def refs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.ones((3, 16), dtype=torch.long),
        torch.full((4, 16), 2, dtype=torch.long),
    )


def test_realtime_session_replays_completed_turns_in_order() -> None:
    store = MossTTSRealtimeSessionStore()
    session = store.create("conversation", refs())

    first = store.begin_turn("conversation", "user", "Hello.")
    assert first.history_segments == ()
    store.commit_turn(first, torch.full((5, 16), 3, dtype=torch.long))

    second = store.begin_turn("conversation", "assistant", "Hi there.")
    assert [(segment.role, segment.text) for segment in second.history_segments] == [("user", "Hello.")]
    torch.testing.assert_close(second.history_segments[0].codes, torch.full((5, 16), 3, dtype=torch.long))

    store.commit_turn(second, torch.full((6, 16), 4, dtype=torch.long))
    assert [(segment.role, segment.text) for segment in session.completed_segments] == [
        ("user", "Hello."),
        ("assistant", "Hi there."),
    ]
    assert session.revision == 2

    third = store.begin_turn("conversation", "user", "Thanks.")
    assert len(third.history_segments) == 2


def test_realtime_session_rejects_wrong_codebook_count() -> None:
    store = MossTTSRealtimeSessionStore()
    session = store.create("conversation", refs())
    turn = store.begin_turn("conversation", "user", "Hello.")

    with pytest.raises(ValueError, match="codebooks differ"):
        store.commit_turn(turn, torch.ones((2, 8), dtype=torch.long))

    assert session.in_flight
    store.abort_turn(turn)
    assert not session.in_flight
