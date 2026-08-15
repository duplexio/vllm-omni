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


def test_realtime_session_exposes_previous_turn_and_commits_new_audio() -> None:
    store = MossTTSRealtimeSessionStore()
    session = store.create("conversation", refs())

    first = store.begin_turn("conversation", "user", "Hello.")
    assert first.previous_text is None
    assert first.previous_codes is None
    store.commit_turn(first, torch.full((5, 16), 3, dtype=torch.long))

    second = store.begin_turn("conversation", "assistant", "Hi there.")
    assert second.previous_text == "Hello."
    assert second.previous_codes is not None
    torch.testing.assert_close(second.previous_codes, torch.full((5, 16), 3, dtype=torch.long))

    store.commit_turn(second, torch.full((6, 16), 4, dtype=torch.long))
    assert session.previous_text == "Hi there."
    assert session.previous_role == "assistant"
    assert session.revision == 2

    third = store.begin_turn("conversation", "user", "Thanks.")
    assert third.previous_text is None
    assert third.previous_codes is None


def test_realtime_session_rejects_wrong_codebook_count() -> None:
    store = MossTTSRealtimeSessionStore()
    session = store.create("conversation", refs())
    turn = store.begin_turn("conversation", "user", "Hello.")

    with pytest.raises(ValueError, match="codebooks differ"):
        store.commit_turn(turn, torch.ones((2, 8), dtype=torch.long))

    assert session.in_flight
    store.abort_turn(turn)
    assert not session.in_flight
