from __future__ import annotations

import torch

from vllm_omni.model_executor.models.moss_tts.session import MossTTSDSessionStore


def reference_codes(value: int) -> torch.Tensor:
    return torch.full((3, 4), value, dtype=torch.long)


def prompt_codes() -> torch.Tensor:
    return torch.full((6, 4), 9, dtype=torch.long)


def test_session_commits_transcript_and_audio_atomically() -> None:
    store = MossTTSDSessionStore()
    session = store.create(
        "conversation",
        ("User reference.", "Assistant reference."),
        (reference_codes(1), reference_codes(2)),
        prompt_codes(),
    )
    turn = store.begin_turn("conversation", "user", "Can you help?")

    assert turn.full_text == "[S1] User reference. [S2] Assistant reference. [S1] Can you help?"
    assert turn.audio_prefix_codes.shape == (6, 4)
    assert torch.equal(turn.audio_prefix_codes, prompt_codes())

    revision = store.commit_turn(
        turn,
        generated_codes=torch.full((5, 4), 3),
    )

    assert revision == 1
    assert session.audio_prefix_codes.shape == (11, 4)
    assert [segment.render() for segment in session.completed_segments] == [
        "[S1] Can you help?"
    ]
    assert not session.in_flight


def test_abort_preserves_the_previous_revision() -> None:
    store = MossTTSDSessionStore()
    session = store.create(
        "conversation",
        ("User reference.", "Assistant reference."),
        (reference_codes(1), reference_codes(2)),
        prompt_codes(),
    )
    turn = store.begin_turn("conversation", "assistant", "Of course.")

    store.abort_turn(turn)

    assert session.revision == 0
    assert session.completed_segments == []
    assert session.audio_prefix_codes.shape == (6, 4)
    assert not session.in_flight


def test_sessions_have_isolated_cache_salts_and_prefixes() -> None:
    store = MossTTSDSessionStore()
    first = store.create(
        "first",
        ("Same user.", "Same assistant."),
        (reference_codes(1), reference_codes(2)),
        prompt_codes(),
    )
    second = store.create(
        "second",
        ("Same user.", "Same assistant."),
        (reference_codes(1), reference_codes(2)),
        prompt_codes(),
    )

    first_turn = store.begin_turn("first", "user", "Hello.")
    second_turn = store.begin_turn("second", "user", "Hello.")

    assert first.cache_salt != second.cache_salt
    assert first_turn.cache_salt == first.cache_salt
    assert second_turn.cache_salt == second.cache_salt


def test_session_rejects_overlapping_turns() -> None:
    store = MossTTSDSessionStore()
    store.create(
        "conversation",
        ("User reference.", "Assistant reference."),
        (reference_codes(1), reference_codes(2)),
        prompt_codes(),
    )
    store.begin_turn("conversation", "user", "First.")

    try:
        store.begin_turn("conversation", "assistant", "Second.")
    except RuntimeError as exc:
        assert "in-flight" in str(exc)
    else:
        raise AssertionError("overlapping turn was accepted")


def test_adjacent_same_speaker_turns_preserve_each_turn_boundary() -> None:
    store = MossTTSDSessionStore()
    session = store.create(
        "conversation",
        ("User reference.", "Assistant reference."),
        (reference_codes(1), reference_codes(2)),
        prompt_codes(),
    )

    first = store.begin_turn("conversation", "assistant", "Running that test now.")
    store.commit_turn(first, generated_codes=torch.full((5, 4), 3))
    second = store.begin_turn("conversation", "assistant", "Great news, it passed.")

    assert second.full_text == (
        "[S1] User reference. [S2] Assistant reference. "
        "[S2] Running that test now. [S2] Great news, it passed."
    )
    assert second.audio_prefix_codes.shape == (11, 4)

    store.commit_turn(second, generated_codes=torch.full((7, 4), 4))

    assert [segment.render() for segment in session.completed_segments] == [
        "[S2] Running that test now.",
        "[S2] Great news, it passed.",
    ]
    assert session.audio_prefix_codes.shape == (18, 4)

    user = store.begin_turn("conversation", "user", "What changed?")

    assert user.full_text == (
        "[S1] User reference. [S2] Assistant reference. "
        "[S2] Running that test now. [S2] Great news, it passed. [S1] What changed?"
    )
