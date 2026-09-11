# SPDX-License-Identifier: Apache-2.0
"""Incremental greedy RNN-T decoding for the live user text stream."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_omni.model_executor.models.duplexio.fastconformer import (
    WORD_END_SILENCE_FRAMES,
    FastConformerRNNT,
    RNNTGreedyState,
)

BLANK = 3
PIECES = {0: "▁hel", 1: "lo", 2: "▁world", 3: "<blank>", 4: "."}


class ScriptedJoint(nn.Module):
    """Emit a fixed token per joint call, so the greedy schedule is under test."""

    def __init__(self, token_ids: list[int]) -> None:
        super().__init__()
        self.token_ids = token_ids
        self.calls = 0

    def forward(self, *, encoder_hidden_states, decoder_hidden_states):
        # Past the script the model is silent, which is what a real one does once
        # the speech ends.
        token_id = (
            self.token_ids[self.calls] if self.calls < len(self.token_ids) else BLANK
        )
        self.calls += 1
        logits = torch.full((1, 1, 1, len(PIECES)), -10.0)
        logits[..., token_id] = 10.0
        return logits


def build_encoder(token_ids: list[int]) -> FastConformerRNNT:
    hidden_size = 4
    model = SimpleNamespace(
        config=SimpleNamespace(
            encoder_config=SimpleNamespace(hidden_size=hidden_size),
            blank_token_id=BLANK,
        ),
        encoder_projector=nn.Identity(),
        decoder=SimpleNamespace(
            embedding=nn.Embedding(len(PIECES), hidden_size),
            lstm=nn.LSTM(hidden_size, hidden_size, batch_first=True),
            decoder_projector=nn.Identity(),
        ),
        joint=ScriptedJoint(token_ids),
        max_symbols_per_step=10,
        requires_grad_=lambda _: None,
        eval=lambda: None,
    )
    processor = SimpleNamespace(
        set_num_lookahead_tokens=lambda _: None,
        tokenizer=SimpleNamespace(
            convert_ids_to_tokens=lambda token_id: PIECES[token_id],
            decode=lambda ids: "".join(PIECES[i] for i in ids).replace("▁", ""),
        ),
        feature_extractor=SimpleNamespace(
            hop_length=160,
            n_fft=512,
            win_length=400,
            preemphasis=0.97,
            mel_filters=torch.zeros(128, 257),
        ),
    )
    return FastConformerRNNT(model, processor)


def test_words_are_released_once_the_next_word_starts() -> None:
    # Two encoder frames: "▁hel" "lo" on the first, "▁world" on the second.
    encoder = build_encoder([0, 1, BLANK, 2, BLANK])
    encoded = torch.zeros(1, 2, 4)

    words, state = encoder.decode_words(encoded, RNNTGreedyState())

    # "hello" is complete once "▁world" starts; "world" itself is still open.
    assert words == (" hello",)
    assert state.word_token_ids == (2,)
    assert encoder.model.joint.calls == 5


def test_decode_resumes_across_appends() -> None:
    encoder = build_encoder([0, BLANK, 1, BLANK, 2, BLANK])
    frame = torch.zeros(1, 1, 4)

    words, state = encoder.decode_words(frame, RNNTGreedyState())
    assert words == ()
    assert state.word_token_ids == (0,)

    # The second 80 ms append finishes the word; nothing is released yet.
    words, state = encoder.decode_words(frame, state)
    assert words == ()
    assert state.word_token_ids == (0, 1)

    words, state = encoder.decode_words(frame, state)
    assert words == (" hello",)
    assert state.word_token_ids == (2,)


def test_silence_releases_the_last_word_of_a_turn() -> None:
    # "▁hel" "lo" and then nothing more: no next word will ever start.
    encoder = build_encoder([0, 1, BLANK] + [BLANK] * WORD_END_SILENCE_FRAMES)
    frame = torch.zeros(1, 1, 4)

    words, state = encoder.decode_words(frame, RNNTGreedyState())
    assert words == ()

    for _ in range(WORD_END_SILENCE_FRAMES - 1):
        words, state = encoder.decode_words(frame, state)
        assert words == ()
    words, state = encoder.decode_words(frame, state)

    assert words == (" hello",)
    assert state.word_token_ids == ()


def test_punctuation_after_a_release_carries_no_space() -> None:
    # "▁hel" "lo", a pause long enough to release it, then a late ".".
    encoder = build_encoder([0, 1, BLANK] + [BLANK] * WORD_END_SILENCE_FRAMES + [4])
    frame = torch.zeros(1, 1, 4)

    state = RNNTGreedyState()
    for _ in range(1 + WORD_END_SILENCE_FRAMES):
        words, state = encoder.decode_words(frame, state)
    assert words == (" hello",)

    # The late "." starts no word, so its own release glues onto "hello".
    for _ in range(1 + WORD_END_SILENCE_FRAMES):
        words, state = encoder.decode_words(frame, state)
    assert words == (".",)


def test_max_symbols_per_step_forces_a_frame_advance() -> None:
    encoder = build_encoder([1] * 10)
    encoder.model.max_symbols_per_step = 10

    words, state = encoder.decode_words(torch.zeros(1, 1, 4), RNNTGreedyState())

    # A model that never emits blank must not spin on one encoder frame.
    assert words == ()
    assert state.word_token_ids == (1,) * 10
    assert encoder.model.joint.calls == 10
