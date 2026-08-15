# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.realtime_prompt import build_realtime_prompt
from vllm_omni.model_executor.models.moss_tts.session import MossTTSRealtimeSegment

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        if text == "<|im_start|>assistant\n":
            return [90, 91]
        return [ord(char) for char in text]


class FakeRealtimeProcessor:
    channels = 2
    audio_channel_pad = 1024
    audio_bos_token = 1025
    delay_tokens_len = 2
    text_pad_token_id = 151655
    tokenizer = FakeTokenizer()

    def make_ensemble(self, prompt_audio_tokens: np.ndarray) -> np.ndarray:
        grid = np.full((prompt_audio_tokens.shape[0], 3), 1024, dtype=np.int64)
        grid[:, 0] = 10
        grid[:, 1:] = prompt_audio_tokens
        return grid

    def make_user_prompt(self, text: str, audio_tokens: np.ndarray) -> np.ndarray:
        grid = np.full((len(text) + len(audio_tokens) + 1, 3), 1024, dtype=np.int64)
        grid[:, 0] = 20
        grid[-len(audio_tokens) - 1 : -1, 1:] = audio_tokens
        grid[-1, 1] = 1026
        return grid


def test_first_turn_uses_reference_and_prefills_only_twelve_tokens() -> None:
    prompt = build_realtime_prompt(
        FakeRealtimeProcessor(),
        text="abcdefghijklmnop",
        reference_codes=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
    )

    assert prompt.text_ids[:2] == [10, 10]
    assert prompt.text_ids[2:4] == [90, 91]
    assert prompt.remaining_text_ids == [ord(char) for char in "mnop"]
    assert prompt.audio_codes.shape == (2 + 2 + 12, 2)
    assert prompt.audio_codes[-1, 0].item() == 1025


def test_next_turn_replays_completed_user_and_assistant_inputs() -> None:
    prompt = build_realtime_prompt(
        FakeRealtimeProcessor(),
        text="next",
        reference_codes=torch.tensor([[8, 9]], dtype=torch.long),
        history_segments=(
            MossTTSRealtimeSegment(
                role="user",
                text="prior",
                codes=torch.tensor([[5, 6], [7, 8]], dtype=torch.long),
            ),
            MossTTSRealtimeSegment(
                role="assistant",
                text="answer",
                codes=torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.long),
            ),
        ),
    )

    assert prompt.text_ids[:1] == [10]
    assert prompt.text_ids[1 : 1 + len("prior") + 2 + 1] == [20] * (len("prior") + 3)
    assistant_history_start = 1 + len("prior") + 2 + 1
    assert prompt.text_ids[assistant_history_start : assistant_history_start + 2] == [ord("a"), ord("n")]
    continuation_start = assistant_history_start + 2
    assert prompt.text_ids[continuation_start : continuation_start + 3] == [ord("s"), ord("w"), ord("e")]
    assert prompt.remaining_text_ids == []
    torch.testing.assert_close(
        prompt.audio_codes[1 + len("prior") : 1 + len("prior") + 2],
        torch.tensor([[5, 6], [7, 8]]),
    )
    torch.testing.assert_close(
        prompt.audio_codes[continuation_start : continuation_start + 3],
        torch.tensor([[11, 12], [13, 14], [15, 16]]),
    )


def test_assistant_history_requires_audio_codes() -> None:
    with pytest.raises(ValueError, match="audio codes"):
        build_realtime_prompt(
            FakeRealtimeProcessor(),
            text="hello",
            reference_codes=torch.ones((1, 2), dtype=torch.long),
            history_segments=(
                MossTTSRealtimeSegment(role="user", text="prior", codes=torch.empty((0, 2), dtype=torch.long)),
            ),
        )
