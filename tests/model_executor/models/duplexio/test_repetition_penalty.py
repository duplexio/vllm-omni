# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A stream's repetition penalty reads only its own recent words, and never the tool grammar's."""

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.duplexio.reference_sampling import sample_text_batch
from tests.model_executor.models.duplexio.test_batched_tool_sampling import sampling_model
from vllm_omni.model_executor.models.duplexio.text_sampling import REPETITION_PENALTY, penalize_repetitions

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
]


def test_penalty_divides_positive_and_multiplies_negative_logits_of_recent_tokens() -> None:
    logits = torch.tensor([[[2.0, -2.0, 4.0, 1.0]]])
    recent = torch.tensor([[[0, 1, -1, -1]]])
    penalized = penalize_repetitions(logits, recent, torch.tensor([[2.0]]))
    torch.testing.assert_close(penalized, torch.tensor([[[1.0, -4.0, 4.0, 1.0]]]))
    # A penalty of one, or no recent words, leaves the logits as they were.
    torch.testing.assert_close(penalize_repetitions(logits, recent, torch.tensor([[1.0]])), logits)
    torch.testing.assert_close(penalize_repetitions(logits, torch.full_like(recent, -1), torch.tensor([[2.0]])), logits)


def penalized_session(model, penalty: float, window: int = 64):
    content = {"temperature": 0.0, "repetition_penalty": penalty, "repetition_window": window}
    sampling = model.resolve_sampling({"agent": {"content": content}, "user": {"content": content}})
    return SimpleNamespace(tool_call_constraint=None, sampling=sampling, recent_text_ids=((), ()))


@pytest.mark.parametrize("device", DEVICES)
def test_greedy_streams_move_off_a_word_they_just_said(device) -> None:
    vocab = 6
    model = sampling_model(vocab, device)
    model.pad_token_id = vocab - 1
    states = [penalized_session(model, 4.0), penalized_session(model, 1.0)]
    # Every stream prefers 3, then 2; silence (0) is suppressed.
    logits = torch.tensor([0.0, 0.0, 2.0, 3.0, 0.0, 0.0], device=device).expand(2, 3, vocab).contiguous()
    emit = torch.full((2, 3), 10.0, device=device)
    for state in states:
        # User, agent and tool words of one live row.
        model.record_recent_words(state, (3, 3, 3))
    infos = [{"duplexio_working_state": state} for state in states]
    inputs = model.sampling_inputs(states, torch.device(device))
    # Logits order (agent, tool, user): the tool stream is neither penalized nor remembered.
    assert inputs.parameters[:, REPETITION_PENALTY].tolist() == [[4.0, 1.0, 4.0], [1.0, 1.0, 1.0]]
    assert inputs.recent_ids[0, :, 0].tolist() == [3, -1, 3]
    assert (inputs.recent_ids[0, :, 1:] == -1).all() and (inputs.recent_ids[1] == -1).all()
    text_ids = sample_text_batch(model, logits, emit, infos).text_ids.tolist()
    # Rows are (user, agent, tool): the penalized session moves to 2; no grammar, so no tool word.
    assert text_ids == [[2, 2, 0], [3, 3, 0]]


def test_only_penalized_streams_keep_their_last_words() -> None:
    model = sampling_model(6, "cpu")
    model.pad_token_id = 5
    state = penalized_session(model, 1.3, window=2)
    for tokens in ((1, 2, 0), (0, 3, 0), (5, 4, 0), (1, 0, 0)):
        model.record_recent_words(state, tokens)
    # Agent then user; silence (0) and padding (5) are not words.
    assert state.recent_text_ids == ((3, 4), (1, 1))
    unpenalized = penalized_session(model, 1.0)
    model.record_recent_words(unpenalized, (1, 2, 0))
    assert unpenalized.recent_text_ids == ((), ())
