"""The serving argmax mode must not randomly suppress emitted text."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    TokenSamplingOptions,
    _sample_factorized_text_ids,
)


def test_argmax_emission_is_deterministic_at_default_temperature() -> None:
    generator = torch.Generator().manual_seed(37)
    initial_rng = generator.get_state().clone()
    logits = torch.tensor([[0.0, 1.0, 3.0]]).expand(128, -1)
    result = _sample_factorized_text_ids(
        logits,
        torch.full((128,), 0.01),
        silence_token_id=0,
        sampling=TokenSamplingOptions("argmax", 1.0, 3, 1.0, torch.tensor([0], dtype=torch.long)),
        emit_temperature=1.0,
        generator=generator,
    )
    assert result.tolist() == [2] * 128
    assert torch.equal(initial_rng, generator.get_state())


@pytest.mark.parametrize("mode", ["argmax", "top_k", "top_p"])
def test_factorized_tokens_match_training(mode: str) -> None:
    training = pytest.importorskip("duplexio.models.duplexio")
    torch.manual_seed(12)
    logits = torch.randn(7, 31)
    emit_logits = torch.randn(7)
    options = training.TokenSamplingOptions(
        mode=mode,
        temperature=0.8,
        top_k=12,
        top_p=0.9,
        emit_temperature=1.0,
    )
    torch.manual_seed(37)
    expected = training._sample_factorized_token_ids(logits, emit_logits, 0, options)
    actual = _sample_factorized_text_ids(
        logits,
        emit_logits,
        silence_token_id=0,
        sampling=TokenSamplingOptions(mode, 0.8, 12, 0.9, torch.tensor([0], dtype=torch.long)),
        emit_temperature=1.0,
        generator=torch.Generator().manual_seed(37),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
