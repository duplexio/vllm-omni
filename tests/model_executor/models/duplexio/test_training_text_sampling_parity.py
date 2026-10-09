# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The reference sampler that batched serving is checked against draws as training does."""

import pytest
import torch

from tests.model_executor.models.duplexio.reference_sampling import ReferenceSampling, sample_factorized_text_ids


@pytest.mark.parametrize("mode", ["argmax", "top_k", "top_p"])
def test_factorized_tokens_match_training(mode: str) -> None:
    training = pytest.importorskip("duplexio.models.duplexio")
    torch.manual_seed(12)
    logits = torch.randn(7, 31)
    emit_logits = torch.randn(7)
    temperature, top_p = 0.0 if mode == "argmax" else 0.8, 0.9 if mode == "top_p" else None
    options = training.TokenSamplingOptions(temperature=temperature, top_k=12, top_p=top_p, emit_temperature=1.0)
    torch.manual_seed(37)
    expected = training._sample_factorized_token_ids(logits, emit_logits, 0, options)
    actual = sample_factorized_text_ids(
        logits,
        emit_logits,
        silence_token_id=0,
        sampling=ReferenceSampling(temperature, 12, top_p, torch.tensor([0], dtype=torch.long)),
        emit_temperature=1.0,
        generator=torch.Generator().manual_seed(37),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
