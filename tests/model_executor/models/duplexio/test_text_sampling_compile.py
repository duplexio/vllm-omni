"""Compilation may fuse filtering, but must preserve its support and probabilities."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import _sample_factorized_text_ids
from vllm_omni.model_executor.models.duplexio.text_sampling import TokenSamplingOptions, content_distribution


@pytest.mark.parametrize("mode", ["top_k", "top_p"])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
)])
@torch.inference_mode()
def test_batched_filter_matches_individual_requests(mode: str, device: str) -> None:
    inputs = torch.Generator(device=device).manual_seed(73)
    options = TokenSamplingOptions(mode, 0.8, 50, 0.95, torch.tensor([0, 1, 2], device=device))
    for batch in (1, 8, 32):
        logits = torch.randn(batch, 248320, device=device, dtype=torch.bfloat16, generator=inputs)
        ids, probabilities = content_distribution(logits, options)
        for row in range(batch):
            expected_ids, expected_probabilities = content_distribution(logits[row:row + 1], options)
            torch.testing.assert_close(ids[row:row + 1], expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(probabilities[row:row + 1], expected_probabilities, rtol=0, atol=0)
            actual_rng = torch.Generator(device=device).manual_seed(row + 101)
            expected_rng = torch.Generator(device=device).manual_seed(row + 101)
            actual = torch.multinomial(probabilities[row:row + 1], 1, generator=actual_rng)
            expected = torch.multinomial(expected_probabilities, 1, generator=expected_rng)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["top_k", "top_p"])
@torch.inference_mode()
def test_compiled_distribution_matches_eager(mode: str) -> None:
    torch.manual_seed(53)
    compiled = torch.compile(
        content_distribution, fullgraph=True, dynamic=True,
        options={"emulate_precision_casts": True, "triton.cudagraphs": True},
    )
    options = TokenSamplingOptions(mode, 0.8, 50, 0.95, torch.tensor([0, 1, 2], device="cuda"))
    for batch in (1, 8, 1):
        logits = torch.randn(batch, 248320, device="cuda", dtype=torch.bfloat16)
        logits[:, :3] = 100
        for _ in range(3):
            expected_ids, expected_probs = content_distribution(logits, options)
            ids, probs = compiled(logits, options)
            torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
            torch.testing.assert_close(probs == 0, expected_probs == 0)
            torch.testing.assert_close(probs, expected_probs, atol=2e-8, rtol=1e-6)
            assert not (ids < 3).any()
            logits[:, 3:].add_(0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_compiled_filter_preserves_seeded_draws() -> None:
    compiled = torch.compile(
        content_distribution, fullgraph=True, dynamic=True,
        options={"emulate_precision_casts": True, "triton.cudagraphs": True},
    )
    options = TokenSamplingOptions("top_p", 0.8, 50, 0.95, torch.tensor([0, 1], device="cuda"))
    eager_generator = torch.Generator(device="cuda").manual_seed(37)
    compiled_generator = torch.Generator(device="cuda").manual_seed(37)
    for _ in range(100):
        logits = torch.randn(1, 248320, device="cuda", dtype=torch.bfloat16)
        emit = torch.randn(1, device="cuda")
        expected = _sample_factorized_text_ids(
            logits, emit, silence_token_id=0, sampling=options,
            emit_temperature=1.0, generator=eager_generator,
        )
        actual = _sample_factorized_text_ids(
            logits, emit, silence_token_id=0, sampling=options,
            emit_temperature=1.0, generator=compiled_generator, distribution=compiled,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(eager_generator.get_state(), compiled_generator.get_state())
