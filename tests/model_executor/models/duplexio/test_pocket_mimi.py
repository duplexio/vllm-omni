"""Long-lived codec state must retain only its declared attention window."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.pocket_mimi import (
    StreamingMultiheadAttention,
)


@torch.inference_mode()
def test_attention_cache_is_bounded_by_its_window() -> None:
    attention = StreamingMultiheadAttention(32, 4, context=7).eval()
    state = attention.get_initial_state(2, torch.device("cpu"), torch.float32)
    for _ in range(10):
        _, state = attention.step(torch.randn(2, 3, 32), state)
        assert state.k.shape[1] <= 6
        assert state.v.shape == state.k.shape
    assert state.seq_len == 30


@torch.inference_mode()
def test_bounded_attention_matches_full_training_history() -> None:
    training = pytest.importorskip("duplexio.modules.continuous_mimi")
    torch.manual_seed(12)
    reference = training.StreamingMultiheadAttention(32, 4, context=7).eval()
    native = StreamingMultiheadAttention(32, 4, context=7).eval()
    native.load_state_dict(reference.state_dict())
    initial = (2, torch.device("cpu"), torch.float32)
    reference_state = reference.get_initial_state(*initial)
    native_state = native.get_initial_state(*initial)
    for length in (3, 5, 2, 4, 9, 1):
        hidden = torch.randn(2, length, 32)
        expected, reference_state = reference.step(hidden, reference_state)
        actual, native_state = native.step(hidden, native_state)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
