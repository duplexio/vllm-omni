"""Long-lived codec state must retain only its declared attention window."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.pocket_mimi import (
    PocketMimi,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA codec")
@torch.inference_mode()
@torch.backends.cudnn.flags(allow_tf32=False)
def test_combined_prefix_preserves_continuous_codec_state() -> None:
    torch.manual_seed(44)
    codec = PocketMimi().cuda().eval()
    speaker = torch.randn(1, 1, 3 * 1920, device="cuda")
    silence = torch.zeros(1, 1, 11 * 1920, device="cuda")
    bulk, bulk_state = codec.encode(torch.cat((speaker, silence), dim=-1), codec.new_state(1))
    serial_state = codec.new_state(1)
    outputs = []
    for chunk in (speaker, *silence.split(1920, dim=-1)):
        output, serial_state = codec.encode(chunk, serial_state)
        outputs.append(output)
    torch.testing.assert_close(bulk, torch.cat(outputs, dim=-1), atol=1e-4, rtol=1e-3)
    for chunk in (speaker[..., :1920], silence[..., :1920]):
        actual, bulk_state = codec.encode(chunk, bulk_state)
        expected, serial_state = codec.encode(chunk, serial_state)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)


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
