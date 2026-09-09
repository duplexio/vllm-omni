"""Packed streaming convolution must match training, not round before SiLU."""

from itertools import accumulate

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIOQwenGatedDeltaNetAttention
from vllm_omni.model_executor.models.duplexio.row_semantics import expand_stream_conv_weight
from vllm_omni.model_executor.models.duplexio.stream_conv import update_stream_conv_state_kernel

training = pytest.importorskip("duplexio.modules.qwen3_5_stream_delta")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires packed CUDA convolution")


@pytest.mark.parametrize("time_major", [False, True])
@torch.inference_mode()
def test_batched_in_place_history_shift(time_major: bool) -> None:
    """Warp scheduling must not overwrite another warp's old history reads."""
    torch.manual_seed(53)
    requests, channels, history = 8, 8192, 18
    state = torch.randn(requests, channels, history, device="cuda", dtype=torch.bfloat16)
    if time_major:
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    slots = torch.arange(requests, device="cuda", dtype=torch.int32)
    boundaries = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * 6
    active = torch.ones(requests, device="cuda", dtype=torch.bool)
    x = torch.randn(requests * 6, channels, device="cuda", dtype=torch.bfloat16)
    for _ in range(100):
        expected = torch.cat((state[:, :, 6:].clone(), x.view(requests, 6, channels).transpose(1, 2)), -1)
        update_stream_conv_state_kernel[(requests, channels // 32)](
            x, state, slots, boundaries, active, channels, history, *x.stride(), *state.stride(), 32, 32,
        )
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
        x.add_(0.01)


@pytest.mark.parametrize("lengths", [[6, 6], [6, 78], [90, 6]])
@pytest.mark.parametrize("capture_graph", [False, True])
@torch.inference_mode()
def test_streaming_matches_packed_training_and_resets_reused_slots(lengths: list[int], capture_graph: bool) -> None:
    torch.manual_seed(613)
    channels, history_length = 32, 18
    source = nn.Conv1d(channels, channels, 4, groups=channels, bias=False, device="cuda", dtype=torch.bfloat16)
    reference = training.BlockCausalConv1d(source, num_channels=6)
    native = DuplexIOQwenGatedDeltaNetAttention.__new__(DuplexIOQwenGatedDeltaNetAttention)
    nn.Module.__init__(native)
    native.full_cudagraph_enabled = False
    native.activation = "silu"
    native.conv1d = nn.Conv1d(channels, channels, 19, groups=channels, bias=False,
                            device="cuda", dtype=torch.bfloat16)
    native.conv1d.weight.copy_(expand_stream_conv_weight(source.weight, num_cells=6))
    state = torch.randn(3, channels, history_length, device="cuda", dtype=torch.bfloat16)
    original_state = state.clone()
    slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    has_state = torch.tensor([True, False], device="cuda")
    x = torch.randn(sum(lengths), channels, device="cuda", dtype=torch.bfloat16)
    boundaries = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
    histories = [state[2].T.clone(), torch.zeros_like(state[0].T)]
    segments = [torch.cat((history, part)) for history, part in zip(histories, x.split(lengths), strict=True)]
    reference_lengths = [part.shape[0] for part in segments]
    reference_boundaries = torch.tensor([0, *accumulate(reference_lengths)], device="cuda", dtype=torch.int32)
    sequence_ids = torch.repeat_interleave(torch.arange(2, device="cuda", dtype=torch.int32),
                                          torch.tensor(reference_lengths, device="cuda"))
    expected = reference(torch.cat(segments).T[None], cu_seqlens=reference_boundaries,
                         sequence_ids=sequence_ids, activation="silu")[0].T
    expected = torch.cat([part[history_length:] for part in expected.split(reference_lengths)])
    chunks = torch.tensor([[request, chunk] for request, length in enumerate(lengths)
                           for chunk in range((length + 63) // 64)], device="cuda", dtype=torch.int32)
    actual = native.apply_stream_causal_conv(x, state, slots, boundaries, has_state, chunks)
    if capture_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            native.apply_stream_causal_conv(x, state, slots, boundaries, has_state, chunks)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = native.apply_stream_causal_conv(x, state, slots, boundaries, has_state, chunks)
        state.copy_(original_state)
        graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[2], segments[0][-history_length:].T, rtol=0, atol=0)
    torch.testing.assert_close(state[0], segments[1][-history_length:].T, rtol=0, atol=0)
    torch.testing.assert_close(state[1], original_state[1], rtol=0, atol=0)
