"""Kernel math, cache ownership and graph replay; bitwise training parity is not required."""

import math

import pytest
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as upstream_gdn

from vllm_omni.model_executor.models.duplexio.stream_gdn import (
    append_gdn,
    gdn_cache_dtypes,
    gdn_cache_shapes,
    prepare_gdn_inputs,
)

pytest.importorskip("duplexio")

from duplexio.modules.fla_block_gated_delta_rule.chunk import (
    chunk_gated_delta_rule as training_gdn,
)
from duplexio.modules.fla_block_gated_delta_rule.chunk import (
    l2_normalize,
)
from duplexio.modules.qwen3_5_stream_delta import _qwen_beta_gate


def make_state(aliased_storage: bool, slots: int = 3) -> torch.Tensor:
    shapes = gdn_cache_shapes(1, 2, 4, 128, 128, 4)
    dtypes = gdn_cache_dtypes(torch.bfloat16)
    if not aliased_storage:
        return torch.empty(slots, *shapes[1], device="cuda", dtype=dtypes[1])
    sizes = [torch.empty((), dtype=dtype).element_size() for dtype in dtypes]
    page_bytes = sum(math.prod(shape) * size for shape, size in zip(shapes, sizes, strict=True))
    raw = torch.empty(slots * page_bytes, device="cuda", dtype=torch.uint8)
    views = []
    offset = 0
    for shape, dtype, size in zip(shapes, dtypes, sizes, strict=True):
        views.append(
            torch.as_strided(
                raw.view(dtype),
                (slots, *shape),
                (page_bytes // size, *torch.empty(shape, device="meta").stride()),
                storage_offset=offset // size,
            )
        )
        offset += math.prod(shape) * size
    return views[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", (6, 64, 138))
@torch.inference_mode()
def test_native_gdn_matches_training(length: int) -> None:
    torch.manual_seed(27)
    q = l2_normalize(torch.randn(1, length, 2, 128, device="cuda", dtype=torch.bfloat16))
    k = l2_normalize(torch.randn_like(q))
    v = torch.randn(1, length, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, length, 4, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, length, 4, device="cuda", dtype=torch.bfloat16)
    expected, _ = training_gdn(q, k, v, g, beta)
    actual, _ = upstream_gdn(q, k, v, g, beta)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", (6, 138))
@torch.inference_mode()
def test_native_gdn_preparation_matches_training(length: int) -> None:
    torch.manual_seed(61)
    qkv = torch.randn(length, 1024, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(length, 4, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(4, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn_like(a_log)
    q, k, v = qkv.split((256, 256, 512), -1)
    beta, g = _qwen_beta_gate(b, a, a_log, dt_bias)
    expected = (
        l2_normalize(q.view(length, 2, 128)),
        l2_normalize(k.view(length, 2, 128)),
        v.view(length, 4, 128),
        g,
        beta,
    )
    q, k, v, g, beta = prepare_gdn_inputs(qkv, a, b, a_log, dt_bias, 2, 128, 128)
    for name, actual, reference in zip(("q", "k", "v", "g", "beta"), (q, k, v, g, beta), expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefixes", ((6, 12), (60, 66), (132, 138), (6, 12, 18, 24, 60, 66, 132, 138)))
@pytest.mark.parametrize("aliased_storage", (False, True))
@torch.inference_mode()
def test_append_gdn_independent_slots_and_reset(prefixes: tuple[int, ...], aliased_storage: bool) -> None:
    torch.manual_seed(39)
    length = 168
    sequences = len(prefixes)
    q = l2_normalize(torch.randn(sequences, length, 2, 128, device="cuda", dtype=torch.bfloat16))
    k = l2_normalize(torch.randn_like(q))
    v = torch.randn(sequences, length, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(sequences, length, 4, device="cuda", dtype=torch.float32)
    beta = torch.rand(sequences, length, 4, device="cuda", dtype=torch.bfloat16)
    inputs = (q, k, v, g, beta)
    expected = dense_recurrence(*inputs)
    state = make_state(aliased_storage, sequences + 1)
    slot_order = (sequences, *range(sequences - 1))
    first_pass = []
    for pass_index, slots in enumerate((slot_order, slot_order[::-1])):
        offsets = [0] * sequences
        append_index = 0
        while min(offsets) < length:
            requests = [i for i in range(sequences) if offsets[i] < length]
            sizes = [min(prefixes[i] if offsets[i] == 0 else 6, length - offsets[i]) for i in requests]
            packed = tuple(
                torch.cat([tensor[i, offsets[i] : offsets[i] + size] for i, size in zip(requests, sizes)])
                for tensor in inputs
            )
            boundaries = torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
            chunks = torch.tensor(
                [(i, chunk) for i, size in enumerate(sizes) for chunk in range((size + 63) // 64)],
                device="cuda",
                dtype=torch.int32,
            )
            actual = append_gdn(
                *packed,
                state,
                torch.tensor([slots[i] for i in requests], device="cuda", dtype=torch.int32),
                boundaries,
                torch.tensor([offsets[i] > 0 for i in requests], device="cuda"),
                chunks,
            )
            wanted = torch.cat([expected[i, offsets[i] : offsets[i] + size] for i, size in zip(requests, sizes)])
            assert_recurrence_close(actual, wanted)
            if pass_index == 0:
                first_pass.append(actual.clone())
            else:
                torch.testing.assert_close(actual, first_pass[append_index], atol=0, rtol=0)
            append_index += 1
            for i, size in zip(requests, sizes):
                offsets[i] += size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@torch.inference_mode()
def test_append_gdn_graph_replay_across_blocks_and_slot_reset() -> None:
    torch.manual_seed(49)
    q = l2_normalize(torch.randn(1, 144, 2, 128, device="cuda", dtype=torch.bfloat16))
    k = l2_normalize(torch.randn_like(q))
    v = torch.randn(1, 144, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, 144, 4, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, 144, 4, device="cuda", dtype=torch.bfloat16)
    inputs = (q, k, v, g, beta)
    expected = dense_recurrence(*inputs)
    state = make_state(True)
    eager_state = make_state(True)
    buffers = tuple(tensor[0, :6].clone() for tensor in inputs)
    slots = torch.tensor([2], device="cuda", dtype=torch.int32)
    boundaries = torch.tensor([0, 6], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([[0, 0]], device="cuda", dtype=torch.int32)
    has_state = torch.tensor([False], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            append_gdn(*buffers, state, slots, boundaries, has_state, chunks)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = append_gdn(*buffers, state, slots, boundaries, has_state, chunks)
    for slot in (2, 0, 2):
        slots.fill_(slot)
        for offset in range(0, 144, 6):
            for buffer, tensor in zip(buffers, inputs, strict=True):
                buffer.copy_(tensor[0, offset : offset + 6])
            has_state.fill_(offset > 0)
            graph.replay()
            eager = append_gdn(*buffers, eager_state, slots, boundaries, has_state, chunks)
            torch.testing.assert_close(output, eager, atol=0, rtol=0)
            assert_recurrence_close(output, expected[0, offset : offset + 6])


def assert_recurrence_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = actual.float() - expected.float()
    assert torch.isfinite(actual).all()
    assert difference.norm() / expected.float().norm() < 0.01
    assert difference.abs().max() / expected.float().abs().max() < 0.01


@torch.inference_mode()
def dense_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """Literal FP64 state update, independent of either CUDA implementation."""
    batches, length, value_heads, value_dim = v.shape
    state = torch.zeros(batches, value_heads, k.shape[-1], value_dim, device=q.device, dtype=torch.float64)
    groups = value_heads // k.shape[-2]
    outputs = []
    for index in range(length):
        key = k[:, index].double().repeat_interleave(groups, 1)
        query = q[:, index].double().repeat_interleave(groups, 1) * q.shape[-1] ** -0.5
        state *= g[:, index, :, None, None].double().exp()
        residual = v[:, index].double() - (state * key[..., None]).sum(-2)
        state += key[..., None] * (beta[:, index, :, None].double() * residual)[..., None, :]
        outputs.append((state * query[..., None]).sum(-2))
    return torch.stack(outputs, 1)
