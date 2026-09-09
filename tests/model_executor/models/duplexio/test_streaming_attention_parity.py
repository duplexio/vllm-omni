"""Paged requests retain training's key order through tools and audio eviction."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.kv_reclamation import DuplexIOKVLayout
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOAttentionMetadata,
    DuplexIOFlexAttentionImpl,
    _encode_uint32,
)

pytest.importorskip("duplexio")
from duplexio.models.duplexio import history_block_mask
from duplexio.modules.stream_attention import aligned_kv_indices, history_and_self_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("sequences", [2, 8])
@torch.inference_mode()
def test_independent_requests_tools_and_audio_eviction(use_graph: bool, sequences: int) -> None:
    torch.manual_seed(89)
    lengths = [83 + 4 * i for i in range(sequences)]
    prefixes = [1 + 3 * i for i in range(sequences)]
    dim, heads, kv_heads, window = 128, 4, 2, 3
    lengths_cells = [frames * 6 for frames in lengths]
    sequence = torch.repeat_interleave(torch.arange(sequences, device="cuda"), torch.tensor(lengths_cells, device="cuda"))
    positions = torch.cat([torch.arange(size, device="cuda") for size in lengths_cells])
    frames, cells = positions // 6, positions % 6
    prefix = torch.tensor(prefixes, device="cuda")[sequence]
    audio_positions = (frames - prefix + 1).clamp_min(0)
    active = ((frames < prefix) & (cells == 0)) | ((frames >= prefix) & (cells >= 4))
    active |= (frames >= prefix) & (frames % 11 == 0) & (cells == 2)
    active |= (frames >= 12) & (frames < 18) & (cells == 3)
    # Tool-only frames preserve the audio clock, just as real tool expansion does.
    audio_frames = (frames >= prefix) & ~((frames >= 12) & (frames < 18))
    audio_positions = torch.cat(
        [chunk.view(-1, 6)[:, 0].long().cumsum(0).repeat_interleave(6) for chunk in audio_frames.split(lengths_cells)]
    )
    active &= (cells < 4) | audio_frames
    ordinals = torch.cat([chunk.long().cumsum(0) for chunk in (active & (cells < 4)).split(lengths_cells)])
    ordinals = torch.where(active & (cells < 4), ordinals, 0)
    epochs = sequence + 7
    tokens = positions.numel()
    query = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    indices, valid = aligned_kv_indices(sequence * 2 + (cells >= 4).long(), active, sequences * 2, 128)
    mask = history_block_mask(
        sequence,
        frames,
        audio_positions,
        sequence[indices],
        frames[indices],
        audio_positions[indices],
        cells[indices],
        valid,
        window,
        (128, 128),
    )
    expected = history_and_self_attention(
        query.transpose(0, 1)[None],
        key[indices].transpose(0, 1)[None],
        value[indices].transpose(0, 1)[None],
        key.transpose(0, 1)[None],
        value.transpose(0, 1)[None],
        torch.zeros(indices.numel(), device="cuda"),
        block_mask=mask,
        scale=dim**-0.5,
    )[0].transpose(0, 1)
    qmeta = torch.cat((_encode_uint32(epochs, query.dtype), query.new_zeros(tokens, 20)), -1)
    kmeta = torch.cat(
        (
            key.new_zeros(tokens, 4),
            *[_encode_uint32(x, key.dtype) for x in (epochs, positions, ordinals, active.long(), audio_positions)],
        ),
        -1,
    )
    query = torch.cat((query, qmeta[:, None].expand(-1, heads, -1)), -1)
    key = torch.cat((key, kmeta[:, None].expand(-1, kv_heads, -1)), -1)
    value = torch.cat((value, value.new_zeros(tokens, kv_heads, 24)), -1)
    layout = DuplexIOKVLayout(block_size=16, audio_window_frames=window, max_model_len=max(lengths) * 6)
    pages = layout.max_blocks * sequences
    block_table = (torch.randperm(pages, device="cuda", dtype=torch.int32) + 1).view(sequences, -1)
    cache = query.new_zeros(pages + 1, 16, kv_heads, (dim + 24) * 2).transpose(1, 2)
    cache[0].fill_(torch.nan)
    backend = DuplexIOFlexAttentionImpl(heads, dim + 24, dim**-0.5, kv_heads, None, None, "auto")
    scales = SimpleNamespace(_k_scale=torch.ones((), device="cuda"), _v_scale=torch.ones((), device="cuda"))
    offsets = [0] * sequences
    starts = list(accumulate(lengths_cells, initial=0))
    graphs = {}
    while offsets != list(lengths_cells):
        requests = [i for i in range(sequences) if offsets[i] < lengths_cells[i]]
        sizes = [prefixes[i] * 6 if offsets[i] == 0 else 6 for i in requests]
        rows = torch.cat(
            [
                torch.arange(offsets[i], offsets[i] + size, device="cuda") + starts[i]
                for i, size in zip(requests, sizes, strict=True)
            ]
        )
        boundaries = torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        metadata = DuplexIOAttentionMetadata(
            num_actual_tokens=sum(sizes),
            num_query_batches=len(requests) if all(size == 6 for size in sizes) else 1,
            query_start_loc=boundaries,
            doc_ids=torch.repeat_interleave(
                torch.arange(len(requests), device="cuda"), torch.tensor(sizes, device="cuda")
            ),
            block_table=block_table[requests],
            block_size=16,
            duplexio_layout=layout,
            duplexio_full_graph=False,
            duplexio_packed_capacity=(layout.max_compact_slots + 254) * len(requests),
        )
        if use_graph and all(size == 6 for size in sizes):
            if len(requests) not in graphs:
                q, k, v = query[rows], key[rows], value[rows]
                output = torch.empty_like(q)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        backend.forward(scales, q, k, v, cache, metadata, output)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    backend.forward(scales, q, k, v, cache, metadata, output)
                graphs[len(requests)] = graph, q, k, v, output, metadata
            graph, q, k, v, output, _ = graphs[len(requests)]
            q.copy_(query[rows])
            k.copy_(key[rows])
            v.copy_(value[rows])
            graph.replay()
            result = output[..., :dim]
        else:
            result = backend.forward(
                scales, query[rows], key[rows], value[rows], cache, metadata, torch.empty_like(query[rows])
            )[..., :dim]
        torch.testing.assert_close(result, expected[rows], rtol=0, atol=0)
        for i, size in zip(requests, sizes, strict=True):
            offsets[i] += size
