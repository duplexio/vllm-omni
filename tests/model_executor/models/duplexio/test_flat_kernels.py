# SPDX-License-Identifier: Apache-2.0
"""The flat attention and GatedDeltaNet kernels against direct PyTorch references."""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_POSITIONS, NUM_GDN_STREAMS
from vllm_omni.model_executor.models.duplexio.stream_attention import ATTENTION_COLUMNS, flat_paged_attention
from vllm_omni.model_executor.models.duplexio.stream_gdn import (
    flat_gdn,
    gdn_gates,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def physical_slot(table: torch.Tensor, logical: int, block_size: int) -> int:
    return int(table[logical // block_size]) * block_size + logical % block_size


@pytest.mark.parametrize("frames", [1, 2, 7])
def test_flat_attention_matches_masked_softmax(frames: int) -> None:
    torch.manual_seed(frames)
    device = torch.device("cuda")
    heads, kv_heads, dim, block_size = 8, 2, 128, 16
    ring_slots, bos_base, persistent_base = 40, 48, 96
    blocks_per_request, requests = 12, 2
    num_blocks = blocks_per_request * requests + 3
    cache = torch.randn(num_blocks, block_size, 2, kv_heads, dim, device=device, dtype=torch.bfloat16)
    key_cache, value_cache = cache[:, :, 0], cache[:, :, 1]
    tables = torch.randperm(num_blocks, device=device)[: requests * blocks_per_request].view(requests, -1).int()
    tokens = frames * FRAME_POSITIONS
    query = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(tokens, kv_heads, dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(tokens, kv_heads, dim, device=device, dtype=torch.bfloat16)
    meta = torch.zeros(tokens, ATTENTION_COLUMNS, dtype=torch.int32)
    for frame in range(frames):
        request = frame % requests
        audio_end, bos_end, persistent_end = (int(torch.randint(lo, hi, ())) for lo, hi in ((0, 60), (0, 50), (0, 90)))
        for position in range(FRAME_POSITIONS):
            token = frame * FRAME_POSITIONS + position
            # Ranges grow within a frame, as in-frame writes do.
            audio_end += int(torch.randint(0, 2, ()))
            persistent_end += int(torch.randint(0, 2, ()))
            meta[token] = torch.tensor(
                [
                    request,
                    max(0, audio_end - int(torch.randint(0, 38, ()))),
                    audio_end,
                    max(0, bos_end - int(torch.randint(0, 38, ()))),
                    bos_end,
                    0,
                    persistent_end,
                    int(torch.rand(()) < 0.4),
                ]
            )
    meta = meta.to(device)
    scale = dim**-0.5
    output = flat_paged_attention(
        query,
        key,
        value,
        key_cache,
        value_cache,
        tables,
        meta,
        scale=scale,
        ring_slots=ring_slots,
        bos_base=bos_base,
        persistent_base=persistent_base,
        block_size=block_size,
    )

    flat_keys = key_cache.flatten(0, 1).float()
    flat_values = value_cache.flatten(0, 1).float()
    for token in range(tokens):
        row = meta[token].tolist()
        table = tables[row[0]].cpu()
        slots = [physical_slot(table, logical % ring_slots, block_size) for logical in range(row[1], row[2])]
        slots += [
            physical_slot(table, bos_base + logical % ring_slots, block_size) for logical in range(row[3], row[4])
        ]
        slots += [physical_slot(table, persistent_base + logical, block_size) for logical in range(row[5], row[6])]
        keys, values = flat_keys[slots], flat_values[slots]
        if row[7]:
            keys = torch.cat((keys, key[token, None].float()))
            values = torch.cat((values, value[token, None].float()))
        grouped = query[token].float().view(kv_heads, heads // kv_heads, dim)
        scores = torch.einsum("hgd,nhd->hgn", grouped, keys) * scale
        expected = torch.einsum("hgn,nhd->hgd", scores.softmax(-1), values).reshape(heads, dim)
        if not slots and not row[7]:
            expected.zero_()
        torch.testing.assert_close(output[token].float(), expected, atol=2e-2, rtol=2e-2)


def reference_gdn(mixed, weight, g, beta, conv, state, slots, positions, requests, key_heads, key_dim, value_dim):
    """``duplexio.decode.gdn_decode``'s per-stream semantics on the serving cache layout."""
    tokens = mixed.shape[0]
    value_heads = g.shape[1]
    output = torch.zeros(tokens, value_heads, value_dim, dtype=mixed.dtype)
    conv, state = conv.clone(), state.clone()
    for start, end, row, fresh in requests.tolist():
        slot = int(slots[row])
        if fresh:
            conv[slot].zero_()
            state[slot].zero_()
        for stream in range(NUM_GDN_STREAMS):
            window = conv[slot, :, 3 * stream : 3 * stream + 3].T.clone()  # (3, channels), oldest first
            heads = state[slot, stream * value_heads : (stream + 1) * value_heads].clone()
            for position in range(start, end):
                stream_id, real = positions[position].tolist()
                if stream_id != stream:
                    continue
                inputs = torch.cat((window, mixed[position, None]))
                mixed_out = F.silu((inputs.float() * weight.float().T).sum(0)).to(mixed.dtype).float()
                q, k, v = torch.split(mixed_out, (key_heads * key_dim, key_heads * key_dim, value_dim * value_heads))
                q, k = (x.view(key_heads, key_dim) for x in (q, k))
                q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * key_dim**-0.5
                k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                repeat = value_heads // key_heads
                q, k = q.repeat_interleave(repeat, 0), k.repeat_interleave(repeat, 0)
                v = v.view(value_heads, value_dim)
                stepped = heads * g[position].float().exp()[:, None, None]
                delta = beta[position].float()[:, None] * (v - torch.einsum("hkv,hk->hv", stepped, k))
                stepped = stepped + torch.einsum("hk,hv->hkv", k, delta)
                output[position] = torch.einsum("hkv,hk->hv", stepped, q).to(mixed.dtype)
                if real:
                    heads = stepped
                    window = inputs[1:]
            conv[slot, :, 3 * stream : 3 * stream + 3] = window.T
            state[slot, stream * value_heads : (stream + 1) * value_heads] = heads
    return output, conv, state


def test_flat_gdn_steps_streams_and_leaves_queries_uncommitted() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    key_heads, value_heads, key_dim, value_dim = 2, 4, 32, 32
    channels = 2 * key_heads * key_dim + value_heads * value_dim
    slots_total = 5
    # Two requests: a prefill-like run of one stream with a query, and a decode frame.
    streams = [0] * 9 + [2, 3, 0, 4, 5] + [1, 2, 3, 4, 5]
    real = [1] * 9 + [0, 1, 1, 1, 1] + [0, 0, 1, 1, 1]
    tokens = len(streams)
    positions = torch.tensor(list(zip(streams, real, strict=True)), dtype=torch.int32)
    requests = torch.tensor([[0, 14, 1, 0], [14, tokens, 0, 1]], dtype=torch.int32)
    slots = torch.tensor([3, 1], dtype=torch.int32)
    mixed = torch.randn(tokens, channels, dtype=torch.bfloat16)
    weight = (torch.randn(channels, 4) * 0.5).bfloat16().float()
    a, b = torch.randn(tokens, value_heads), torch.randn(tokens, value_heads, dtype=torch.bfloat16)
    g, beta = gdn_gates(a, b, torch.randn(value_heads) * 0.5, torch.randn(value_heads))
    conv = torch.randn(slots_total, channels, 3 * NUM_GDN_STREAMS, dtype=torch.bfloat16)
    state = torch.randn(slots_total, NUM_GDN_STREAMS * value_heads, key_dim, value_dim) * 0.1
    expected, expected_conv, expected_state = reference_gdn(
        mixed, weight, g, beta, conv, state, slots, positions, requests, key_heads, key_dim, value_dim
    )
    conv_device, state_device = conv.to(device), state.to(device)
    output = flat_gdn(
        mixed.to(device),
        weight.to(device),
        g.to(device),
        beta.to(device),
        conv_device,
        state_device,
        slots.to(device),
        positions.to(device),
        requests.to(device),
        2,
        key_heads=key_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    torch.testing.assert_close(output.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(conv_device.cpu(), expected_conv)
    torch.testing.assert_close(state_device.cpu(), expected_state, atol=1e-4, rtol=1e-3)
