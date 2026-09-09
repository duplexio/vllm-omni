"""The actual paged backend must preserve every query's diagonal attention."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.kv_reclamation import DuplexIOKVLayout
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOAttentionMetadata,
    DuplexIOFlexAttentionImpl,
    _encode_uint32,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import duplexio_attention_visible
from vllm_omni.model_executor.models.duplexio.stream_attention import gather_history, packed_history_indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 129, 257])
def test_history_gather_preserves_values_and_zeros_poisoned_padding(dtype: torch.dtype, rows: int) -> None:
    torch.manual_seed(17)
    storage = torch.full((512, 4, 560), torch.nan, device="cuda", dtype=dtype)
    cache = storage[..., :280]
    indices = torch.randperm(512, device="cuda")[:rows]
    valid = torch.arange(rows, device="cuda") % 3 != 0
    cache[indices[valid]] = torch.randn((valid.sum().item(), 4, 280), device="cuda", dtype=dtype)
    expected = torch.where(valid[:, None, None], cache[indices, :, :256], 0)
    torch.testing.assert_close(gather_history(cache, indices, valid, 256), expected.transpose(0, 1)[None], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_history_packing_ignores_unowned_cache_metadata() -> None:
    layout = DuplexIOKVLayout(block_size=16, audio_window_frames=2, max_model_len=120)
    blocks = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]], device="cuda", dtype=torch.int32)
    length = 18 * 16
    positions = torch.full((length,), -100000, device="cuda", dtype=torch.long)
    epochs = torch.zeros_like(positions)
    audio = torch.zeros_like(positions)
    active = torch.zeros(length, device="cuda", dtype=torch.bool)
    compact = layout.persistent_text_base + torch.arange(3, device="cuda")
    physical = blocks[0, compact // 16].long() * 16 + compact % 16
    positions[physical] = torch.tensor([0, 6, 12], device="cuda")
    epochs[physical] = 7
    active[physical] = True
    owned = (blocks.long()[:, :, None] * 16 + torch.arange(16, device="cuda")).flatten(1)
    indices, valid, owners, _, _ = packed_history_indices(
        owned,
        torch.tensor([7], device="cuda"),
        torch.tensor([0], device="cuda"),
        epochs[owned],
        positions[owned],
        audio[owned],
        active[owned],
        120,
        2,
        256,
    )
    torch.testing.assert_close(indices[valid], physical, rtol=0, atol=0)
    assert not owners[valid].count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_history_packing_preserves_request_order_epochs_and_audio_retention() -> None:
    block_size = 784
    blocks = torch.tensor([[7, 2, 11, 17], [5, 19, 23, 1], [13, 4, 29, 9]], device="cuda", dtype=torch.int32)
    slots = 32 * block_size
    positions = torch.full((slots,), 2**32 - 1, device="cuda", dtype=torch.long)
    epochs = torch.zeros_like(positions)
    audio = torch.zeros_like(positions)
    active = torch.ones(slots, device="cuda", dtype=torch.bool)
    expected_indices = []
    expected_owners = []
    for owner, first_page in enumerate((7, 5, 13)):
        physical = first_page * block_size + torch.arange(8, device="cuda")
        positions[physical] = torch.tensor([0, 7, 14, 424, 419, 388, 21, 28], device="cuda")
        epochs[physical] = 17 + owner
        epochs[physical[6]] = 0
        active[physical[7]] = False
        audio[physical] = torch.tensor([0, 0, 0, 70, 69, 64, 0, 0], device="cuda")
        expected_indices.append(physical[torch.tensor([0, 1, 2, 4, 3], device="cuda")])
        expected_owners.append(torch.full((5,), owner, device="cuda", dtype=torch.long))
    owned = (blocks.long()[:, :, None] * block_size + torch.arange(block_size, device="cuda")).flatten(1)
    indices, valid, owners, _, _ = packed_history_indices(
        owned, torch.tensor([17, 18, 19], device="cuda"),
        torch.full((3,), 150, device="cuda", dtype=torch.long),
        epochs[owned], positions[owned], audio[owned], active[owned], 24576, 64, 777,
    )
    torch.testing.assert_close(indices[valid], torch.cat(expected_indices), rtol=0, atol=0)
    torch.testing.assert_close(owners[valid], torch.cat(expected_owners), rtol=0, atol=0)
    starts = torch.tensor([0, 128, 256, 384, 512, 640], device="cuda")
    assert valid[starts].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@pytest.mark.parametrize(
    "frames,live_audio,system_prefix",
    [
        (1, False, True),
        (3, False, True),
        (1, True, True),
        (160, False, True),
        (3, True, False),
    ],
)
@pytest.mark.parametrize("dtype,dim", ((torch.float32, 16), (torch.bfloat16, 128), (torch.bfloat16, 256)))
@torch.inference_mode()
def test_attention_matches_dense_including_inactive_self(
    frames: int,
    live_audio: bool,
    system_prefix: bool,
    dtype: torch.dtype,
    dim: int,
) -> None:
    torch.manual_seed(731)
    device = torch.device("cuda")
    query_heads, kv_heads = (16, 4) if dim == 256 else (4, 2)
    tokens = frames * 6
    layout = DuplexIOKVLayout(block_size=16, audio_window_frames=2, max_model_len=(frames + 12) * 6)
    positions = torch.arange(tokens, device=device)
    active = (positions % 6 == 0) & system_prefix
    if live_audio:
        active |= positions % 6 >= 4
    audio_pos = positions // 6 + 1 if live_audio else torch.zeros_like(positions)
    epochs = torch.full_like(positions, 19)
    ordinals = torch.where((positions % 6 == 0) & system_prefix, positions // 6 + 1, 0)
    query = torch.randn(tokens, query_heads, dim, device=device, dtype=dtype)
    key = torch.randn(tokens, kv_heads, dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    q_metadata = torch.cat((_encode_uint32(epochs, query.dtype), query.new_zeros(tokens, 20)), -1)
    k_metadata = torch.cat(
        (
            key.new_zeros(tokens, 4),
            *[_encode_uint32(x, key.dtype) for x in (epochs, positions, ordinals, active.long(), audio_pos)],
        ),
        -1,
    )
    augmented_query = torch.cat((query, q_metadata[:, None].expand(-1, query_heads, -1)), -1)
    augmented_key = torch.cat((key, k_metadata[:, None].expand(-1, kv_heads, -1)), -1)
    augmented_value = torch.cat((value, value.new_zeros(tokens, kv_heads, 24)), -1)
    cache = query.new_zeros(layout.max_blocks + 1, 16, kv_heads, 2 * (dim + 24)).transpose(1, 2)
    cache[0].fill_(torch.nan)
    boundaries = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    block_ids = torch.arange(1, layout.max_blocks + 1, device=device, dtype=torch.int32)[None]
    metadata = DuplexIOAttentionMetadata(
        num_actual_tokens=tokens,
        num_query_batches=1,
        query_start_loc=boundaries,
        block_table=block_ids,
        block_size=16,
        doc_ids=torch.zeros(tokens, device=device, dtype=torch.long),
        duplexio_layout=layout,
        duplexio_full_graph=False,
        duplexio_packed_capacity=layout.max_compact_slots + 254,
    )
    backend = DuplexIOFlexAttentionImpl(query_heads, dim + 24, dim**-0.5, kv_heads, None, None, "auto")
    scales = SimpleNamespace(_k_scale=torch.ones((), device=device), _v_scale=torch.ones((), device=device))
    output = backend.forward(
        scales, augmented_query, augmented_key, augmented_value, cache, metadata, torch.empty_like(augmented_query)
    )[..., :dim]
    visible = duplexio_attention_visible(
        positions[:, None],
        positions[None],
        audio_pos[:, None],
        audio_pos[None],
        active[None],
        audio_attention_window_frames=2,
    )
    if dtype == torch.bfloat16:
        pytest.importorskip("duplexio")
        from duplexio.models.duplexio import history_block_mask
        from duplexio.modules.stream_attention import (
            aligned_kv_indices,
            history_and_self_attention,
        )

        indices, valid = aligned_kv_indices((positions % 6 >= 4).long(), active, 2, 128)
        block = history_block_mask(
            torch.zeros_like(positions),
            positions // 6,
            audio_pos,
            torch.zeros_like(indices),
            positions[indices] // 6,
            audio_pos[indices],
            positions[indices] % 6,
            valid,
            2,
            (128, 128),
        )
        expected = (
            history_and_self_attention(
                query.transpose(0, 1).unsqueeze(0),
                key[indices].transpose(0, 1).unsqueeze(0),
                value[indices].transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                torch.zeros(indices.numel(), device=device),
                block_mask=block,
                scale=dim**-0.5,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        return
    scores = torch.einsum("thd,shd->hts", query, key.repeat_interleave(2, 1)) * dim**-0.5
    probabilities = scores.masked_fill(~visible, -torch.inf).softmax(-1)
    expected = torch.einsum("hts,shd->thd", probabilities, value.repeat_interleave(2, 1))
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
