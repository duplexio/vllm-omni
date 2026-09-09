"""Training-ordered attention reductions for the native paged cache."""

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, create_block_mask, flex_attention
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from vllm_omni.model_executor.models.duplexio.numerics import call_compiled_function
from vllm_omni.model_executor.models.duplexio.self_attention_merge import merge_self_attention

compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)


@torch.compile(dynamic=True, fullgraph=True)
def cached_rotary_pos_emb(
    query: Tensor, key: Tensor, positions: Tensor, cos_sin_cache: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply training's fused rotation to (T, H, D) Q/K using cached phases."""
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    query, key = apply_rotary_pos_emb(
        query.transpose(0, 1)[None],
        key.transpose(0, 1)[None],
        torch.cat((cos, cos), dim=-1)[None],
        torch.cat((sin, sin), dim=-1)[None],
    )
    return query[0].transpose(0, 1), key[0].transpose(0, 1)


@torch.compiler.disable
@torch.no_grad()
def history_block_mask(
    query_seq_idx: Tensor,
    query_frame_pos: Tensor,
    query_audio_pos: Tensor,
    key_seq_idx: Tensor,
    key_frame_pos: Tensor,
    key_audio_pos: Tensor,
    key_cell_ids: Tensor,
    key_active: Tensor,
    audio_window: int,
    block_size: tuple[int, int],
) -> BlockMask:
    """Match packed visibility for (B, T) queries and shared packed keys."""
    # Two metadata tensors give the compiler one layout, independent of whether
    # a caller's frame/audio clocks alias (as they do for uninterrupted audio).
    queries = torch.stack((query_seq_idx, query_frame_pos, query_audio_pos - audio_window), dim=-1)
    keys = torch.stack((key_seq_idx, key_frame_pos, key_audio_pos, key_cell_ids, key_active), dim=-1)

    def visible(_batch: Tensor, _head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        same_sequence = queries[_batch, query, 0] == keys[key, 0]
        prior_row = queries[_batch, query, 1] > keys[key, 1]
        key_is_audio = keys[key, 3] >= 4
        audio_in_window = queries[_batch, query, 2] <= keys[key, 2]
        return same_sequence & prior_row & (keys[key, 4] != 0) & (~key_is_audio | audio_in_window)

    if query_frame_pos.shape[1] == 6:
        present = call_compiled_function(frame_history_blocks, query_seq_idx, key_seq_idx, key_active)
    else:
        with torch.autocast(query_frame_pos.device.type, enabled=False):
            block = call_compiled_function(
                compiled_create_block_mask,
                visible,
                B=query_frame_pos.shape[0],
                H=None,
                Q_LEN=query_frame_pos.shape[1],
                KV_LEN=key_frame_pos.shape[0],
                device=query_frame_pos.device,
                BLOCK_SIZE=block_size,
            )
        present = block.to_dense().bool()
    return ordered_block_mask(
        present, visible, block_size, (query_frame_pos.shape[1], key_frame_pos.shape[0]),
    )


@torch.compile(dynamic=True, fullgraph=True)
def frame_history_blocks(query_owners: Tensor, key_owners: Tensor, key_active: Tensor) -> Tensor:
    """Packed 128-key groups start with a valid key and belong to one request.

    The element mask still applies frame/window visibility within these blocks.
    """
    present = (query_owners[:, :1] == key_owners[None, ::128]) & key_active[None, ::128]
    return present[:, None, None]


@torch.compile(dynamic=True, fullgraph=True)
def packed_history_indices(
    physical: Tensor,
    request_epochs: Tensor,
    current_audio: Tensor,
    epochs: Tensor,
    positions: Tensor,
    audio_positions: Tensor,
    active: Tensor,
    max_model_len: int,
    audio_window: int,
    capacity: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Read paged storage in each request's text-then-audio chronological order.

    Padding aligns groups to 128 keys; physical page placement never determines
    softmax reduction order. The caller supplies a CPU-known capacity bound.
    """
    requests = physical.shape[0]
    owners = torch.arange(requests, device=positions.device)[:, None]
    audio = positions % 6 >= 4
    first_audio = ((current_audio - audio_window - 1) // 64 * 64 + 1).clamp_min(1)
    active = active & (epochs == request_epochs[:, None]) & (
        ~audio | (audio_positions >= first_audio[:, None])
    )
    groups = owners * 2 + audio.long()
    order = (groups * (2 * max_model_len) + torch.where(active, positions, max_model_len)).flatten().argsort()
    sizes = torch.stack(((~audio).sum(1), audio.sum(1)), 1).flatten()
    counts = torch.stack(((active & ~audio).sum(1), (active & audio).sum(1)), 1).flatten()
    ends = ((counts + 127) // 128 * 128).cumsum(0)
    offsets = torch.arange(capacity, device=positions.device)
    selected_groups = torch.searchsorted(ends, offsets, right=True)
    zero = counts.new_zeros(1)
    local = offsets - torch.cat((zero, ends))[selected_groups]
    valid = local < torch.cat((counts, zero))[selected_groups]
    starts = torch.cat((sizes.cumsum(0) - sizes, zero))[selected_groups]
    indices = order[torch.where(valid, starts + local, 0)]
    return (
        physical.flatten()[indices], valid, indices // physical.shape[1],
        positions.flatten()[indices], audio_positions.flatten()[indices],
    )


def ordered_block_mask(
    present: Tensor,
    mask_mod: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    block_size: tuple[int, int],
    seq_lengths: tuple[int, int],
) -> BlockMask:
    """Visit visible blocks in key order, regardless of full/partial classification."""
    columns = torch.arange(present.shape[-1], device=present.device, dtype=torch.int32)
    indices = ((~present) * present.shape[-1] + columns).argsort(-1).int()
    ordered = BlockMask.from_kv_blocks(
        present.sum(-1, dtype=torch.int32),
        indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=seq_lengths,
    )
    assert ordered.q_num_blocks is not None and ordered.q_indices is not None
    # Dynamo otherwise specializes these nested tensor attributes by size.
    torch._dynamo.mark_dynamic(ordered.kv_num_blocks, 2)
    torch._dynamo.mark_dynamic(ordered.kv_indices, (2, 3))
    torch._dynamo.mark_dynamic(ordered.q_num_blocks, 2)
    torch._dynamo.mark_dynamic(ordered.q_indices, (2, 3))
    return ordered


@triton.jit
def gather_history_kernel(
    cache, indices, valid, output,
    num_rows: tl.constexpr, num_heads: tl.constexpr, dim: tl.constexpr,
    row_stride: tl.constexpr, head_stride: tl.constexpr, dim_stride: tl.constexpr,
    block_dim: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    head = tl.program_id(1)
    dims = tl.arange(0, block_dim)
    active = tl.load(valid + rows, rows < num_rows, other=False)
    slots = tl.load(indices + rows, rows < num_rows, other=0)
    values = tl.load(
        cache + slots[:, None] * row_stride + head * head_stride + dims[None, :] * dim_stride,
        active[:, None] & (dims[None, :] < dim), other=0,
    )
    tl.store(
        output + (rows[:, None] * num_heads + head) * dim + dims[None, :], values,
        (rows[:, None] < num_rows) & (dims[None, :] < dim),
    )


def gather_history(cache: Tensor, indices: Tensor, valid: Tensor, dim: int) -> Tensor:
    """Padding is zero, never a read of undefined cache storage into attention."""
    rows, heads = indices.shape[0], cache.shape[1]
    output = torch.empty((rows, heads, dim), dtype=cache.dtype, device=cache.device)
    gather_history_kernel[(triton.cdiv(rows, 16), heads)](
        cache, indices, valid, output,
        rows, heads, dim, *cache.stride(), triton.next_power_of_2(dim),
    )
    return output.transpose(0, 1)[None]


@torch.compile(dynamic=True, fullgraph=True)
def gated_attention_output(output: Tensor, gate: Tensor) -> Tensor:
    """Use the same fused sigmoid/product rounding in training and decoding."""
    return output * gate.sigmoid()


@torch.compile(dynamic=True, fullgraph=True)
def fixed_history_attention(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    block_mask: BlockMask,
    scale: float,
    key_log_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Own the compiled entrypoint; an unfused fallback changes its arithmetic."""

    def score_mod(
        score: Tensor,
        _batch: Tensor,
        _head: Tensor,
        _query_idx: Tensor,
        key_idx: Tensor,
    ) -> Tensor:
        return score + key_log_weights[key_idx]

    history, auxiliary = flex_attention(
        query,
        keys,
        values,
        block_mask=block_mask,
        scale=scale,
        score_mod=score_mod,
        enable_gqa=query.shape[1] != keys.shape[1],
        return_aux=AuxRequest(lse=True),
        kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1, "FORCE_USE_FLEX_ATTENTION": True},
    )
    assert auxiliary.lse is not None
    return history, auxiliary.lse


def history_and_self_attention(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    self_key: Tensor,
    self_value: Tensor,
    key_log_weights: Tensor,
    *,
    block_mask: BlockMask,
    scale: float,
) -> Tensor:
    """Use one fixed reduction for history, followed by the query-local diagonal.

    The mask must exclude the diagonal; self K/V are passed separately because
    inactive cells are not retained in the history cache.
    """
    # Q/K/V already carry the projection dtype; outer AMP is not kernel policy.
    with torch.autocast(query.device.type, enabled=False):
        history, lse = call_compiled_function(
            fixed_history_attention,
            query,
            keys,
            values,
            block_mask,
            scale,
            key_log_weights,
        )
        return call_compiled_function(
            merge_self_attention,
            query,
            self_key,
            self_value,
            history,
            lse,
            scale,
        )
