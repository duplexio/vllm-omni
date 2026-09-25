# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The paged backend must reproduce dense DuplexIO attention over live sessions."""

from collections.abc import Sequence
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import AuxRequest, flex_attention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata, FlexAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOFrameMetadata,
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    KV_TILE,
    QUERY_BLOCK_ROWS,
    DuplexIOFlexAttentionImpl,
    DuplexIOFlexAttentionMetadataBuilder,
    DuplexIOPagedAttention,
    kv_splits,
    physical_slots,
    step_kv_tiles,
    step_page_bounds,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    duplexio_attention_visible,
)
from vllm_omni.model_executor.models.duplexio.stream_attention import merge_self_attention

WINDOW = 2
# FlexAttention tiles a page with BLOCK_N, so a page cannot be smaller than it.
BLOCK_SIZE = 64
NUM_AUDIO_CELLS = DUPLEXIO_NUM_CELLS - DUPLEXIO_NUM_TEXT_CELLS
FRAME_FIELDS = (
    "key_active",
    "persistent_ordinal",
    "persistent_last",
    "audio_first",
    "audio_last",
)
# Paged attention reduces the history in page order and merges the diagonal
# afterwards, so it does not round like a dense matmul. Bit-exactness with
# training was traded for that simpler reduction.
TOLERANCE = {torch.float32: 2e-5, torch.bfloat16: 3e-2}
# Tables the paged mask closure reads at kernel launch.
UPSTREAM_TABLES = ("doc_ids", "physical_to_logical", "decode_offset", "num_blocks_per_seq", "seq_lens", "block_table")


def cells(values: Tensor) -> Tensor:
    """Give every cell of a row the row's value."""
    return values[:, None].expand(-1, DUPLEXIO_NUM_CELLS).flatten()


def session_fields(audio_active: Tensor, text_active: Tensor, prompt_frames: int = 0) -> dict[str, Tensor]:
    """Per-token frame fields for a whole session, as ``frame_inputs`` derives them.

    ``audio_active`` is (rows,) and ``text_active`` (rows, 4). A row without
    active audio is a token burst - a spliced tool result - which keeps the audio
    clock frozen and therefore consumes no window budget.
    """
    rows = audio_active.shape[0]
    pinned = torch.arange(rows) < prompt_frames
    text_active = text_active & ~pinned[:, None]
    audio_position = audio_active.cumsum(0, dtype=torch.int32)
    persistent = torch.cat(
        (text_active, torch.zeros(rows, 1, dtype=torch.bool), pinned[:, None]), 1
    )
    ordinal = persistent.int().flatten().cumsum(0, dtype=torch.int32).view(rows, -1)
    return {
        "key_active": torch.cat(
            (text_active, torch.stack((audio_active, audio_active | pinned), -1)), 1
        ).flatten(),
        "persistent_ordinal": torch.where(persistent, ordinal, 0).flatten(),
        "persistent_last": cells(
            torch.cat((torch.zeros(1, dtype=torch.int32), ordinal[:-1, -1]))
        ),
        "audio_first": cells((audio_position - WINDOW).clamp_min(1)),
        "audio_last": cells(audio_position - audio_active.int()),
        "pinned": cells(pinned),
        # Audio time, for the dense reference: not a cache-addressing field.
        "audio_position": cells(audio_position),
    }


def batched(source: Sequence[Tensor], live: list[int], used: list[Tensor]) -> Tensor:
    """Concatenate each live session's slice of a per-session tensor."""
    return torch.cat(
        [source[request][rows] for request, rows in zip(live, used, strict=True)]
    )


def dense_attention(
    query: Tensor, key: Tensor, value: Tensor, fields: dict[str, Tensor], scale: float
) -> Tensor:
    """Reduce the exact visibility relation in fp32 over every session token."""
    positions = torch.arange(query.shape[0], device=query.device)
    visible = duplexio_attention_visible(
        positions[:, None],
        positions[None],
        fields["audio_position"][:, None],
        fields["audio_position"][None],
        fields["key_active"][None],
        fields["pinned"][None],
        audio_attention_window_frames=WINDOW,
    )
    groups = query.shape[1] // key.shape[1]
    keys = key.float().repeat_interleave(groups, 1)
    values = value.float().repeat_interleave(groups, 1)
    scores = torch.einsum("thd,shd->hts", query.float(), keys) * scale
    return torch.einsum(
        "hts,shd->thd", scores.masked_fill(~visible, -torch.inf).softmax(-1), values
    )


def training_attention(query: Tensor, key: Tensor, value: Tensor, fields: dict[str, Tensor], scale: float) -> Tensor:
    training = pytest.importorskip("duplexio.models.duplexio")
    from duplexio.modules.stream_attention import unified_attention

    positions = torch.arange(query.shape[0], device=query.device)
    sequence = torch.zeros_like(positions)
    frame = positions // DUPLEXIO_NUM_CELLS
    cell = positions % DUPLEXIO_NUM_CELLS
    audio = fields["audio_position"]
    mask, indices = training.attention_mask(
        sequence, frame, audio, cell, sequence, frame, audio, cell,
        fields["key_active"], fields["pinned"], WINDOW, query.shape[-1],
    )
    return unified_attention(
        query.transpose(0, 1)[None], key[indices].transpose(0, 1)[None],
        value[indices].transpose(0, 1)[None], mask, scale,
    )[0].transpose(0, 1).float()


def paged_backend(
    spec: DuplexIOKVCacheSpec,
    frame: DuplexIOFrameMetadata,
    heads: int,
    max_seqs: int,
    pages: int,
) -> tuple[nn.Module, DuplexIOFlexAttentionMetadataBuilder, DuplexIOFlexAttentionImpl]:
    """Wire the real builder and impl to one attention layer's frame metadata."""
    device = frame.cell.device
    layer = object.__new__(DuplexIOPagedAttention)
    nn.Module.__init__(layer)
    layer._k_scale = torch.ones((), device=device)
    layer._v_scale = torch.ones((), device=device)
    layer.frame = frame
    layer.logical_mask_mod = frame.visible
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: heads,
            get_num_kv_heads=lambda _: spec.num_kv_heads,
            get_head_size=lambda: spec.head_size,
            max_model_len=spec.max_model_len,
            rswa_window=None,
        ),
        parallel_config=None,
        cache_config=SimpleNamespace(num_gpu_blocks=pages),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_seqs, max_num_batched_tokens=frame.cell.shape[0]
        ),
        attention_config=SimpleNamespace(
            flex_attn_q_block_size=None,
            flex_attn_kv_block_size=None,
            flex_attn_block_m=None,
            flex_attn_block_n=None,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True),
            static_forward_context={"attn": layer},
        ),
    )
    builder = DuplexIOFlexAttentionMetadataBuilder(spec, ["attn"], vllm_config, device)
    impl = DuplexIOFlexAttentionImpl(
        heads,
        spec.head_size,
        spec.head_size**-0.5,
        spec.num_kv_heads,
        None,
        None,
        "auto",
    )
    return layer, builder, impl


def step_common(sizes: list[int], seq_lens: list[int], block_table: Tensor) -> CommonAttentionMetadata:
    device = block_table.device
    starts = torch.tensor([0, *accumulate(sizes)], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=len(sizes),
        num_actual_tokens=sum(sizes),
        max_query_len=max(sizes),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(sum(sizes), dtype=torch.long, device=device),
    )


def step_metadata(
    builder: DuplexIOFlexAttentionMetadataBuilder,
    sizes: list[int],
    seq_lens: list[int],
    block_table: Tensor,
) -> FlexAttentionMetadata:
    return builder.build(0, step_common(sizes, seq_lens, block_table))


def session_activity(rows: int, prefix: int, request: int) -> tuple[Tensor, Tensor]:
    """Text-only prefix, then live audio rows with a mid-session token burst."""
    row = torch.arange(rows)
    audio_active = (row >= prefix) & ((row < 12 + request) | (row >= 15 + request))
    text_active = torch.stack(
        (
            row < prefix,
            torch.zeros(rows, dtype=torch.bool),
            audio_active & (row % 5 == request),
            ~audio_active & (row >= prefix),
        ),
        dim=1,
    )
    return audio_active, text_active


def run_session(
    prefixes: list[int],
    rows: int,
    dtype: torch.dtype,
    dim: int,
    heads: int,
    kv_heads: int,
    *,
    prompt_frames: int = 0,
    training_reference: bool = False,
) -> None:
    """Replay ``len(prefixes)`` sessions through the paged backend, step by step."""
    torch.manual_seed(731)
    device = torch.device("cuda")
    requests = len(prefixes)
    spec = make_duplexio_kv_cache_spec(
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=kv_heads,
            head_size=dim,
            head_size_v=dim,
            dtype=dtype,
        ),
        audio_window_frames=WINDOW,
        max_model_len=rows * DUPLEXIO_NUM_CELLS,
    )
    layout = spec.layout
    tokens = rows * DUPLEXIO_NUM_CELLS
    query = torch.randn(requests, tokens, heads, dim, device=device, dtype=dtype)
    key = torch.randn(requests, tokens, kv_heads, dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    fields = [
        {
            name: field.to(device)
            for name, field in session_fields(
                *session_activity(rows, prefix, request), prompt_frames,
            ).items()
        }
        for request, prefix in enumerate(prefixes)
    ]
    reference = training_attention if training_reference else dense_attention
    expected = [
        reference(query[i], key[i], value[i], fields[i], dim**-0.5)
        for i in range(requests)
    ]

    pages = layout.max_blocks * requests
    frame = DuplexIOFrameMetadata(
        layout, sum(prefixes) * DUPLEXIO_NUM_CELLS, device
    )
    layer, builder, impl = paged_backend(spec, frame, heads, requests, pages + 1)
    # Page 0 belongs to no request: a page outside the block table must never be
    # read. vLLM zeroes the pages it does hand out, which is what keeps
    # masked-out slots from poisoning the reduction.
    cache = torch.zeros(
        pages + 1, BLOCK_SIZE, kv_heads, 2 * dim, device=device, dtype=dtype
    ).transpose(1, 2)
    cache[0].fill_(torch.nan)
    block_table = (torch.randperm(pages, device=device, dtype=torch.int32) + 1).view(
        requests, layout.max_blocks
    )

    # Every session prefills its text-only prefix in the first step, then walks
    # one row per step until it runs out of rows.
    live = list(range(requests))
    consumed = [0] * requests
    steps = list(prefixes)
    while live:
        used = [
            torch.arange(
                consumed[request] * DUPLEXIO_NUM_CELLS,
                (consumed[request] + step) * DUPLEXIO_NUM_CELLS,
                device=device,
            )
            for request, step in zip(live, steps, strict=True)
        ]

        frame.update(
            **{
                name: batched([field[name] for field in fields], live, used)
                for name in FRAME_FIELDS
            }
        )
        sizes = [step * DUPLEXIO_NUM_CELLS for step in steps]
        metadata = step_metadata(
            builder,
            sizes,
            [
                (consumed[request] + step) * DUPLEXIO_NUM_CELLS
                for request, step in zip(live, steps, strict=True)
            ],
            block_table[live],
        )
        batch = batched(query, live, used)
        output = impl.forward(
            layer,
            batch,
            batched(key, live, used),
            batched(value, live, used),
            cache,
            metadata,
            torch.empty_like(batch),
        )
        for request, rows_used, part in zip(
            live, used, output.split(sizes), strict=True
        ):
            torch.testing.assert_close(
                part.float(),
                expected[request][rows_used],
                atol=TOLERANCE[dtype],
                rtol=TOLERANCE[dtype],
            )

        for request, step in zip(live, steps, strict=True):
            consumed[request] += step
        live = [request for request in live if consumed[request] < rows]
        steps = [1] * len(live)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@pytest.mark.parametrize(
    "dtype,dim,heads,kv_heads,rows",
    (
        (torch.float32, 16, 4, 2, 24),
        # The ring holds `window + 64` frames, so only a long session wraps it
        # and forces the mask to reject a slot the window has passed.
        (torch.bfloat16, 256, 16, 4, 72),
    ),
)
@torch.inference_mode()
def test_paged_session_matches_dense_attention(
    dtype: torch.dtype, dim: int, heads: int, kv_heads: int, rows: int
) -> None:
    run_session([3], rows, dtype, dim, heads, kv_heads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@pytest.mark.parametrize("requests", [3, 32, 64])
@torch.inference_mode()
def test_batched_sessions_stay_isolated_in_a_shuffled_block_table(requests: int) -> None:
    run_session([3 + request % 3 for request in range(requests)], 20, torch.bfloat16, 128, 8, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@pytest.mark.parametrize("training_reference", [False, True])
@torch.inference_mode()
def test_paged_sessions_with_pinned_voice_and_tool_bursts(training_reference: bool) -> None:
    run_session([3, 5], 72, torch.bfloat16, 256, 16, 4, prompt_frames=2, training_reference=training_reference)


@pytest.mark.parametrize("audio_frame", [511, 512, 2112, 2200])
def test_text_compaction_preserves_audio_pages(audio_frame: int) -> None:
    """Text retention cannot bound the independently advancing audio ring."""
    layout = DuplexIOKVLayout(1024, 2048, 24576)
    table = torch.arange(20, 20 + layout.max_blocks, dtype=torch.int32)[None]
    listed = torch.empty_like(table)
    compact = torch.empty(1, dtype=torch.int32)
    step_page_bounds(
        torch.tensor([398 * DUPLEXIO_NUM_CELLS]), table,
        torch.arange(layout.max_blocks, dtype=torch.int32), compact, listed, layout,
    )
    slots = torch.tensor([layout.audio_slot(audio_frame, cell) for cell in range(NUM_AUDIO_CELLS)])
    requests = torch.zeros_like(slots)
    torch.testing.assert_close(
        physical_slots(listed, requests, slots, layout.block_size),
        physical_slots(table, requests, slots, layout.block_size),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("rows", [1, 30, 70])
def test_only_pages_a_step_can_touch_are_listed(rows: int) -> None:
    """Keep the reserved audio pages and bound only the growing persistent region."""
    layout = DuplexIOKVLayout(
        block_size=BLOCK_SIZE,
        audio_window_frames=WINDOW,
        max_model_len=600 * DUPLEXIO_NUM_CELLS,
    )
    seq_lens = torch.tensor([rows * DUPLEXIO_NUM_CELLS])
    block_table = torch.arange(1, layout.max_blocks + 1, dtype=torch.int32)[None]
    compact = torch.zeros(1, dtype=torch.int32)
    listed = torch.zeros_like(block_table)

    step_page_bounds(
        seq_lens,
        block_table,
        torch.arange(layout.max_blocks, dtype=torch.int32),
        compact,
        listed,
        layout,
    )

    audio_pages = -(-layout.audio_slots // BLOCK_SIZE)
    persistent_pages = -(-rows * DUPLEXIO_NUM_TEXT_CELLS // BLOCK_SIZE)
    base = layout.persistent_base_page
    expected = {*range(audio_pages), *range(base, base + persistent_pages)}
    assert {page for page, entry in enumerate(listed[0].tolist()) if entry} == expected
    assert len(expected) < layout.max_blocks
    assert int(compact) == layout.persistent_base + rows * DUPLEXIO_NUM_TEXT_CELLS

    # Whatever the row writes has to be among the pages the step listed.
    frame = DuplexIOFrameMetadata(layout, DUPLEXIO_NUM_CELLS, torch.device("cpu"))
    fields = session_fields(*session_activity(rows, 3, 0))
    frame.update(
        **{name: fields[name][-DUPLEXIO_NUM_CELLS:] for name in FRAME_FIELDS}
    )
    written = frame.write_slots(DUPLEXIO_NUM_CELLS)
    assert (written >= 0).any()
    assert {int(slot) // BLOCK_SIZE for slot in written[written >= 0]} <= expected


PAGED_STEPS = pytest.mark.parametrize(
    "sizes,rows",
    [
        ([6], [1]),
        ([18, 6, 6], [40, 3, 70]),
        # A graph-padded batch can carry requests with no tokens.
        ([6, 0, 6, 0], [9, 1, 12, 1]),
        ([6] * 17, list(range(5, 90, 5))),
    ],
)


def paged_step(sizes: list[int], rows: list[int], heads: int = 4, kv_heads: int = 2):
    """Build one step's metadata over live frames, with a shuffled block table."""
    device = torch.device("cuda")
    spec = make_duplexio_kv_cache_spec(
        FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=kv_heads, head_size=64, head_size_v=64, dtype=torch.bfloat16),
        audio_window_frames=WINDOW,
        max_model_len=100 * DUPLEXIO_NUM_CELLS,
    )
    layout = spec.layout
    requests = len(sizes)
    pages = layout.max_blocks * requests
    frame = DuplexIOFrameMetadata(layout, sum(sizes), device)
    fields = [
        session_fields(*session_activity(row, 3, request), 2) for request, row in enumerate(rows)
    ]
    frame.update(**{
        name: torch.cat([field[name][field[name].shape[0] - size:] for field, size in zip(fields, sizes)]).to(device)
        for name in FRAME_FIELDS
    })
    layer, builder, impl = paged_backend(spec, frame, heads, requests, pages + 1)
    block_table = (torch.randperm(pages, device=device, dtype=torch.int32) + 1).view(requests, layout.max_blocks)
    seq_lens = [row * DUPLEXIO_NUM_CELLS for row in rows]
    return layer, builder, impl, block_table, seq_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@PAGED_STEPS
@torch.inference_mode()
def test_paged_tables_match_upstream_direct_build(sizes: list[int], rows: list[int]) -> None:
    _, builder, _, block_table, seq_lens = paged_step(sizes, rows)
    layout = builder.layout
    requests = len(sizes)
    metadata = step_metadata(builder, sizes, seq_lens, block_table)
    tables = {name: getattr(metadata, name).clone() for name in UPSTREAM_TABLES}

    # The upstream direct build over the same compact inputs. It sizes its
    # index table by max_model_len, which the compact layout outgrows.
    builder.max_num_kv_indices = builder.q_block_size * layout.max_blocks
    common = step_common(sizes, seq_lens, block_table)
    step_page_bounds(
        common.seq_lens,
        common.block_table_tensor,
        builder.page_ids,
        builder.compact_seq_lens[:requests],
        builder.touched_block_table[:requests],
        layout,
    )
    upstream = FlexAttentionMetadataBuilder.build(
        builder,
        0,
        common.replace(
            causal=False,
            seq_lens=builder.compact_seq_lens[:requests],
            max_seq_len=layout.max_compact_slots,
            block_table_tensor=builder.touched_block_table[:requests],
        ),
    )
    upstream.decode_offset.copy_(common.query_start_loc[:requests])

    for name in UPSTREAM_TABLES:
        torch.testing.assert_close(tables[name], getattr(upstream, name), rtol=0, atol=0, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@PAGED_STEPS
@pytest.mark.parametrize("heads", [4, 8])
@torch.inference_mode()
def test_kv_tiles_are_exactly_those_with_a_visible_key(sizes: list[int], rows: list[int], heads: int) -> None:
    """Each query block lists every tile its packed rows can see, once, and no other."""
    layer, builder, _, block_table, seq_lens = paged_step(sizes, rows, heads)
    metadata = step_metadata(builder, sizes, seq_lens, block_table)
    mask = metadata.block_mask
    tokens = sum(sizes)
    step_kv_tiles(
        layer.logical_mask_mod,
        metadata.doc_ids,
        metadata.seq_lens,
        metadata.block_table,
        mask.kv_num_blocks[:, 0],
        mask.kv_indices[:, 0],
        QUERY_BLOCK_ROWS // metadata.kv_groups,
        BLOCK_SIZE,
    )

    packed_rows = tokens * metadata.kv_groups
    assert mask.seq_lengths == (packed_rows, metadata.total_cache_tokens)
    device = block_table.device
    zero = torch.zeros((), dtype=torch.long, device=device)
    seen = metadata.packed_mask_mod(
        zero, zero, torch.arange(packed_rows, device=device)[:, None],
        torch.arange(metadata.total_cache_tokens, device=device)[None],
    )
    blocks = -(-packed_rows // QUERY_BLOCK_ROWS)
    seen = torch.nn.functional.pad(seen, (0, 0, 0, blocks * QUERY_BLOCK_ROWS - packed_rows))
    visible_tiles = seen.view(blocks, QUERY_BLOCK_ROWS, -1, KV_TILE).any(-1).any(1)
    counts = mask.kv_num_blocks[:, 0]
    assert counts.shape == (kv_splits(tokens), blocks)
    assert (counts.amax(0) - counts.amin(0) <= 1).all()
    assert bool(visible_tiles.any()) == (max(rows) > 1)
    for block in range(blocks):
        listed = torch.cat([mask.kv_indices[split, 0, block, :count] for split, count in enumerate(counts[:, block].tolist())])
        assert listed.unique().numel() == listed.numel()
        assert set(listed.tolist()) == set(visible_tiles[block].nonzero()[:, 0].tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@torch.inference_mode()
def test_tiled_split_kv_matches_the_page_granular_kernel() -> None:
    """Rollout-shaped decode: 1024-key pages, a wrapped 2048-frame window, a pinned prompt.

    The page-granular path runs upstream's direct block mask with per-head
    query rows and one KV pass. Both see the same keys through the same bf16
    kernel arithmetic; the tiled path only reorders the reduction into splits
    and rounds each split to bf16 before the FP32 merge. So both stay within
    bf16 noise of an FP32 reference, and of each other.
    """
    # A second layout in one process turns its constants symbolic, which
    # Inductor's flex lowering cannot take. Serving builds a single layout.
    torch._dynamo.reset()
    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, kv_heads, dim = 16, 4, 64
    spec = make_duplexio_kv_cache_spec(
        FullAttentionSpec(block_size=1024, num_kv_heads=kv_heads, head_size=dim, head_size_v=dim, dtype=torch.bfloat16),
        audio_window_frames=2048,
        max_model_len=2400 * DUPLEXIO_NUM_CELLS,
    )
    layout = spec.layout
    frames = [126, 700, 1500, 2150, 2600, 3300]
    requests, tokens = len(frames), len(frames) * DUPLEXIO_NUM_CELLS
    # A 125-frame voice prompt, then the emitted text.
    keys = [125 + 400 + frame // 3 for frame in frames]
    frame = DuplexIOFrameMetadata(layout, tokens, device)

    def per_row(values: list[int]) -> Tensor:
        return cells(torch.tensor(values, dtype=torch.int32, device=device))

    frame.update(
        key_active=torch.ones(tokens, dtype=torch.bool, device=device),
        persistent_ordinal=torch.zeros(tokens, dtype=torch.int32, device=device),
        persistent_last=per_row(keys),
        audio_first=per_row([max(value - 2048, 1) for value in frames]),
        audio_last=per_row([value - 1 for value in frames]),
    )
    pages = layout.max_blocks * requests
    layer, builder, impl = paged_backend(spec, frame, heads, requests, pages + 1)
    cache = torch.randn(pages + 1, 1024, kv_heads, 2 * dim, device=device, dtype=torch.bfloat16).transpose(1, 2)
    # Halved values keep the outputs below 0.25 (see the tolerance below).
    cache[..., dim:] *= 0.5
    cache[0].fill_(torch.nan)
    block_table = (torch.randperm(pages, device=device, dtype=torch.int32) + 1).view(requests, layout.max_blocks)
    sizes = [DUPLEXIO_NUM_CELLS] * requests
    seq_lens = [(-(-key // DUPLEXIO_NUM_TEXT_CELLS) + 1) * DUPLEXIO_NUM_CELLS for key in keys]
    query = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(tokens, kv_heads, dim, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key) * 0.5
    scale = dim**-0.5

    metadata = step_metadata(builder, sizes, seq_lens, block_table)
    assert metadata.block_mask.kv_num_blocks.shape[0] > 1
    tiled = impl.forward(layer, query, key, value, cache, metadata, torch.empty_like(query))

    builder.max_num_kv_indices = builder.q_block_size * layout.max_blocks
    common = step_common(sizes, seq_lens, block_table)
    upstream = FlexAttentionMetadataBuilder.build(
        builder,
        0,
        common.replace(
            causal=False,
            seq_lens=metadata.seq_lens,
            max_seq_len=layout.max_compact_slots,
            block_table_tensor=metadata.block_table,
        ),
    )
    upstream.decode_offset.copy_(common.query_start_loc[:requests])
    keys, values = cache.transpose(1, 2).split(dim, dim=-1)
    keys, values = keys.reshape(-1, kv_heads, dim), values.reshape(-1, kv_heads, dim)
    history, auxiliary = torch.compile(flex_attention, fullgraph=True)(
        query.transpose(0, 1)[None], keys.transpose(0, 1)[None], values.transpose(0, 1)[None],
        block_mask=upstream.block_mask, scale=scale, enable_gqa=True, return_aux=AuxRequest(lse=True),
        kernel_options={"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": 16, "BLOCK_N": 64},
    )
    by_head = (kv_heads, heads // kv_heads)
    paged = merge_self_attention(
        query, key, value,
        history[0].transpose(0, 1).unflatten(1, by_head)[None],
        auxiliary.lse[0].transpose(0, 1).unflatten(1, by_head)[None],
        scale,
    )

    # FP32 over each request's own pages, the diagonal appended as one more key.
    zero = torch.zeros((), dtype=torch.long, device=device)
    reference = []
    for request in range(requests):
        slots = (block_table[request].long()[:, None] * 1024 + torch.arange(1024, device=device)).flatten()
        rows = torch.arange(request * DUPLEXIO_NUM_CELLS, (request + 1) * DUPLEXIO_NUM_CELLS, device=device)
        seen = torch.cat(
            (metadata.mask_mod(zero, zero, rows[:, None], slots[None]), torch.ones_like(rows, dtype=torch.bool)[:, None]), -1
        )
        row_keys = torch.cat((keys[slots][None].expand(rows.shape[0], -1, -1, -1), key[rows, None]), 1)
        row_values = torch.cat((values[slots][None].expand(rows.shape[0], -1, -1, -1), value[rows, None]), 1)
        scores = torch.einsum(
            "thgd,tshd->thgs", query[rows].float().unflatten(1, by_head), row_keys.float()
        ).mul(scale).masked_fill(~seen[:, None, None], -torch.inf)
        reference.append(torch.einsum("thgs,tshd->thgd", scores.softmax(-1), row_values.float()).flatten(1, 2))
    reference = torch.cat(reference)

    paged_error = (paged.float() - reference).abs().max()
    tiled_error = (tiled.float() - reference).abs().max()
    assert tiled_error <= 2 * paged_error
    # Outputs stay below 0.25, where a bf16 ulp is 2**-10; the paths differ in
    # reduction order and one rounding per split, so allow two ulps.
    assert reference.abs().max() < 0.25
    torch.testing.assert_close(tiled, paged, atol=2**-9, rtol=0)
