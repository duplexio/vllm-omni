# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The paged backend must reproduce dense DuplexIO attention over live sessions."""

from collections.abc import Sequence
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOFrameMetadata,
    DuplexIOKVCacheSpec,
    DuplexIOKVLayout,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOFlexAttentionImpl,
    DuplexIOFlexAttentionMetadataBuilder,
    DuplexIOPagedAttention,
    step_page_bounds,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    duplexio_attention_visible,
)

WINDOW = 2
# FlexAttention tiles a page with BLOCK_N, so a page cannot be smaller than it.
BLOCK_SIZE = 64
NUM_AUDIO_CELLS = DUPLEXIO_NUM_CELLS - DUPLEXIO_NUM_TEXT_CELLS
FRAME_FIELDS = ("key_active", "text_ordinal", "text_last", "audio_first", "audio_last")
# Paged attention reduces the history in page order and merges the diagonal
# afterwards, so it does not round like a dense matmul. Bit-exactness with
# training was traded for that simpler reduction.
TOLERANCE = {torch.float32: 2e-5, torch.bfloat16: 3e-2}


def cells(values: Tensor) -> Tensor:
    """Give every cell of a row the row's value."""
    return values[:, None].expand(-1, DUPLEXIO_NUM_CELLS).flatten()


def session_fields(audio_active: Tensor, text_active: Tensor) -> dict[str, Tensor]:
    """Per-token frame fields for a whole session, as ``frame_inputs`` derives them.

    ``audio_active`` is (rows,) and ``text_active`` (rows, 4). A row without
    active audio is a token burst - a spliced tool result - which keeps the audio
    clock frozen and therefore consumes no window budget.
    """
    rows = audio_active.shape[0]
    audio_position = audio_active.cumsum(0, dtype=torch.int32)
    emitted = text_active.int().flatten().cumsum(0, dtype=torch.int32).view(rows, -1)
    audio_padding = torch.zeros(rows, NUM_AUDIO_CELLS, dtype=torch.int32)
    return {
        "key_active": torch.cat(
            (text_active, audio_active[:, None].expand(-1, NUM_AUDIO_CELLS)), 1
        ).flatten(),
        "text_ordinal": torch.cat(
            (torch.where(text_active, emitted, 0), audio_padding), 1
        ).flatten(),
        "text_last": cells(
            torch.cat((torch.zeros(1, dtype=torch.int32), emitted[:-1, -1]))
        ),
        "audio_first": cells((audio_position - WINDOW).clamp_min(1)),
        "audio_last": cells(audio_position - audio_active.int()),
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
        audio_attention_window_frames=WINDOW,
    )
    groups = query.shape[1] // key.shape[1]
    keys = key.float().repeat_interleave(groups, 1)
    values = value.float().repeat_interleave(groups, 1)
    scores = torch.einsum("thd,shd->hts", query.float(), keys) * scale
    return torch.einsum(
        "hts,shd->thd", scores.masked_fill(~visible, -torch.inf).softmax(-1), values
    )


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


def step_metadata(
    builder: DuplexIOFlexAttentionMetadataBuilder,
    sizes: list[int],
    seq_lens: list[int],
    block_table: Tensor,
) -> FlexAttentionMetadata:
    device = block_table.device
    starts = torch.tensor([0, *accumulate(sizes)], dtype=torch.int32)
    return builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=starts.to(device),
            query_start_loc_cpu=starts,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            num_reqs=len(sizes),
            num_actual_tokens=sum(sizes),
            max_query_len=max(sizes),
            max_seq_len=max(seq_lens),
            block_table_tensor=block_table,
            slot_mapping=torch.zeros(sum(sizes), dtype=torch.long, device=device),
        ),
    )


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
                *session_activity(rows, prefix, request)
            ).items()
        }
        for request, prefix in enumerate(prefixes)
    ]
    expected = [
        dense_attention(query[i], key[i], value[i], fields[i], dim**-0.5)
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
@torch.inference_mode()
def test_batched_sessions_stay_isolated_in_a_shuffled_block_table() -> None:
    run_session([3, 5, 4], 20, torch.bfloat16, 128, 8, 2)


def test_only_pages_a_step_can_touch_are_listed() -> None:
    """The scan must follow the session, not the layout bound.

    A long-session layout leaves a wide band of never-written pages between the
    audio ring and the text base, and scanning one costs as much as scanning
    live keys.
    """
    rows = 30
    layout = DuplexIOKVLayout(
        block_size=BLOCK_SIZE,
        audio_window_frames=WINDOW,
        max_model_len=600 * DUPLEXIO_NUM_CELLS,
    )
    text_base = layout.text_base_page
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

    audio_pages = -(-rows * NUM_AUDIO_CELLS // BLOCK_SIZE)
    text_pages = -(-rows * DUPLEXIO_NUM_TEXT_CELLS // BLOCK_SIZE)
    expected = {*range(audio_pages), *range(text_base, text_base + text_pages)}
    assert {page for page, entry in enumerate(listed[0].tolist()) if entry} == expected
    assert len(expected) < layout.max_blocks
    assert int(compact) == layout.persistent_text_base + rows * DUPLEXIO_NUM_TEXT_CELLS

    # Whatever the row writes has to be among the pages the step listed.
    frame = DuplexIOFrameMetadata(layout, DUPLEXIO_NUM_CELLS, torch.device("cpu"))
    fields = session_fields(*session_activity(rows, 3, 0))
    frame.update(
        **{name: fields[name][-DUPLEXIO_NUM_CELLS:] for name in FRAME_FIELDS}
    )
    written = frame.write_slots(DUPLEXIO_NUM_CELLS)
    assert (written >= 0).any()
    assert {int(slot) // BLOCK_SIZE for slot in written[written >= 0]} <= expected
