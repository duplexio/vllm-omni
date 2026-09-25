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
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOFrameMetadata,
    DuplexIOKVCacheSpec,
    make_duplexio_kv_cache_spec,
)
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOAttentionMetadata,
    DuplexIOFlashAttentionImpl,
    DuplexIOFlashAttentionMetadataBuilder,
    DuplexIOPagedAttention,
)
from vllm_omni.model_executor.models.duplexio.row_semantics import (
    DUPLEXIO_NUM_CELLS,
    DUPLEXIO_NUM_TEXT_CELLS,
    duplexio_attention_visible,
)

WINDOW = 2
BLOCK_SIZE = 64
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
TOLERANCE = {torch.bfloat16: 3e-2}


def cells(values: Tensor) -> Tensor:
    """Give every cell of a row the row's value."""
    return values[:, None].expand(-1, DUPLEXIO_NUM_CELLS).flatten()


def session_fields(
    audio_active: Tensor, text_active: Tensor, prompt_frames: int = 0, window: int = WINDOW,
) -> dict[str, Tensor]:
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
        "audio_first": cells((audio_position - window).clamp_min(1)),
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
    query: Tensor, key: Tensor, value: Tensor, fields: dict[str, Tensor], scale: float, window: int,
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
        audio_attention_window_frames=window,
    )
    groups = query.shape[1] // key.shape[1]
    keys = key.float().repeat_interleave(groups, 1)
    values = value.float().repeat_interleave(groups, 1)
    scores = torch.einsum("thd,shd->hts", query.float(), keys) * scale
    return torch.einsum(
        "hts,shd->thd", scores.masked_fill(~visible, -torch.inf).softmax(-1), values
    )


def training_attention(
    query: Tensor, key: Tensor, value: Tensor, fields: dict[str, Tensor], scale: float, window: int,
) -> Tensor:
    training = pytest.importorskip("duplexio.models.duplexio")
    from duplexio.modules.stream_attention import unified_attention

    positions = torch.arange(query.shape[0], device=query.device)
    sequence = torch.zeros_like(positions)
    frame = positions // DUPLEXIO_NUM_CELLS
    cell = positions % DUPLEXIO_NUM_CELLS
    audio = fields["audio_position"]
    mask, indices = training.attention_mask(
        sequence, frame, audio, cell, sequence, frame, audio, cell,
        fields["key_active"], fields["pinned"], window, query.shape[-1],
    )
    return unified_attention(
        query.transpose(0, 1)[None], key[indices].transpose(0, 1)[None],
        value[indices].transpose(0, 1)[None], mask, scale,
    )[0].transpose(0, 1).float()


def paged_backend(
    spec: DuplexIOKVCacheSpec, frame: DuplexIOFrameMetadata, heads: int,
) -> tuple[nn.Module, DuplexIOFlashAttentionMetadataBuilder, DuplexIOFlashAttentionImpl]:
    """Wire the real builder and impl to one attention layer's frame metadata."""
    device = frame.cell.device
    layer = object.__new__(DuplexIOPagedAttention)
    nn.Module.__init__(layer)
    layer._k_scale = torch.ones((), device=device)
    layer._v_scale = torch.ones((), device=device)
    layer.frame = frame
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"attn": layer}),
    )
    builder = DuplexIOFlashAttentionMetadataBuilder(spec, ["attn"], vllm_config, device)
    impl = DuplexIOFlashAttentionImpl(heads, spec.head_size, spec.head_size**-0.5, spec.num_kv_heads, None, None, "auto")
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
    builder: DuplexIOFlashAttentionMetadataBuilder,
    sizes: list[int],
    seq_lens: list[int],
    block_table: Tensor,
) -> DuplexIOAttentionMetadata:
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
    window: int = WINDOW,
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
        audio_window_frames=window,
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
                *session_activity(rows, prefix, request), prompt_frames, window,
            ).items()
        }
        for request, prefix in enumerate(prefixes)
    ]
    reference = training_attention if training_reference else dense_attention
    expected = [
        reference(query[i], key[i], value[i], fields[i], dim**-0.5, window)
        for i in range(requests)
    ]

    pages = layout.max_blocks * requests
    frame = DuplexIOFrameMetadata(
        layout, sum(prefixes) * DUPLEXIO_NUM_CELLS, device
    )
    layer, builder, impl = paged_backend(spec, frame, heads)
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
    "dim,heads,kv_heads,rows,window",
    (
        (128, 4, 2, 24, 2),
        # The ring spans whole pages, so only a long session wraps it and the
        # window's page table has to rotate.
        (256, 16, 4, 72, 2),
        # A window wider than a page, whose table lists a ring page twice.
        (256, 16, 4, 72, 40),
        # No audio window: a frozen row still sees its last frame.
        (128, 8, 2, 24, 0),
    ),
)
@torch.inference_mode()
def test_paged_session_matches_dense_attention(
    dim: int, heads: int, kv_heads: int, rows: int, window: int
) -> None:
    run_session([3], rows, torch.bfloat16, dim, heads, kv_heads, window=window)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged cache")
@torch.inference_mode()
def test_long_prefill_matches_dense_attention() -> None:
    # Hundreds of rows in one step, most with an empty audio read: FA2's
    # grouped decode path lets an empty sequence overwrite other rows'
    # normalizers, which a handful of rows does not expose.
    run_session([336], 340, torch.bfloat16, 256, 16, 4)


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
