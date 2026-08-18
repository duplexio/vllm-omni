# SPDX-License-Identifier: Apache-2.0
"""Arbitrary-width causal-conv fallback for the dilated DuplexIO stream conv.

DuplexIO expands Qwen's 4-tap GDN convolution into a dilated row-major kernel
of (4 - 1) * 6 + 1 = 19 taps (see ``expand_stream_conv_weight``). Upstream
vLLM's Triton conv kernels specialize KERNEL_WIDTH in {2..5} and fail to
compile anything wider, so this module routes wide kernels through a plain
PyTorch implementation with identical semantics and leaves narrow widths on
the Triton fast path.

The reference deliberately supports only the argument shapes DuplexIO's
runtime contract produces (no speculative decode, no chunked prefill, no
prefix-cache block chaining) and fails fast on anything else.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MAX_TRITON_KERNEL_WIDTH = 5


def _apply_activation(out: torch.Tensor, activation: str | bool | None) -> torch.Tensor:
    if activation is None or activation is False:
        return out
    if activation is True or activation in ("silu", "swish"):
        return F.silu(out)
    raise ValueError(f"Unsupported causal-conv activation: {activation!r}")


def _require_none(**kwargs: object) -> None:
    for name, value in kwargs.items():
        if value is not None:
            raise ValueError(
                f"wide causal-conv fallback does not support {name!r}"
            )


def _state_view(state: torch.Tensor, width: int) -> torch.Tensor:
    """Return the per-sequence conv state as (dim, width - 1).

    vLLM's cache allocator may orient the state as (dim, state_len) or
    (state_len, dim); both views share storage, so in-place updates work
    through the transpose.
    """
    if state.shape[-1] == width - 1:
        return state
    if state.shape[-2] == width - 1:
        return state.transpose(-1, -2)
    raise ValueError(
        f"conv state shape {tuple(state.shape)} does not match width {width}"
    )


def wide_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int | None = None,
    metadata: object = None,
    validate_data: bool = False,
    **advanced: object,
) -> torch.Tensor:
    """Varlen prefill conv over ``x: (dim, total_tokens)``; updates states."""
    del metadata, validate_data
    _require_none(**advanced)
    dim, _total = x.shape
    width = weight.shape[1]
    compute_dtype = conv_states.dtype
    out = torch.empty_like(x)
    starts = query_start_loc.tolist()
    grouped_weight = weight.to(compute_dtype).unsqueeze(1)
    for i in range(len(starts) - 1):
        start, end = int(starts[i]), int(starts[i + 1])
        if end <= start:
            continue
        state_index = int(cache_indices[i]) if cache_indices is not None else i
        if pad_slot_id is not None and state_index == pad_slot_id:
            continue
        seq = x[:, start:end].to(compute_dtype)
        state = _state_view(conv_states[state_index], width)
        if has_initial_state is not None and bool(has_initial_state[i]):
            initial = state
        else:
            initial = seq.new_zeros(dim, width - 1)
        context = torch.cat((initial, seq), dim=1)
        convolved = F.conv1d(
            context.unsqueeze(0),
            grouped_weight,
            bias=None if bias is None else bias.to(compute_dtype),
            groups=dim,
        ).squeeze(0)
        out[:, start:end] = _apply_activation(convolved, activation).to(x.dtype)
        state.copy_(context[:, -(width - 1):])
    return out


def wide_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int | None = None,
    validate_data: bool = False,
    **advanced: object,
) -> torch.Tensor:
    """Decode-step conv over ``x: (num_tokens, dim)``; updates states."""
    del max_query_len, validate_data
    _require_none(num_accepted_tokens=num_accepted_tokens, **advanced)
    if x.ndim != 2:
        raise ValueError(
            f"wide causal-conv fallback expects 2D decode input, got {x.shape}"
        )
    width = weight.shape[1]
    compute_dtype = conv_state.dtype
    dim = x.shape[1]
    out = torch.empty_like(x)
    if query_start_loc is None:
        starts = list(range(x.shape[0] + 1))
    else:
        starts = query_start_loc.tolist()
    grouped_weight = weight.to(compute_dtype).unsqueeze(1)
    for i in range(len(starts) - 1):
        start, end = int(starts[i]), int(starts[i + 1])
        if end <= start:
            continue
        state_index = (
            int(conv_state_indices[i]) if conv_state_indices is not None else i
        )
        if null_block_id is not None and state_index == null_block_id:
            continue
        seq = x[start:end].transpose(0, 1).to(compute_dtype)
        state = _state_view(conv_state[state_index], width)
        context = torch.cat((state, seq), dim=1)
        convolved = F.conv1d(
            context.unsqueeze(0),
            grouped_weight,
            bias=None if bias is None else bias.to(compute_dtype),
            groups=dim,
        ).squeeze(0)
        out[start:end] = (
            _apply_activation(convolved, activation).transpose(0, 1).to(x.dtype)
        )
        state.copy_(context[:, -(width - 1):])
    return out


def install_wide_conv_fallback() -> None:
    """Route wide-kernel calls in the GDN core to the PyTorch fallback."""
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_module

    original_fn = gdn_module.causal_conv1d_fn
    original_update = gdn_module.causal_conv1d_update
    if getattr(original_fn, "_duplexio_wide_conv", False):
        return

    def dispatch_fn(x, weight, *args, **kwargs):
        if weight.shape[-1] > _MAX_TRITON_KERNEL_WIDTH:
            return wide_causal_conv1d_fn(x, weight, *args, **kwargs)
        return original_fn(x, weight, *args, **kwargs)

    def dispatch_update(x, conv_state, weight, *args, **kwargs):
        if weight.shape[-1] > _MAX_TRITON_KERNEL_WIDTH:
            return wide_causal_conv1d_update(x, conv_state, weight, *args, **kwargs)
        return original_update(x, conv_state, weight, *args, **kwargs)

    dispatch_fn._duplexio_wide_conv = True  # type: ignore[attr-defined]
    dispatch_update._duplexio_wide_conv = True  # type: ignore[attr-defined]
    gdn_module.causal_conv1d_fn = dispatch_fn
    gdn_module.causal_conv1d_update = dispatch_update
