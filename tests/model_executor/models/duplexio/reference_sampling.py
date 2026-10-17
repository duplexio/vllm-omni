# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-row reference samplers that the batched, captured sampling is checked against."""

from dataclasses import dataclass
from typing import Any

import torch
import xgrammar as xgr
from torch import Tensor

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    TextSamplingResult,
)
from vllm_omni.model_executor.models.duplexio.tool_calling import ToolCallConstraintState


@dataclass(frozen=True)
class ReferenceSampling:
    temperature: float
    top_k: int | None
    top_p: float | None
    suppressed_token_ids: Tensor


@dataclass
class ToolTokenSample:
    token_id: Tensor
    logprob: Tensor


def content_distribution(logits: Tensor, sampling: ReferenceSampling) -> tuple[Tensor, Tensor]:
    """Top-k token IDs and probabilities, applying top-p within that cap."""
    if sampling.suppressed_token_ids.numel():
        logits = logits.clone()
        logits.index_fill_(-1, sampling.suppressed_token_ids, torch.finfo(logits.dtype).min)
    scaled = logits.float() / sampling.temperature
    if sampling.top_k is None and sampling.top_p is None:
        indices = torch.arange(scaled.shape[-1], device=scaled.device).expand_as(scaled)
        return indices, torch.softmax(scaled, dim=-1)
    k = min(sampling.top_k, scaled.shape[-1]) if sampling.top_k is not None else scaled.shape[-1]
    top_values, top_indices = torch.topk(scaled, k=k, dim=-1)
    if sampling.top_p is not None:
        remove = torch.softmax(top_values, dim=-1).cumsum(dim=-1) > sampling.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        top_values = top_values.masked_fill(remove, torch.finfo(top_values.dtype).min)
    return top_indices, torch.softmax(top_values, dim=-1)


def sample_content_token_ids(
    logits: Tensor,
    sampling: ReferenceSampling,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    if sampling.temperature == 0:
        if sampling.suppressed_token_ids.numel():
            logits = logits.clone()
            logits.index_fill_(-1, sampling.suppressed_token_ids, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)
    top_indices, probabilities = content_distribution(logits, sampling)
    sampled = torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
    return top_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


def sample_emit(emit_logits: Tensor, temperature: float, *, generator: torch.Generator | None = None) -> Tensor:
    if temperature == 0:
        return emit_logits >= 0
    return torch.bernoulli(torch.sigmoid(emit_logits.float() / temperature), generator=generator).bool()


def sample_factorized_text_ids(
    logits: Tensor,
    emit_logits: Tensor,
    *,
    silence_token_id: int,
    sampling: ReferenceSampling,
    emit_temperature: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw speak-versus-silence independently of the content token."""
    emit = sample_emit(emit_logits, emit_temperature, generator=generator)
    content_ids = sample_content_token_ids(logits, sampling, generator=generator)
    return torch.where(emit, content_ids, torch.full_like(content_ids, silence_token_id))


def exponential_race(probabilities: Tensor, generator: torch.Generator | None = None) -> Tensor:
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    return probabilities.div(noise).argmax(dim=-1)


def sample_tool_token(
    logits: Tensor,
    *,
    constraint: ToolCallConstraintState | None,
    emit: bool,
    sampling: ReferenceSampling,
    generator: torch.Generator | None = None,
) -> ToolTokenSample | None:
    """Sample and score the grammar-constrained distribution in one draw."""
    if constraint is None or not constraint.enabled:
        return None
    if not constraint.active:
        if not emit:
            return None
        constraint.begin()
    constrained_logits = logits.clone()
    bitmask = xgr.allocate_token_bitmask(1, constrained_logits.shape[-1])
    constraint.matcher.fill_next_token_bitmask(bitmask)
    xgr.apply_token_bitmask_inplace(
        constrained_logits,
        bitmask.to(constrained_logits.device),
        vocab_size=constrained_logits.shape[-1],
    )
    if sampling.temperature == 0:
        token_id = sample_content_token_ids(constrained_logits, sampling, generator=generator)
        return ToolTokenSample(token_id=token_id, logprob=torch.zeros_like(token_id, dtype=torch.float32))
    indices, probabilities = content_distribution(constrained_logits, sampling)
    selected = exponential_race(probabilities, generator).unsqueeze(-1)
    return ToolTokenSample(
        token_id=indices.gather(-1, selected).squeeze(-1),
        logprob=probabilities.gather(-1, selected).squeeze(-1).float().log(),
    )


def sample_text_batch(
    model: DuplexIOForConditionalGeneration,
    logits: Tensor,
    emit_logits: Tensor,
    infos: list[dict[str, Any]],
) -> TextSamplingResult:
    """Run the model's batched text sampling outside a frame step, then apply the host's tool decisions."""
    inputs = model.sampling_inputs(infos, logits.device)
    text_ids, tool_starts, frame_logprobs, support_ids = model.sample_text(
        logits,
        emit_logits,
        inputs.parameters,
        inputs.tool_bitmask,
        top_k=inputs.top_k,
        support_width=inputs.support_width,
    )
    text = TextSamplingResult(
        text_ids,
        tool_starts,
        frame_logprobs,
        support_ids,
        inputs.pending,
        inputs.calling,
        [None] * len(infos),
    )
    model.finish_text_batch(text, infos, text_ids.tolist(), tool_starts.tolist())
    return text
