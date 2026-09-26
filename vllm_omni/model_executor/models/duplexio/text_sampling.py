"""Deterministic token filtering, separate from request-owned random draws."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TokenSamplingOptions:
    temperature: float
    top_k: int | None
    top_p: float | None
    suppressed_token_ids: Tensor


def content_distribution(logits: Tensor, sampling: TokenSamplingOptions) -> tuple[Tensor, Tensor]:
    """Return top-k token IDs and probabilities, applying top-p within that cap."""
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
        sorted_probabilities = torch.softmax(top_values, dim=-1)
        remove = sorted_probabilities.cumsum(dim=-1) > sampling.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        top_values = top_values.masked_fill(remove, torch.finfo(top_values.dtype).min)
    return top_indices, torch.softmax(top_values, dim=-1)


# Columns of the per-row parameters that ``sample_streams`` reads; each setting
# holds one value per stream, in logits order (agent, tool, user).
TEMPERATURE, TOP_K, TOP_P, EMIT_TEMPERATURE = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
# 0: no tool grammar, 1: inside or forced into a call, 2: idle and free to start one.
TOOL_STATE = 12
SAMPLING_PARAMETERS = 13
NO_TOP_P = 2.0  # A cumulative probability never exceeds it.


def sampling_parameters(agent: TokenSamplingOptions, user: TokenSamplingOptions, emit: tuple[float, float, float],
                        vocab_size: int) -> tuple[float, ...]:
    """A request's row of ``sample_streams`` parameters, without its tool state."""
    streams = (agent, agent, user)
    return (
        *(options.temperature for options in streams),
        *(float(options.top_k or vocab_size) for options in streams),
        *(NO_TOP_P if options.top_p is None else options.top_p for options in streams),
        *emit,
    )


def sampled_top_k(streams: list[TokenSamplingOptions], vocab_size: int) -> int | None:
    """The one top-k width a batch shares; ``None`` samples the whole softmax.

    Greedy rows take the distribution's mode at any width.
    """
    sampled = [options for options in streams if options.temperature != 0]
    if not sampled:
        return 1
    if all(options.top_k is None and options.top_p is None for options in sampled):
        return None
    return min(vocab_size, max(options.top_k or vocab_size for options in sampled))


def support_width(agents: list[TokenSamplingOptions], top_k: int | None) -> int:
    """Width of the agent and tool content supports a batch records; 0 records none.

    Supports are recorded only when every sampled agent stream is top-k truncated,
    so each fits the widest one. A greedy stream's support is its argmax.
    """
    sampled = [options for options in agents if options.temperature != 0]
    if top_k is None or any(options.top_k is None for options in sampled):
        return 0
    return min(top_k, max((options.top_k for options in sampled), default=1))


def sample_streams(
    logits: Tensor, emit_logits: Tensor, parameters: Tensor, blocked: Tensor, *, top_k: int | None,
    support_width: int, silence_token_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Draw every row's three emit decisions and content tokens in one pass.

    ``logits`` is ``[rows, 3, vocab]`` and ``emit_logits`` ``[rows, 3]``, in stream order
    (agent, tool, user); ``blocked`` marks suppressed and grammar-forbidden ids. Each row
    brings its own settings in ``parameters``; only the widest ``top_k`` is shared.

    Returns the ``[rows, 3]`` ids in text-input order (user, agent, tool), the idle rows
    that started a call, the six ``[rows, 6]`` frame log probabilities, and the
    ``[rows, 2, support_width]`` agent and tool content supports: the ids the draw
    could pick, best first and padded with -1. A learner that renormalizes over a
    recorded support scores exactly the distribution sampled, even where its own
    logits would move a top-p or top-k boundary.
    """
    temperature, emit_temperature = parameters[:, TEMPERATURE], parameters[:, EMIT_TEMPERATURE]
    greedy = temperature == 0
    scaled = (logits.float() / temperature.masked_fill(greedy, 1).unsqueeze(-1)).masked_fill(
        blocked, torch.finfo(torch.float32).min,
    )
    if top_k is None:
        indices, probabilities = None, torch.softmax(scaled, dim=-1)
    else:
        values, indices = torch.topk(scaled, k=top_k, dim=-1)
        rank = torch.arange(top_k, device=values.device)
        values = values.masked_fill(rank >= parameters[:, TOP_K, None], torch.finfo(values.dtype).min)
        # Keep the smallest prefix reaching top-p, including the token that crosses it.
        crossed = torch.softmax(values, dim=-1).cumsum(dim=-1) > parameters[:, TOP_P, None]
        remove = F.pad(crossed[..., :-1], (1, 0))
        probabilities = torch.softmax(values.masked_fill(remove, torch.finfo(values.dtype).min), dim=-1)
    # The exponential race samples as vLLM does, without multinomial's validation sync.
    raced = probabilities.div(torch.empty_like(probabilities).exponential_()).argmax(dim=-1)
    selected = torch.where(greedy, probabilities.argmax(dim=-1), raced).unsqueeze(-1)
    token_logprobs = probabilities.gather(-1, selected).squeeze(-1).log().masked_fill(greedy, 0)
    content = selected.squeeze(-1) if indices is None else indices.gather(-1, selected).squeeze(-1)
    if indices is None or not support_width:
        support = torch.full((logits.shape[0], 2, 0), -1, dtype=torch.int32, device=logits.device)
    else:
        kept = probabilities[:, :2, :support_width] > 0
        kept[..., 1:] &= ~greedy[:, :2, None]
        support = indices[:, :2, :support_width].masked_fill(~kept, -1).int()

    emit_logits = emit_logits.float()
    emit_greedy = emit_temperature == 0
    probability = torch.sigmoid(emit_logits / emit_temperature.masked_fill(emit_greedy, 1))
    emitted = torch.where(emit_greedy, emit_logits >= 0, torch.rand_like(probability) < probability)
    emit_logprobs = torch.where(emitted, probability, 1 - probability).log().masked_fill(emit_greedy, 0)
    # A call's rows always emit; an idle row's emit draw decides whether one starts.
    tool_state = parameters[:, TOOL_STATE]
    tool_starts = emitted[:, 1] & (tool_state == 2)
    tool_emitted = (tool_state == 1) | tool_starts
    emitted = torch.stack((emitted[:, 0], tool_emitted, emitted[:, 2]), dim=1)
    # Score the tool decision with the raw head, even when serving forced it.
    start_logits = emit_logits[:, 1]
    emit_logprobs = torch.stack((
        emit_logprobs[:, 0], F.logsigmoid(torch.where(tool_emitted, start_logits, -start_logits)), emit_logprobs[:, 2],
    ), dim=1)
    ids = torch.where(emitted, content, silence_token_id)
    # A discarded content draw on a wait frame is not an action.
    token_logprobs = torch.where(emitted, token_logprobs, 0)
    return (
        torch.stack((ids[:, 2], ids[:, 0], ids[:, 1]), dim=1), tool_starts,
        torch.stack((emit_logprobs, token_logprobs), dim=-1).flatten(1), support,
    )
