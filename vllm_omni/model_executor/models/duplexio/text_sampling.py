"""Deterministic token filtering, separate from request-owned random draws."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TokenSamplingOptions:
    mode: str
    temperature: float
    top_k: int
    top_p: float
    suppressed_token_ids: Tensor


def content_distribution(logits: Tensor, sampling: TokenSamplingOptions) -> tuple[Tensor, Tensor]:
    """Return top-k token IDs and probabilities, applying top-p within that cap."""
    if sampling.suppressed_token_ids.numel():
        logits = logits.clone()
        logits.index_fill_(-1, sampling.suppressed_token_ids, torch.finfo(logits.dtype).min)
    scaled = logits.float() / sampling.temperature
    top_values, top_indices = torch.topk(scaled, k=min(sampling.top_k, scaled.shape[-1]), dim=-1)
    if sampling.mode == "top_p":
        sorted_probabilities = torch.softmax(top_values, dim=-1)
        remove = sorted_probabilities.cumsum(dim=-1) > sampling.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        top_values = top_values.masked_fill(remove, torch.finfo(top_values.dtype).min)
    elif sampling.mode != "top_k":
        raise ValueError(f"Unsupported DuplexIO sampling mode: {sampling.mode!r}")
    return top_indices, torch.softmax(top_values, dim=-1)
