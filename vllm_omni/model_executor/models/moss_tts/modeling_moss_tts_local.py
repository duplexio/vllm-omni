# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Local depth transformer for MossTTSRealtime.

A small (4-layer) Qwen3-style decoder that generates the rvq=16 RVQ codebook
codes for one audio frame, autoregressively over codebooks. It runs inside the
talker's per-step ``make_omni_output``, independent from vLLM's main scheduler.

The transformer body is shared with Qwen3-TTS and Qwen3-Omni via
``common.qwen3_code_predictor.CodePredictorBaseModel`` (HF-compatible
numerics). MossTTSRealtime differs in two ways:

  * codebook 0 is generated here from ``backbone_last_hidden`` (the other models
    receive it from the talker's main LM head), so we run one extra step and own
    all ``rvq`` LM heads;
  * sampling adds top-p and a windowed repetition penalty on top of
    temperature + top-k, matching upstream ``MossTTSRealtimeInference.generate``.

The regular body still supports re-prefill for the other model families.  This
decoder uses its incremental path: the K/V cache is created for one frame and
discarded before the next frame, so each codebook step computes only the new
position while preserving the causal attention result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSLocalTransformerConfig,
)

HistoryPerCodebook = list[list[int]] | list[list[list[int]]]


class MossTTSRealtimeLocalTransformer(nn.Module):
    """Per-frame depth transformer. Mirrors upstream ``...LocalTransformer``.

    State per audio frame:
      - The first input token uses ``backbone_last_hidden_state`` as the
        embedding (codebook 0's "input" is the backbone hidden, not a token).
      - Subsequent tokens (1..rvq-1) embed via ``model.codec_embedding[idx-1]``.

    Outputs:
      - One logit row per codebook position, projected through
        ``local_lm_heads[codebook_idx]`` (passed in by the talker).
    """

    def __init__(self, cfg: MossTTSLocalTransformerConfig) -> None:
        super().__init__()
        self.config = cfg
        # Shared body. Its codec_embedding holds rvq-1 embeddings -- upstream's
        # embed_tokens for codebooks 1..rvq-1 (codebook 0 uses the backbone
        # hidden). embedding_dim == hidden_size, so no projection is needed.
        self.model = CodePredictorBaseModel(
            cfg,
            embedding_dim=cfg.hidden_size,
            use_parallel_embedding=False,
            prefix="model",
        )

    @torch.no_grad()
    def generate_frame(
        self,
        backbone_last_hidden: torch.Tensor,  # (B, H)
        lm_heads: nn.ModuleList,  # ModuleList of rvq Linear(H -> audio_vocab_size)
        *,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        history_per_codebook: HistoryPerCodebook | None = None,
    ) -> torch.Tensor:
        """Generate one audio frame (rvq codebook tokens) for batch B.

        Returns a ``(B, rvq)`` LongTensor.

        For a single request, ``history_per_codebook[i]`` is a list of
        recently-emitted token ids for codebook ``i``. For a batch, use
        ``history_per_codebook[batch][codebook]``. When
        ``repetition_penalty != 1.0`` those tokens get their logits scaled
        down (mirrors upstream's rep-penalty behaviour).
        """
        device = backbone_last_hidden.device
        B = backbone_last_hidden.shape[0]
        rvq = self.config.rvq
        hidden_size = self.config.hidden_size

        codec_embeds = self.model.codec_embedding

        codes = backbone_last_hidden.new_zeros((B, rvq), dtype=torch.long)

        histories: list[list[list[int]]] | None
        if history_per_codebook is None:
            histories = None
        elif B == 1 and (
            not history_per_codebook
            or not history_per_codebook[0]
            or not isinstance(history_per_codebook[0][0], list)
        ):
            # Preserve the original single-request calling convention.
            histories = [history_per_codebook]  # type: ignore[list-item]
        else:
            histories = history_per_codebook  # type: ignore[assignment]

        # The cache is scoped to this frame.  Position zero is the talker
        # hidden state; every later position is the embedding of the previous
        # codebook token.
        frame_embed = backbone_last_hidden.to(dtype=next(self.model.parameters()).dtype).unsqueeze(1)
        past_key_values = None
        for step in range(rvq):
            pos_ids = torch.full((B, 1), step, dtype=torch.long, device=device)
            hidden, past_key_values = self.model.forward_incremental(
                frame_embed,
                pos_ids,
                past_key_values=past_key_values,
            )
            logits = lm_heads[step](hidden[:, -1, :]).float()

            if repetition_penalty != 1.0 and histories is not None:
                apply_repetition_penalty(logits, histories, step, repetition_penalty)

            codes[:, step] = _sample_token(logits, temperature, top_k, top_p, do_sample)

            if step + 1 < rvq:
                frame_embed = codec_embeds[step](codes[:, step].view(B, 1)).view(B, 1, hidden_size)

        return codes


def apply_repetition_penalty(
    logits: torch.Tensor,
    histories: list[list[list[int]]],
    codebook: int,
    repetition_penalty: float,
) -> None:
    """Apply one history penalty mask across a batched codebook decode."""
    batch_size, vocab_size = logits.shape
    token_lists = [
        histories[index][codebook] if codebook < len(histories[index]) else []
        for index in range(batch_size)
    ]
    max_history = max((len(tokens) for tokens in token_lists), default=0)
    if max_history == 0:
        return

    # Use vocab_size as a sentinel so padded history entries cannot collide
    # with a real token id, including token 0.
    history_ids = torch.full(
        (batch_size, max_history),
        vocab_size,
        dtype=torch.long,
        device=logits.device,
    )
    for index, tokens in enumerate(token_lists):
        if tokens:
            history_ids[index, : len(tokens)] = torch.as_tensor(
                tokens,
                dtype=torch.long,
                device=logits.device,
            )

    seen = torch.zeros(
        (batch_size, vocab_size + 1),
        dtype=torch.bool,
        device=logits.device,
    )
    seen.scatter_(1, history_ids, True)
    seen = seen[:, :vocab_size]
    penalized = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
    logits.copy_(torch.where(seen, penalized, logits))


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Top-k + top-p sampling (matches upstream's ``sample_token`` for the
    inference branch).
    """
    if not do_sample or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        top_vals, _ = torch.topk(logits, top_k, dim=-1)
        thresh = top_vals[..., -1:].expand_as(logits)
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Drop tail beyond top_p (keep at least one token).
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1, generator=generator).reshape(probs.shape[:-1])
    return sampled


__all__ = ["MossTTSRealtimeLocalTransformer"]
