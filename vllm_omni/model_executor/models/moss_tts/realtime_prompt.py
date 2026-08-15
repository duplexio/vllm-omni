# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt construction for the MOSS-TTS-Realtime talker.

MOSS-TTS-Realtime does not use the delay model's ``AutoProcessor`` format.
The talker consumes a 17-column grid: column zero is the language-model token
and columns one through sixteen are the RVQ codebooks.  Keeping construction
here makes the serving layer independent of the remote-code processor's
details and gives us a small, CPU-testable contract for turnwise requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class MossTTSRealtimePrompt:
    """The vLLM prompt and per-step text continuation for one turn."""

    text_ids: list[int]
    audio_codes: torch.Tensor
    remaining_text_ids: list[int]


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text without adding a model-level BOS/EOS token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("MOSS-TTS-Realtime text is empty after tokenization")
    return [int(token) for token in ids]


def _audio_grid(processor: Any, token_rows: int) -> np.ndarray:
    return np.full(
        (token_rows, int(processor.channels) + 1),
        int(processor.audio_channel_pad),
        dtype=np.int64,
    )


def _as_audio_tokens(codes: torch.Tensor | None, channels: int) -> np.ndarray | None:
    if codes is None:
        return None
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"audio codes must be a tensor, got {type(codes).__name__}")
    if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != channels:
        raise ValueError(
            "audio codes must have shape (frames, codebooks) with "
            f"codebooks={channels}, got {tuple(codes.shape)}"
        )
    return codes.detach().to("cpu", torch.long).numpy()


def build_realtime_prompt(
    processor: Any,
    *,
    text: str,
    reference_codes: torch.Tensor,
    previous_text: str | None = None,
    previous_codes: torch.Tensor | None = None,
    prefill_text_tokens: int = 12,
) -> MossTTSRealtimePrompt:
    """Build the exact mixed text/audio grid expected by Realtime.

    ``previous_text`` and ``previous_codes`` describe at most the immediately
    preceding turn.  Realtime's reference implementation uses
    ``make_user_prompt`` for this context; retaining only one turn keeps the
    prompt bounded and matches the live turnwise recipe.  The current turn is
    always synthesized as an assistant section, with ``reference_codes``
    selecting the voice for that turn.
    """
    channels = int(processor.channels)
    reference_tokens = _as_audio_tokens(reference_codes, channels)
    assert reference_tokens is not None

    system_grid = processor.make_ensemble(prompt_audio_tokens=reference_tokens)
    grids = [np.asarray(system_grid, dtype=np.int64)]

    if (previous_text is None) != (previous_codes is None):
        raise ValueError("previous_text and previous_codes must be provided together")
    if previous_text is not None and previous_codes is not None:
        previous_tokens = _as_audio_tokens(previous_codes, channels)
        assert previous_tokens is not None
        grids.append(
            np.asarray(
                processor.make_user_prompt(previous_text, previous_tokens),
                dtype=np.int64,
            )
        )
    else:
        assistant_ids = _encode_without_special_tokens(
            processor.tokenizer,
            "<|im_start|>assistant\n",
        )
        assistant_grid = _audio_grid(processor, len(assistant_ids))
        assistant_grid[:, 0] = assistant_ids
        grids.append(assistant_grid)

    current_ids = _encode_without_special_tokens(processor.tokenizer, text)
    if prefill_text_tokens <= 0:
        raise ValueError(f"prefill_text_tokens must be positive, got {prefill_text_tokens}")
    current_prefill_len = min(len(current_ids), prefill_text_tokens)
    text_grid = _audio_grid(processor, current_prefill_len)
    text_grid[:, 0] = current_ids[:current_prefill_len]
    text_grid[-1, 1] = int(processor.audio_bos_token)
    grids.append(text_grid)

    grid = np.concatenate(grids, axis=0)
    return MossTTSRealtimePrompt(
        text_ids=grid[:, 0].tolist(),
        audio_codes=torch.from_numpy(grid[:, 1:].copy()).to(torch.long),
        remaining_text_ids=current_ids[current_prefill_len:],
    )


def prepare_realtime_reference_wav(
    wav_list: list[float],
    sr: int,
    *,
    target_sr: int = 24_000,
) -> torch.Tensor:
    """Convert a resolved reference clip to mono ``(1, samples)`` audio."""
    wav = torch.tensor(wav_list, dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(f"reference audio must be 1-D or 2-D, got {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def encode_realtime_reference(
    codec: Any,
    wav_list: list[float],
    sr: int,
    *,
    target_sr: int = 24_000,
    channels: int = 16,
) -> torch.Tensor:
    """Encode a resolved reference clip with the standalone MOSS codec."""
    wav = prepare_realtime_reference_wav(wav_list, sr, target_sr=target_sr)
    with torch.no_grad():
        encoded = codec.batch_encode([wav.squeeze(0)], num_quantizers=channels)
    lengths = encoded.audio_codes_lengths
    frame_count = int(lengths[0].item())
    codes = encoded.audio_codes[:channels, 0, :frame_count]
    if codes.ndim != 2 or codes.shape[0] != channels or codes.shape[1] == 0:
        raise ValueError(f"codec returned invalid reference codes with shape {tuple(codes.shape)}")
    return codes.transpose(0, 1).contiguous().to(torch.long).cpu()


__all__ = [
    "MossTTSRealtimePrompt",
    "build_realtime_prompt",
    "encode_realtime_reference",
    "prepare_realtime_reference_wav",
]
