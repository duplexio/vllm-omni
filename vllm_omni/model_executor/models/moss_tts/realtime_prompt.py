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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_omni.model_executor.models.moss_tts.session import MossTTSRealtimeSegment


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


def _assistant_prefix_grid(processor: Any, *, after_user: bool) -> np.ndarray:
    prefix_text = (
        "<|im_end|>\n<|im_start|>assistant\n"
        if after_user
        else "<|im_start|>assistant\n"
    )
    prefix_ids = _encode_without_special_tokens(processor.tokenizer, prefix_text)
    prefix = _audio_grid(processor, len(prefix_ids))
    prefix[:, 0] = prefix_ids
    return prefix


def _assistant_history_grid(
    processor: Any,
    *,
    text: str,
    audio_codes: torch.Tensor,
) -> np.ndarray:
    """Reconstruct the input rows cached by upstream after one assistant turn.

    The Realtime reference keeps the assistant model cache between turns. Its
    generated audio is not itself an input row until the next model step, so
    each emitted frame is paired with the next text token (or ``text_pad``)
    here. The EOS frame is intentionally absent from ``audio_codes``.
    """
    text_ids = _encode_without_special_tokens(processor.tokenizer, text)
    prefix_len = min(len(text_ids), int(processor.delay_tokens_len))
    prefix = _audio_grid(processor, prefix_len)
    prefix[:, 0] = text_ids[:prefix_len]
    if prefix_len:
        prefix[-1, 1] = int(processor.audio_bos_token)

    codes = _as_audio_tokens(audio_codes, int(processor.channels))
    assert codes is not None
    continuation = _audio_grid(processor, codes.shape[0])
    text_pad_id = getattr(processor, "text_pad_token_id", None)
    if text_pad_id is None:
        text_pad_id = processor.tokenizer.convert_tokens_to_ids("<|text_pad|>")
    text_pad_id = int(text_pad_id)
    remaining = text_ids[prefix_len:]
    continuation[:, 0] = [
        remaining[index] if index < len(remaining) else text_pad_id
        for index in range(codes.shape[0])
    ]
    continuation[:, 1:] = codes
    return np.concatenate([prefix, continuation], axis=0)


def _history_grid(
    processor: Any,
    segments: Sequence[MossTTSRealtimeSegment],
) -> np.ndarray:
    grids: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        if segment.role == "user":
            user_codes = _as_audio_tokens(segment.codes, int(processor.channels))
            assert user_codes is not None
            grids.append(
                np.asarray(
                    processor.make_user_prompt(segment.text, user_codes),
                    dtype=np.int64,
                )
            )
        elif segment.role == "assistant":
            if index == 0 or segments[index - 1].role == "assistant":
                grids.append(_assistant_prefix_grid(processor, after_user=index > 0))
            grids.append(
                _assistant_history_grid(
                    processor,
                    text=segment.text,
                    audio_codes=segment.codes,
                )
            )
        else:
            raise ValueError(f"Unknown MOSS Realtime history role: {segment.role!r}")
    if not grids:
        return np.empty((0, int(processor.channels) + 1), dtype=np.int64)
    return np.concatenate(grids, axis=0)


def build_realtime_prompt(
    processor: Any,
    *,
    text: str,
    reference_codes: torch.Tensor,
    history_segments: Sequence[MossTTSRealtimeSegment] = (),
    prefill_text_tokens: int = 12,
) -> MossTTSRealtimePrompt:
    """Build the exact mixed text/audio grid expected by Realtime.

    ``history_segments`` contains completed turns from the assistant session.
    The upstream realtime implementation keeps those rows in its KV cache; a
    stateless vLLM request must present the same rows explicitly.
    """
    channels = int(processor.channels)
    reference_tokens = _as_audio_tokens(reference_codes, channels)
    assert reference_tokens is not None

    system_grid = processor.make_ensemble(prompt_audio_tokens=reference_tokens)
    grids = [np.asarray(system_grid, dtype=np.int64)]

    history = _history_grid(processor, history_segments)
    if history.shape[0]:
        grids.append(history)
    if not history_segments or history_segments[-1].role != "user":
        grids.append(
            _assistant_prefix_grid(
                processor,
                after_user=bool(history_segments),
            )
        )

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
