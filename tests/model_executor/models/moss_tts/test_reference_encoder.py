# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.reference_encoder import (
    encode_concatenated_reference_codes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingProcessor:
    def __init__(self) -> None:
        self.encoded_wav: torch.Tensor | None = None

    def encode_audios_from_wav(
        self,
        wavs: list[torch.Tensor],
        sampling_rate: int,
        n_vq: int,
    ) -> list[torch.Tensor]:
        assert sampling_rate == 24_000
        assert n_vq == 2
        self.encoded_wav = wavs[0]
        return [torch.tensor([[1, 2], [3, 4]], dtype=torch.long)]


def test_prompt_audio_is_encoded_after_waveform_concatenation() -> None:
    processor = RecordingProcessor()

    codes = encode_concatenated_reference_codes(
        processor,
        [([[0.1, 0.2]], 24_000), ([[0.3, 0.4]], 24_000)],
        sr_target=24_000,
        n_vq=2,
    )

    assert processor.encoded_wav is not None
    torch.testing.assert_close(
        processor.encoded_wav,
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    torch.testing.assert_close(codes, torch.tensor([[1, 2], [3, 4]]))
