# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The ASR frontend tensors come from the processor, on the model's device, outside the checkpoint."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_omni.model_executor.models.duplexio.fastconformer import FastConformerEncoder


def encoder_parts() -> tuple[nn.Module, SimpleNamespace]:
    """A stand-in encoder and the processor fields the frontend reads; the filters start on the CPU."""
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(hidden_size=4)
    processor = SimpleNamespace(
        set_num_lookahead_tokens=lambda _: None,
        feature_extractor=SimpleNamespace(
            hop_length=160,
            n_fft=512,
            win_length=400,
            preemphasis=0.97,
            mel_filters=torch.zeros(128, 257, device="cpu"),
        ),
    )
    return model, processor


def test_frontend_buffers_come_from_the_processor_not_the_checkpoint() -> None:
    model, processor = encoder_parts()
    encoder = FastConformerEncoder(model, processor)
    assert "mel_filters" not in encoder.state_dict() and "stft_window" not in encoder.state_dict()
    torch.testing.assert_close(encoder.mel_filters, processor.feature_extractor.mel_filters)
    torch.testing.assert_close(encoder.stft_window, torch.hann_window(400, periodic=False))


def test_frontend_buffers_live_on_the_device_the_model_is_built_on() -> None:
    # vLLM builds models under a device context and never moves them afterwards.
    with torch.device("meta"):
        encoder = FastConformerEncoder(*encoder_parts())
    assert encoder.mel_filters.device.type == encoder.stft_window.device.type == "meta"
