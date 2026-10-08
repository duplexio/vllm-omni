# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The ASR frontend tensors are loadable, persistent model state on the model's device."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

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


def test_frontend_buffers_load_through_the_model_loader() -> None:
    # Only serialization is under test; no encoder/processor arithmetic is mocked.
    encoder = FastConformerEncoder(*encoder_parts())
    filters = torch.randn_like(encoder.mel_filters)
    window = torch.randn_like(encoder.stft_window)
    loaded = AutoWeightsLoader(encoder).load_weights([("mel_filters", filters), ("stft_window", window)])
    assert loaded == {"mel_filters", "stft_window"}
    torch.testing.assert_close(encoder.state_dict()["mel_filters"], filters)
    torch.testing.assert_close(encoder.state_dict()["stft_window"], window)


def test_frontend_buffers_live_on_the_device_the_model_is_built_on() -> None:
    # vLLM builds models under a device context and never moves them afterwards.
    with torch.device("meta"):
        encoder = FastConformerEncoder(*encoder_parts())
    assert encoder.mel_filters.device.type == encoder.stft_window.device.type == "meta"
