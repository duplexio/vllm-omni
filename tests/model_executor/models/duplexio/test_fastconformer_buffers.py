# SPDX-License-Identifier: Apache-2.0
"""The exported ASR frontend tensors are loadable, persistent model state."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.duplexio.fastconformer import FastConformerRNNT


def test_exported_frontend_buffers_load_through_native_loader() -> None:
    # Only serialization is under test; no encoder/processor arithmetic is mocked.
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(encoder_config=SimpleNamespace(hidden_size=4), blank_token_id=0)
    processor = SimpleNamespace(
        set_num_lookahead_tokens=lambda _: None,
        feature_extractor=SimpleNamespace(
            hop_length=160,
            n_fft=512,
            win_length=400,
            preemphasis=0.97,
            mel_filters=torch.zeros(128, 257),
        ),
    )
    encoder = FastConformerRNNT(model, processor)
    filters = torch.randn_like(encoder.mel_filters)
    window = torch.randn_like(encoder.stft_window)
    loaded = AutoWeightsLoader(encoder).load_weights(
        [
            ("mel_filters", filters),
            ("stft_window", window),
        ]
    )
    assert loaded == {"mel_filters", "stft_window"}
    torch.testing.assert_close(encoder.state_dict()["mel_filters"], filters)
    torch.testing.assert_close(encoder.state_dict()["stft_window"], window)
