# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeCodec(nn.Module):
    downsample_rate = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(codebook_size=1024)

    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        num_quantizers: int,
    ) -> SimpleNamespace:
        frames = codes_list[0].shape[1]
        samples = torch.arange(frames * self.downsample_rate, dtype=torch.float32)
        return SimpleNamespace(
            audio=samples.reshape(1, 1, -1),
            audio_lengths=torch.tensor([samples.numel()]),
        )


def bare_decoder() -> MossTTSCodecDecoder:
    decoder = MossTTSCodecDecoder.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder._n_vq = 2
    decoder._n_channels = 1
    decoder._sr_tensor = torch.tensor(24_000, dtype=torch.int32)
    decoder._codec = FakeCodec()
    decoder._cuda_graph_wrapper = None
    decoder._stream_session = None
    decoder._stream_req_slots = {}
    decoder._stream_pending_codes = {}
    decoder._stream_starved_reqs = set()
    return decoder


def test_codec_trims_context_audio_and_returns_only_new_codes() -> None:
    codes = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [10, 20], [11, 21]],
        dtype=torch.long,
    )
    input_ids = codes.transpose(0, 1).contiguous().reshape(-1)

    output = bare_decoder()(
        input_ids=input_ids,
        runtime_additional_information=[{"meta": {"left_context_size": 3}}],
        seq_token_counts=[input_ids.numel()],
    )

    assert output.multimodal_outputs["model_outputs"][0].tolist() == [6.0, 7.0, 8.0, 9.0]
    torch.testing.assert_close(
        output.multimodal_outputs["audio_codes"][0],
        codes[3:],
    )
