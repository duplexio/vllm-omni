"""Input graphs must replay new data exactly for both audio representations."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.duplexio.audio_adapters import AgentAudioInputAdapter, AudioInputAdapter
from vllm_omni.model_executor.models.duplexio.audio_input_graph import AudioInputGraph
from vllm_omni.model_executor.models.duplexio.audio_representation import MimiEmbedding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_audio_input_graph_replays_changed_inputs(batch: int, quantized: bool) -> None:
    torch.manual_seed(71)
    user = AudioInputAdapter(1024, 512, 2560).cuda().bfloat16()
    embedding = MimiEmbedding(8, 2048, 32).cuda().bfloat16() if quantized else nn.Identity()
    agent = AgentAudioInputAdapter(32, 2048, 512, 2560).cuda().bfloat16()
    features = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    speakers = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    codes = (
        torch.randint(0, 2048, (batch, 8), device="cuda")
        if quantized else torch.randn(batch, 32, device="cuda")
    )
    graph = AudioInputGraph(user, embedding, agent, features, codes, speakers, torch.bfloat16)
    for _ in range(3):
        features.normal_()
        speakers.normal_()
        if quantized:
            codes.random_(0, 2048)
        else:
            codes.normal_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = user(features), agent(embedding(codes), speakers)
        actual = graph(features, codes, speakers)
        for output, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
