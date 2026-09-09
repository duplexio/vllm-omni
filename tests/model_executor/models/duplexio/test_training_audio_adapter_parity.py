"""Audio embeddings must not depend on prefill/decode partition sizes."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.audio_adapters import AgentAudioInputAdapter, AudioInputAdapter

pytest.importorskip("duplexio")
from duplexio.modules.audio_adapter import AgentAudioInputAdapter as TrainingAgentAdapter
from duplexio.modules.audio_adapter import AudioInputAdapter as TrainingUserAdapter
from duplexio.modules.audio_adapter import audio_adapter_hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_adapter_normalization_is_rowwise() -> None:
    torch.manual_seed(52)
    gate = torch.randn(736, 2560, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn_like(gate)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = audio_adapter_hidden(gate, modulation)
        actual = torch.cat([
            audio_adapter_hidden(g, m)
            for g, m in zip(gate.split(1), modulation.split(1), strict=True)
        ])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("agent", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("hidden_dim", [512, 1280])
@torch.inference_mode()
def test_audio_adapter_partition_invariance(agent: bool, compiled: bool, hidden_dim: int) -> None:
    torch.manual_seed(52)
    if agent:
        training = TrainingAgentAdapter(32, 2048, hidden_dim, 2560, 0.02).cuda()
        native = AgentAudioInputAdapter(32, 2048, hidden_dim, 2560).cuda().bfloat16()
        training.speaker_modulation.weight.normal_(std=0.02)
    else:
        training = TrainingUserAdapter(1024, hidden_dim, 2560, 0.02).cuda()
        native = AudioInputAdapter(1024, hidden_dim, 2560).cuda().bfloat16()
    native.load_state_dict(training.state_dict())
    if compiled:
        training.compile(dynamic=True)
    features = torch.randn(736, 32 if agent else 1024, device="cuda", dtype=torch.bfloat16)
    speakers = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = training(features, speakers.expand(736, -1)) if agent else training(features)
        actual = torch.cat(
            [
                native(chunk, speakers) if agent else native(chunk)
                for chunk in features.split([256, 256, 212, *([1] * 12)])
            ]
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
