"""Audio embeddings must not depend on prefill/decode partition sizes."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.audio_adapters import AudioInputAdapter

pytest.importorskip("duplexio")
from duplexio.modules.audio_adapter import AudioInputAdapter as TrainingAdapter
from duplexio.modules.audio_adapter import audio_adapter_hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_adapter_normalization_is_rowwise() -> None:
    torch.manual_seed(52)
    gate = torch.randn(736, 2560, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = audio_adapter_hidden(gate)
        actual = torch.cat([audio_adapter_hidden(g) for g in gate.split(1)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dim", [32, 1024])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("hidden_dim", [512, 1280])
@torch.inference_mode()
def test_audio_adapter_partition_invariance(
    input_dim: int,
    compiled: bool,
    hidden_dim: int,
) -> None:
    torch.manual_seed(52)
    # Both cells now use the same adapter: 32 is the agent's codec width, 1024
    # the user encoder's.
    training = TrainingAdapter(input_dim, hidden_dim, 2560, 0.02).cuda()
    native = AudioInputAdapter(input_dim, hidden_dim, 2560).cuda().bfloat16()
    native.load_state_dict(training.state_dict())
    if compiled:
        training.compile(dynamic=True)
    features = torch.randn(736, input_dim, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = training(features)
        actual = torch.cat(
            [
                native(chunk)
                for chunk in features.split([256, 256, 212, *([1] * 12)])
            ]
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
