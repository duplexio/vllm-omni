"""Qwen's rotary phases and arithmetic must match at actual frame positions."""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIORotaryEmbedding
from vllm_omni.model_executor.models.duplexio.stream_attention import cached_rotary_pos_emb

pytest.importorskip("duplexio")
from duplexio.modules.qwen3_5_stream_delta import compiled_rotary_pos_emb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("frequency_dtype", (torch.float32, torch.bfloat16))
@torch.inference_mode()
def test_qwen_rotary_matches_training(frequency_dtype: torch.dtype) -> None:
    torch.manual_seed(59)
    config = Qwen3_5TextConfig(
        hidden_size=2560,
        head_dim=256,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=262144,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    training = Qwen3_5TextRotaryEmbedding(config).to(device="cuda", dtype=frequency_dtype)
    native = DuplexIORotaryEmbedding(64, 262144, torch.bfloat16).cuda()
    model = torch.nn.Module()
    model.add_module("rotary_emb", native)
    loaded = AutoWeightsLoader(model).load_weights([
        ("rotary_emb.inverse_frequencies", training.inv_freq),
        ("rotary_emb.attention_scaling", torch.tensor(training.attention_scaling, device="cuda", dtype=torch.float32)),
    ])
    assert loaded == {"rotary_emb.inverse_frequencies", "rotary_emb.attention_scaling"}
    positions = torch.tensor([0, 1, 3, 723, 724, 1023, 4096, 8192], device="cuda").repeat_interleave(6)
    q = torch.randn(positions.numel(), 16, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(positions.numel(), 4, 256, device="cuda", dtype=torch.bfloat16)
    cos, sin = training(q[None], positions[None])
    native_cos, native_sin = native.cos_sin_cache[positions].chunk(2, dim=-1)
    torch.testing.assert_close(native_cos, cos[0, :, :32], rtol=0, atol=0, msg="cosine cache")
    torch.testing.assert_close(native_sin, sin[0, :, :32], rtol=0, atol=0, msg="sine cache")
    expected = compiled_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    actual = cached_rotary_pos_emb(q, k, positions, native.cos_sin_cache)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference[0].transpose(0, 1), rtol=0, atol=0)
