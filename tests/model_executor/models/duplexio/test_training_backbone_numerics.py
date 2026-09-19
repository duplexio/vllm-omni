"""Match training's normalization precision, including learned FP32 scales."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIORMSNorm

training = pytest.importorskip("duplexio.modules.qwen3_5_rmsnorm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Quack CUDA kernels")


@pytest.mark.parametrize("shape", [(6, 2560), (64, 256)])
@pytest.mark.parametrize("prenorm", [False, True])
@torch.inference_mode()
def test_norm_matches_training_with_fp32_scales(shape: tuple[int, int], prenorm: bool) -> None:
    torch.manual_seed(512)
    reference = training.QuackQwen3_5RMSNorm(Qwen3_5RMSNorm(shape[-1], eps=1e-6)).cuda()
    reference.scale.normal_(1, 0.2)
    native = DuplexIORMSNorm(shape[-1], eps=1e-6).cuda()
    native.weight.copy_(reference.scale)
    assert native.weight.dtype == torch.float32
    hidden = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(hidden) if prenorm else None
    torch.testing.assert_close(
        native(hidden, residual),
        reference(hidden, residual, prenorm=prenorm),
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_mlp_long_prefill_partition() -> None:
    from duplexio.modules.qwen3_5_mlp import QuackQwen3_5MLP
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    torch.manual_seed(42)
    config = SimpleNamespace(hidden_size=2560, intermediate_size=9728, hidden_act="silu")
    model = QuackQwen3_5MLP(Qwen3_5MLP(config, config.intermediate_size)).cuda().bfloat16()
    hidden = torch.randn(4422, 2560, device="cuda", dtype=torch.bfloat16)
    expected = model(hidden)
    actual = torch.cat([model(chunk) for chunk in hidden.split(1536)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_attention_qkv_fusion_preserves_projection_values() -> None:
    from torch.nn.functional import linear

    torch.manual_seed(60)
    hidden = torch.randn(1536, 2560, device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn(width, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
        for width in (8192, 1024, 1024)
    ]
    expected = torch.cat([linear(hidden, weight) for weight in weights], -1)
    fused_weight = torch.cat(weights)
    for rows in (6, 768, 1536):
        actual = linear(hidden[:rows], fused_weight)
        torch.testing.assert_close(actual, expected[:rows], rtol=0, atol=0)


@pytest.mark.parametrize("rows", [6, 96, 1536, 20622])
@torch.inference_mode()
def test_mlp_native_matches_training_without_runtime_tuning(rows: int) -> None:
    from duplexio.modules.qwen3_5_mlp import QuackQwen3_5MLP
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIOQwenMLP

    torch.manual_seed(314)
    config = SimpleNamespace(hidden_size=2560, intermediate_size=9728, hidden_act="silu")
    reference = QuackQwen3_5MLP(Qwen3_5MLP(config, config.intermediate_size)).cuda().bfloat16()
    native = DuplexIOQwenMLP.__new__(DuplexIOQwenMLP)
    torch.nn.Module.__init__(native)
    native.gate_up_proj = reference.gate_up_proj
    native.down_proj = reference.down_proj
    native.down_proj.tp_size = 1
    hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    expected = reference(hidden)
    torch.testing.assert_close(native(hidden), expected, rtol=0, atol=0)
    partitioned = torch.cat([native(chunk) for chunk in hidden.split(1536)])
    torch.testing.assert_close(partitioned, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_vocab_projection_matches_learner_across_batch_sizes() -> None:
    from torch.nn.functional import linear
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.model_executor.models.duplexio.modeling_duplexio import DuplexIOLogitsProcessor

    torch.manual_seed(912)
    hidden = torch.randn(32, 2560, device="cuda", dtype=torch.bfloat16)
    head = torch.nn.Linear(2560, 248320, bias=False, device="cuda", dtype=torch.bfloat16)
    head.weight.normal_(0, 0.02)
    expected = linear(hidden, head.weight)
    with set_current_vllm_config(VllmConfig()):
        processor = DuplexIOLogitsProcessor(248320)
    for rows in (1, 2, 17, 32):
        actual = processor._apply_head(head, hidden[:rows], None)
        torch.testing.assert_close(actual, expected[:rows], rtol=0, atol=0)


@torch.inference_mode()
def test_emit_projection_batch_matches_individual_frames() -> None:
    torch.manual_seed(425)
    rows = torch.randn(32, 6, 2560, device="cuda", dtype=torch.bfloat16)
    head = torch.nn.Linear(6 * 2560, 1, device="cuda", dtype=torch.bfloat16)
    expected = torch.stack([head(row.flatten()) for row in rows])
    for requests in (1, 2, 4, 8, 16, 32):
        torch.testing.assert_close(head(rows[:requests].flatten(1)), expected[:requests], rtol=0, atol=0)
