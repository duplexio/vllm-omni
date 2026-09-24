"""Match training's compiled Qwen MLP and BF16 compute-time normalization."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIORMSNorm

training = pytest.importorskip("duplexio.modules.qwen3_5_rmsnorm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("shape", [(6, 2560), (64, 256)])
@pytest.mark.parametrize("prenorm", [False, True])
@torch.inference_mode()
def test_norm_matches_training_compute_dtype(shape: tuple[int, int], prenorm: bool) -> None:
    torch.manual_seed(512)
    reference = training.Qwen3_5DirectRMSNorm(Qwen3_5RMSNorm(shape[-1], eps=1e-6)).cuda()
    reference.scale.normal_(1, 0.2)
    reference.bfloat16()
    native = DuplexIORMSNorm(shape[-1], eps=1e-6, dtype=torch.bfloat16).cuda()
    native.weight.copy_(reference.scale)
    assert native.weight.dtype == torch.bfloat16
    hidden = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(hidden) if prenorm else None
    torch.testing.assert_close(
        native(hidden, residual),
        reference(hidden, residual, prenorm=prenorm),
        rtol=1e-2,
        atol=1e-5,
    )


@torch.inference_mode()
def test_mlp_long_prefill_partition() -> None:
    from duplexio.modules.qwen3_5_mlp import Qwen3_5PackedMLP
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    torch.manual_seed(42)
    config = SimpleNamespace(hidden_size=2560, intermediate_size=9728, hidden_act="silu")
    model = Qwen3_5PackedMLP(Qwen3_5MLP(config, config.intermediate_size)).cuda().bfloat16()
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
def test_mlp_native_matches_training(rows: int) -> None:
    from duplexio.modules.qwen3_5_mlp import Qwen3_5PackedMLP
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIOQwenMLP

    torch.manual_seed(314)
    config = SimpleNamespace(hidden_size=2560, intermediate_size=9728, hidden_act="silu")
    reference = Qwen3_5PackedMLP(Qwen3_5MLP(config, config.intermediate_size)).cuda().bfloat16()
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
def test_vocab_projection_matches_learner_across_batch_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    from torch.nn.functional import linear
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    torch.manual_seed(912)
    hidden = torch.randn(32, 2560, device="cuda", dtype=torch.bfloat16)
    head = torch.nn.Linear(2560, 248320, bias=False, device="cuda", dtype=torch.bfloat16)
    head.quant_method = UnquantizedEmbeddingMethod()
    head.weight.normal_(0, 0.02)
    # The learner's accurate linear CE accumulates vocabulary logits in FP32.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    expected = linear(hidden.float(), head.weight.float())
    processor = SimpleNamespace(head_dtype=torch.float32)
    for rows in (1, 2, 17, 32):
        actual = LogitsProcessor._apply_head(processor, head, hidden[:rows], None)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected[:rows], rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_emit_projection_batch_matches_individual_frames() -> None:
    torch.manual_seed(425)
    rows = torch.randn(32, 6, 2560, device="cuda", dtype=torch.bfloat16)
    head = torch.nn.Linear(6 * 2560, 1, device="cuda", dtype=torch.bfloat16)
    expected = torch.stack([head(row.flatten()) for row in rows])
    for requests in (1, 2, 4, 8, 16, 32):
        torch.testing.assert_close(head(rows[:requests].flatten(1)), expected[:requests], rtol=0, atol=0)
