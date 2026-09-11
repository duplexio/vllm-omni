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
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    from duplexio.modules.qwen3_5_mlp import QuackQwen3_5MLP

    torch.manual_seed(42)
    config = SimpleNamespace(hidden_size=2560, intermediate_size=9728, hidden_act="silu")
    model = QuackQwen3_5MLP(Qwen3_5MLP(config, config.intermediate_size)).cuda().bfloat16()
    hidden = torch.randn(4422, 2560, device="cuda", dtype=torch.bfloat16)
    expected = model(hidden)
    actual = torch.cat([model(chunk) for chunk in hidden.split(1536)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_attention_qkv_fusion_preserves_projection_values() -> None:
    from vllm_omni.model_executor.models.duplexio.numerics import fixed_linear

    from duplexio.modules.fixed_linear import fixed_linear as training_linear

    torch.manual_seed(60)
    hidden = torch.randn(1536, 2560, device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn(width, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
        for width in (8192, 1024, 1024)
    ]
    expected = torch.cat([training_linear(hidden, weight) for weight in weights], -1)
    fused_weight = torch.cat(weights)
    for rows in (6, 768, 1536):
        actual = fixed_linear(hidden[:rows], fused_weight)
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
def test_self_attention_merge_matches_training_within_bf16_rounding() -> None:
    from duplexio.modules.stream_attention import merge_self_attention as training_merge

    from vllm_omni.model_executor.models.duplexio.stream_attention import merge_self_attention

    torch.manual_seed(716)
    tokens, dim = 4416, 256
    query = torch.randn(tokens, 16, dim, device="cuda", dtype=torch.bfloat16)
    self_key = torch.randn(tokens, 4, dim, device="cuda", dtype=torch.bfloat16)
    self_value = torch.randn_like(self_key)
    # Flex hands the history back transposed, so keep it non-contiguous here too.
    history = torch.randn(1, 16, tokens, dim, device="cuda", dtype=torch.bfloat16)[0].transpose(0, 1)
    lse = torch.randn(tokens, 16, device="cuda")
    expected = training_merge(
        query.transpose(0, 1)[None].contiguous(),
        self_key.transpose(0, 1)[None].contiguous(),
        self_value.transpose(0, 1)[None].contiguous(),
        history.transpose(0, 1)[None].contiguous(),
        lse.transpose(0, 1)[None].contiguous(),
        dim**-0.5,
    )[0].transpose(0, 1)
    for length in (1536, 6):
        actual = merge_self_attention(
            query[:length],
            self_key[:length],
            self_value[:length],
            history[:length],
            lse[:length],
            dim**-0.5,
        )
        # Same algebra, different kernel. One bf16 mantissa step is 4e-3
        # relative, and differently ordered rounding lands up to two apart.
        torch.testing.assert_close(actual, expected[:length], rtol=1e-2, atol=1e-2)


@torch.inference_mode()
def test_vocab_projection_matches_learner_across_batch_sizes() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.model_executor.models.duplexio.modeling_duplexio import DuplexIOLogitsProcessor

    from duplexio.losses.fused_linear_kl import _linear_bf16_out

    torch.manual_seed(912)
    hidden = torch.randn(32, 2560, device="cuda", dtype=torch.bfloat16)
    head = torch.nn.Linear(2560, 248320, bias=False, device="cuda", dtype=torch.bfloat16)
    head.weight.normal_(0, 0.02)
    expected = _linear_bf16_out(
        hidden, head.weight, torch.empty(32, 248320, device="cuda", dtype=torch.bfloat16)
    )
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
