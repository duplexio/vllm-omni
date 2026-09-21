"""Compiled serving SwiGLU preserves standard BF16 linear semantics."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIOQwenMLP


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_mlp_matches_standard_linear_across_frame_batches() -> None:
    torch.manual_seed(314)
    model = DuplexIOQwenMLP.__new__(DuplexIOQwenMLP)
    nn.Module.__init__(model)
    model.gate_up_proj = nn.Linear(2560, 2 * 9216, bias=False, device="cuda", dtype=torch.bfloat16)
    model.down_proj = nn.Linear(9216, 2560, bias=False, device="cuda", dtype=torch.bfloat16)
    model.down_proj.tp_size = 1
    for rows in (6, 12, 768, 6):
        hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
        gate, up = F.linear(hidden, model.gate_up_proj.weight).chunk(2, dim=-1)
        expected = F.linear(F.silu(gate) * up, model.down_proj.weight)
        torch.testing.assert_close(model(hidden), expected, rtol=0, atol=0)
