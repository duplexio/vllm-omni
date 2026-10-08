# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""User heads reproduce the training repository's logits and losses on its reference cases."""

import os

import pytest
import torch
from torch.nn import functional as F

from tests.model_executor.models.duplexio.test_user_heads import head_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_training_user_losses_and_next_frame_targets_match_serving():
    reference_path = os.environ.get("DUPLEXIO_USER_HEAD_REFERENCE")
    if reference_path is None:
        pytest.skip("Write the reference cases with training_user_head_reference.py first")
    torch.backends.cuda.matmul.allow_tf32 = False
    for case in torch.load(reference_path, weights_only=True):
        model = head_model("cuda")
        dtype = getattr(torch, case["dtype"])
        model.lm_head.to(dtype=dtype)
        weights = [
            (name if name.startswith("user_") else "llm.base_model.lm_head.weight", tensor.cuda())
            for name, tensor in case["weights"].items()
        ]
        model.load_weights(weights)
        hidden = case["hidden"].cuda()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            logits, emissions = model.project_text(hidden)
        torch.testing.assert_close(logits[:, 2].cpu(), case["logits"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(emissions[:, 2].float().cpu(), case["emit_logits"], rtol=1e-5, atol=1e-6)
        target_rows = case["token_rows"].cuda()
        ids = case["user_ids"].cuda()
        # Both supervised and replayed user content use the previous frame.
        ce = F.cross_entropy(logits[target_rows - 1, 2, 1:], ids[target_rows] - 1)
        labels = ids[1:] != 0
        bce = F.binary_cross_entropy_with_logits(emissions[:-1, 2].float(), labels.float())
        torch.testing.assert_close(ce.cpu(), case["losses"]["user_token_ce"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bce.cpu(), case["losses"]["user_emit"], rtol=1e-5, atol=1e-6)
