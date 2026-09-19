"""Write user-head reference tensors using the training runtime (Torch 2.13)."""

import sys

import torch
from duplexio.models.duplexio import DuplexIOModel
from torch import nn


class Vocabulary(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 64, bias=False)

    def lm_head(self):
        return self.head


def reference_case(dtype):
    torch.manual_seed(912)
    model = DuplexIOModel.__new__(DuplexIOModel)
    nn.Module.__init__(model)
    model.llm = Vocabulary()
    model.user_token_projection = nn.Linear(6 * 32, 32)
    model.user_emit_head = nn.Linear(6 * 32, 1)
    model.silence_token_id = 0
    model.linear_ce_options = nn.LinearCrossEntropyOptions(batch_chunk_size=16)
    model.cuda()
    # The pretrained vocabulary weights have the backbone's dtype in training;
    # the full-frame projection and emit head remain FP32 under autocast.
    model.llm.to(dtype=dtype)
    hidden = torch.randn(9, 6, 32, device="cuda", dtype=dtype)
    user_ids = torch.tensor([0, 0, 3, 0, 0, 4, 0, 0, 5], device="cuda")
    token_rows = torch.tensor([2, 5, 8], device="cuda")
    offsets = torch.zeros(9, 5, device="cuda", dtype=torch.long)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        logits = model.token_head(model.user_token_projection(hidden.flatten(-2)))
        emit_logits = model.user_emit_head(hidden.flatten(-2)).squeeze(-1)
        losses, counts, _, _ = model.user_timing_losses(
            hidden, user_ids, token_rows, offsets, torch.arange(1, 9, device="cuda")
        )
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "weights": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "hidden": hidden.cpu(),
        "logits": logits.float().cpu(),
        "emit_logits": emit_logits.float().cpu(),
        "user_ids": user_ids.cpu(),
        "token_rows": token_rows.cpu(),
        "losses": {name: value.detach().cpu() for name, value in losses.items()},
        "counts": {name: value.cpu() for name, value in counts.items()},
    }


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [reference_case(dtype) for dtype in (torch.float32, torch.bfloat16)]
    torch.save(cases, sys.argv[1])
    print(f"Wrote {len(cases)} training user-head reference cases to {sys.argv[1]}", flush=True)
