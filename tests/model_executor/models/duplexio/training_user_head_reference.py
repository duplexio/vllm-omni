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
    model.linear_ce_options = nn.LinearCrossEntropyOptions(
        batch_chunk_size=16, acc_policy="accurate", acc_dtype=torch.float32,
    )
    model.cuda()
    # The pretrained vocabulary weights have the backbone's dtype in training;
    # the full-frame projection and emit head remain FP32 under autocast.
    model.llm.to(dtype=dtype)
    hidden = torch.randn(9, 6, 32, device="cuda", dtype=dtype)
    user_ids = torch.tensor([0, 0, 3, 0, 0, 4, 0, 0, 5], device="cuda")
    token_rows = torch.tensor([2, 5, 8], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        projected = model.user_token_projection(hidden.flatten(-2))
        emit_logits = model.user_emit_head(hidden.flatten(-2)).squeeze(-1)
        content_loss = model.user_content_loss(hidden, user_ids, token_rows)
        emit_loss, _, _ = model.user_emit_loss(
            hidden, user_ids, token_rows, torch.tensor([1, 3, 4, 6, 7], device="cuda"),
        )
    with torch.no_grad():
        logits = nn.functional.linear(projected.float(), model.token_head.weight.float())
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "weights": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "hidden": hidden.cpu(),
        "logits": logits.float().cpu(),
        "emit_logits": emit_logits.float().cpu(),
        "user_ids": user_ids.cpu(),
        "token_rows": token_rows.cpu(),
        "losses": {"user_token_ce": content_loss.cpu(), "user_emit": emit_loss.cpu()},
    }


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [reference_case(dtype) for dtype in (torch.float32, torch.bfloat16)]
    torch.save(cases, sys.argv[1])
    print(f"Wrote {len(cases)} training user-head reference cases to {sys.argv[1]}", flush=True)
