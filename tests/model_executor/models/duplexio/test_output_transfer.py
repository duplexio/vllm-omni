# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import to_host_async


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_host_async_copies_outputs_without_blocking():
    device = torch.device("cuda")
    batch = torch.randn(3, 4, device=device)
    host_flag = torch.tensor([True])

    def outputs():
        return {
            "audio": [batch[0], torch.empty(0), batch[2, 1:3]],
            "chunk": {
                "ids": [torch.tensor([5], device=device), torch.tensor([7, 8], device=device), torch.tensor([9])],
                "emit": [torch.tensor([True], device=device), host_flag, torch.tensor([False], device=device)],
            },
        }

    expected = {
        "audio": [value.cpu() for value in outputs()["audio"]],
        "chunk": {name: [value.cpu() for value in values] for name, values in outputs()["chunk"].items()},
    }
    to_host_async(outputs()).wait()  # warm the pinned host allocator
    sources = outputs()
    torch.cuda.set_sync_debug_mode("error")
    try:
        pending = to_host_async(sources)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    host = pending.wait()

    assert host["chunk"]["emit"][1] is host_flag
    assert pending.wait() is host
    for actual, reference in zip(
        [*host["audio"], *(value for values in host["chunk"].values() for value in values)],
        [*expected["audio"], *(value for values in expected["chunk"].values() for value in values)],
        strict=True,
    ):
        assert actual.device.type == "cpu"
        assert actual.dtype == reference.dtype
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("on_device", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compute_logits_forces_host_or_device_ids(on_device):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.duplexio.modeling_duplexio import DuplexIOForConditionalGeneration

    model = object.__new__(DuplexIOForConditionalGeneration)
    object.__setattr__(model, "text_config", SimpleNamespace(vocab_size=11))
    object.__setattr__(model, "silence_token_id", 3)
    hidden = torch.zeros(3, 2, device="cuda")
    forced = [4, 9, 3]
    object.__setattr__(
        model,
        "_forced_next_token_ids",
        torch.tensor(forced, device="cuda") if on_device else forced,
    )

    torch.cuda.set_sync_debug_mode("error")  # the batch queue overlaps only if the host never waits here
    try:
        logits = DuplexIOForConditionalGeneration.compute_logits(model, hidden)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert model._forced_next_token_ids is None
    assert logits.argmax(-1).tolist() == forced
    assert torch.isinf(logits).sum().item() == 3 * 10
