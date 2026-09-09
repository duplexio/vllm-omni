"""Compiled audio graphs consume new noise and cannot overwrite prior outputs."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.flowmap import FlowMapSampler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("steps", [1, 2])
@torch.inference_mode()
def test_flowmap_graph_replays_owned_outputs(steps: int) -> None:
    torch.manual_seed(23)
    sampler = FlowMapSampler(
        4, 12, 32, 2, inference_steps=steps, sampling_temperature=0.3,
        use_cuda_graph=True,
    ).cuda()
    for batch in (1, 8, 3, 8):
        conditioning = torch.randn(batch, 12, device="cuda")
        noise = torch.randn(batch, 4, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = sampler.sample_function(conditioning, noise)
            actual = sampler.sample(conditioning, noise)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            saved = actual.clone()
            conditioning.normal_()
            noise.normal_()
            second = sampler.sample(conditioning, noise)
            torch.testing.assert_close(second, sampler.sample_function(conditioning, noise), rtol=0, atol=0)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            sampler.flow.final_layer.linear.weight.add_(0.01)
            updated = sampler.sample(conditioning, noise)
            torch.testing.assert_close(updated, sampler.sample_function(conditioning, noise), rtol=0, atol=0)
