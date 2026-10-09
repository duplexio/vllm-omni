# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The compiled sampler captured as serving does: new noise, owned outputs, live weights."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.flowmap import FlowMap
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import FrameInputGraph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("steps", [1, 2])
@torch.inference_mode()
def test_flowmap_graph_replays_owned_outputs(steps: int) -> None:
    torch.manual_seed(23)
    flow = FlowMap(4, 32, 12, 2, inference_steps=steps).cuda()
    compiled = torch.compile(flow.sample, fullgraph=True, dynamic=True)
    graphs = {}
    for batch in (1, 8, 3, 8):
        conditioning = torch.randn(batch, 12, device="cuda")
        noise = torch.randn(batch, 4, device="cuda")
        temperature = torch.rand(batch, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if batch not in graphs:
                graphs[batch] = FrameInputGraph(
                    lambda *inputs: (compiled(*inputs),), (conditioning, noise, temperature)
                )

            def sample(conditioning, noise, graph=graphs[batch]):
                return graph((conditioning, noise, temperature))[0]

            expected = compiled(conditioning, noise, temperature)
            actual = sample(conditioning, noise)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            saved = actual.clone()
            conditioning.normal_()
            noise.normal_()
            second = sample(conditioning, noise)
            torch.testing.assert_close(second, compiled(conditioning, noise, temperature), rtol=0, atol=0)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            flow.final_layer.linear.weight.add_(0.01)
            updated = sample(conditioning, noise)
            torch.testing.assert_close(updated, compiled(conditioning, noise, temperature), rtol=0, atol=0)
