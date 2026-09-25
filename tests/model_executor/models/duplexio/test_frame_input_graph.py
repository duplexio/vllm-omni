"""Packed frame projection preserves numerical results and graph output ownership."""

import pytest
import torch

from tests.model_executor.models.duplexio.test_bulk_prefill import model_fixture
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import FrameInputGraph, frame_inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_frame_projection_graph_matches_independent_requests_and_weight_refresh() -> None:
    model = model_fixture().cuda()
    model.llm.channel_emb = torch.nn.Parameter(model.llm.channel_emb.cuda())
    model.llm.base_model.model.cuda()
    model.vllm_config.model_config.dtype = torch.bfloat16
    model.frame_inputs = torch.compile(
        frame_inputs, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True},
    )
    for size in (1, 3, 15):
        ids = torch.randint(1, 20, (size, 4), device="cuda")
        ids[:, 0] = model.silence_token_id
        metadata = torch.tensor(
            [(index * 17, index * 9, True, False, index * 20) for index in range(size)],
            dtype=torch.int32, device="cuda",
        )
        user = torch.randn(size, 8, device="cuda")
        agent = torch.randint(0, 64, (size, 3), device="cuda")
        references = [model.project_frames(ids[i:i+1], metadata[i:i+1], user[i:i+1], agent[i:i+1])
                      for i in range(size)]
        preceding = ((ids != 1) & (ids != 2)).sum(1).cumsum(0)
        metadata[1:, 0] -= preceding[:-1]
        inputs = (ids, metadata, user, agent)
        graph = FrameInputGraph(model.project_frames, inputs)
        actual = graph(inputs)
        expected = [torch.cat(parts) for parts in zip(*references, strict=True)]
        for index, (output, reference) in enumerate(zip(actual, expected, strict=True)):
            torch.testing.assert_close(output, reference, rtol=0.02 if index == 0 else 0, atol=0.02 if index == 0 else 0)
        saved = tuple(value.clone() for value in actual)
        ids[:, 1] = 21
        user.mul_(2)
        model.user_audio_input_adapter.output_proj.weight.mul_(0.75)
        refreshed = graph(inputs)
        eager = model.project_frames(*inputs)
        for output, reference in zip(refreshed, eager, strict=True):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        for output, reference in zip(actual, saved, strict=True):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
