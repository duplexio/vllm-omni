"""Packed position embedding preserves numerical results and graph output ownership."""

import pytest
import torch

from tests.model_executor.models.duplexio.test_bulk_prefill import model_fixture
from vllm_omni.model_executor.models.duplexio.flat_plan import Embed
from vllm_omni.model_executor.models.duplexio.frame_layout import AGENT, QUERY_SLOT_OFFSET, TOOL_CALL
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import FrameInputGraph
from vllm_omni.model_executor.models.duplexio.pocket_mimi import LATENT_DIM


def live_frames(size: int, device: str) -> torch.Tensor:
    """One live frame per request: a user token, an agent query, a tool token, then the frame's audio."""
    rows = []
    for request in range(size):
        rows += [
            (Embed.TOKEN, 3 + request, 1, 0, 0),
            (Embed.CELL, QUERY_SLOT_OFFSET + AGENT, 0, 0, 0),
            (Embed.TOKEN, 9, TOOL_CALL, 0, 0),
            (Embed.USER_AUDIO, 0, 0, request + 1, 0),
            (Embed.AGENT_AUDIO, 0, 0, 0, request + 1),
        ]
    return torch.tensor(rows, dtype=torch.long, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_frame_embedding_graph_matches_independent_requests_and_weight_refresh() -> None:
    model = model_fixture().cuda()
    model.vllm_config.model_config.dtype = torch.bfloat16
    model.project_positions = torch.compile(model.project_positions, fullgraph=True, dynamic=True)
    for size in (1, 3, 15):
        packed = live_frames(size, "cuda")
        user = torch.randn(size + 1, 8, device="cuda")
        agent = torch.randn(size + 1, LATENT_DIM, device="cuda")
        user[0] = agent[0] = 0
        references = []
        for request in range(size):
            single = packed[5 * request : 5 * request + 5].clone()
            single[3, 3] = single[4, 4] = 1
            sources = (user[[0, request + 1]], agent[[0, request + 1]])
            references.append(model.project_positions(single, *sources)[0])
        inputs = (packed, user, agent)
        graph = FrameInputGraph(model.project_positions, inputs)
        (actual,) = graph(inputs)
        torch.testing.assert_close(actual, torch.cat(references), rtol=0.02, atol=0.02)
        saved = actual.clone()
        packed[:, 1] = torch.where(packed[:, 0] == Embed.TOKEN, 21, packed[:, 1])
        user.mul_(2)
        model.user_audio_input_adapter.output_proj.weight.mul_(0.75)
        (refreshed,) = graph(inputs)
        (eager,) = model.project_positions(*inputs)
        torch.testing.assert_close(refreshed, eager, rtol=0, atol=0)
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
