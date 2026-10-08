"""Tests for batched MOSS realtime depth decoding."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
    MossRealtimeFrameGraphs,
    MossTTSRealtimeLocalTransformer,
    apply_repetition_penalty,
)


class _Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codec_embedding = nn.ModuleList([nn.Embedding(8, 3)])

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        del position_ids
        return inputs_embeds

    def forward_incremental(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        past_key_values: object = None,
    ) -> tuple[torch.Tensor, tuple]:
        del position_ids, past_key_values
        return inputs_embeds, ()


def make_fake_transformer() -> MossTTSRealtimeLocalTransformer:
    model = MossTTSRealtimeLocalTransformer.__new__(MossTTSRealtimeLocalTransformer)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(rvq=2, hidden_size=3)
    model.model = _Body()
    return model


def make_heads() -> nn.ModuleList:
    torch.manual_seed(7)
    heads = nn.ModuleList([nn.Linear(3, 8, bias=False), nn.Linear(3, 8, bias=False)])
    return heads


def test_batched_generation_matches_independent_greedy_rows() -> None:
    model = make_fake_transformer()
    heads = make_heads()
    hidden = torch.randn(2, 3)
    histories = torch.tensor([[[1, 2], [3, 8]], [[4, 8], [5, 6]]])

    batched = model.generate_frame(
        hidden,
        lm_heads=heads,
        do_sample=False,
        repetition_penalty=1.1,
        history_ids=histories,
    )
    independent = torch.cat(
        [
            model.generate_frame(
                hidden[index : index + 1],
                lm_heads=heads,
                do_sample=False,
                repetition_penalty=1.1,
                history_ids=histories[index:index + 1],
            )
            for index in range(hidden.shape[0])
        ],
        dim=0,
    )

    torch.testing.assert_close(batched, independent)


def test_repetition_penalty_is_applied_per_batch_row() -> None:
    logits = torch.tensor([[2.0, -2.0, 1.0], [2.0, -2.0, 1.0]])
    apply_repetition_penalty(logits, torch.tensor([[0, 0, 3], [1, 3, 3]]), 2.0)

    torch.testing.assert_close(logits, torch.tensor([[1.0, -2.0, 1.0], [2.0, -4.0, 1.0]]))


def test_incremental_code_predictor_matches_reprefill() -> None:
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_code_groups=3,
        vocab_size=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
    )
    torch.manual_seed(11)
    model = CodePredictorBaseModel(config)
    model.eval()
    inputs = torch.randn(2, 3, config.hidden_size)
    positions = torch.arange(3).unsqueeze(0).expand(2, -1)

    expected = model(inputs, positions)
    cache = None
    actual_steps = []
    for step in range(inputs.shape[1]):
        actual, cache = model.forward_incremental(
            inputs[:, step : step + 1],
            positions[:, step : step + 1],
            past_key_values=cache,
        )
        actual_steps.append(actual)

    actual = torch.cat(actual_steps, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a GPU")
def test_frame_graph_matches_eager_and_owns_returned_codes() -> None:
    from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import MossTTSLocalTransformerConfig

    cfg = MossTTSLocalTransformerConfig(
        hidden_size=32, intermediate_size=64, head_dim=8,
        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        rvq=4, audio_vocab_size=32,
    )
    torch.manual_seed(42)
    model = MossTTSRealtimeLocalTransformer(cfg).cuda().to(torch.bfloat16).eval()
    heads = nn.ModuleList([nn.Linear(32, 32, bias=False) for _ in range(4)]).cuda().to(torch.bfloat16)
    generate = partial(model.generate_frame, lm_heads=heads, do_sample=False, repetition_penalty=1.1)
    hidden = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
    history = torch.full((1, 4, 50), 32, device="cuda", dtype=torch.long)
    with torch.inference_mode():
        graph = MossRealtimeFrameGraphs(generate, hidden, history, 4)
        for batch in (1, 3, 4, 2):
            inputs = torch.randn(batch, 32, device="cuda", dtype=torch.bfloat16)
            histories = torch.randint(0, 33, (batch, 4, 50), device="cuda")
            expected = generate(inputs, histories)
            actual = graph(inputs, histories)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            saved = actual.clone()
            graph(inputs + 1, histories)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a GPU")
def test_frame_graph_sampling_advances_rng() -> None:
    model = make_fake_transformer().cuda()
    heads = make_heads().cuda()
    with torch.no_grad():
        for head in heads:
            head.weight.zero_()
    generate = partial(model.generate_frame, lm_heads=heads, top_k=0, top_p=1.0)
    hidden = torch.ones(1, 3, device="cuda")
    history = torch.full((1, 2, 50), 8, device="cuda", dtype=torch.long)
    graph = MossRealtimeFrameGraphs(generate, hidden, history, 4)
    draws = torch.stack([graph(hidden, history) for _ in range(8)])
    assert torch.unique(draws.reshape(-1)).numel() > 1
