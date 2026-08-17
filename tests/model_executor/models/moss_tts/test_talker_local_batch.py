"""Tests for batched MOSS realtime depth decoding."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
    MossTTSRealtimeLocalTransformer,
    apply_repetition_penalty,
)
from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel


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
    histories = [[[1, 2], [3]], [[4], [5, 6]]]

    batched = model.generate_frame(
        hidden,
        heads,
        do_sample=False,
        repetition_penalty=1.1,
        history_per_codebook=histories,
    )
    independent = torch.cat(
        [
            model.generate_frame(
                hidden[index : index + 1],
                heads,
                do_sample=False,
                repetition_penalty=1.1,
                history_per_codebook=histories[index],
            )
            for index in range(hidden.shape[0])
        ],
        dim=0,
    )

    torch.testing.assert_close(batched, independent)


def test_repetition_penalty_is_applied_per_batch_row() -> None:
    logits = torch.tensor([[2.0, -2.0, 1.0], [2.0, -2.0, 1.0]])
    apply_repetition_penalty(logits, [[[0], [2]], [[1], []]], 0, 2.0)

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
