# SPDX-License-Identifier: Apache-2.0
"""User policy heads read the full frame and load in place."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from tests.model_executor.models.duplexio.reference_sampling import sample_text_batch
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
)


class LocalVocabulary(nn.Module):
    """One-rank vocabulary projection, using the actual serving head kernel."""

    head_dtype = torch.float32

    def forward(self, head, hidden):
        return LogitsProcessor._apply_head(self, head, hidden, None)


def head_model(device="cpu"):
    torch.manual_seed(93)
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    model.policy_version = 0
    model.silence_token_id = 0
    model.lm_head = nn.Linear(32, 64, bias=False)
    model.lm_head.quant_method = UnquantizedEmbeddingMethod()
    model.user_token_projection = nn.Linear(6 * 32, 32)
    model.user_emit_head = nn.Linear(6 * 32, 1)
    model.agent_emit_head = nn.Linear(6 * 32, 1)
    model.tool_call_emit_head = nn.Linear(6 * 32, 1)
    model.logits_processor = LocalVocabulary()
    model.text_config = SimpleNamespace(vocab_size=64)
    model.config = SimpleNamespace(flowmap_config={"sampling_temperature": 1.0})
    model.register_buffer("user_suppressed_token_ids", torch.tensor([0]), persistent=False)
    model.register_buffer("agent_suppressed_token_ids", torch.tensor([0]), persistent=False)
    model.register_buffer("tool_suppressed_token_ids", torch.tensor([0]), persistent=False)
    model.init_text_sampling(64, 64)
    return model.to(device)


def sample_user(model, logits, emissions, infos):
    """The user stream of a batch draw; other streams wait and extra ids are impossible."""
    padded = F.pad(logits, (0, 64 - logits.shape[-1]), value=-torch.inf)
    silent = torch.full_like(emissions, -100.0)
    sampled = sample_text_batch(
        model,
        padded.unsqueeze(1).expand(-1, 3, -1),
        torch.stack((silent, silent, emissions), dim=1),
        infos,
    )
    return sampled.text_ids[:, 0], sampled.frame_logprobs[:, 4], sampled.frame_logprobs[:, 5]


def sampling_info(model, device="cpu", mode="top_k", emit_temperature=0.7):
    sampling = model.resolve_sampling(
        {
            "agent": {"emission": {"temperature": 1.0}, "content": {"temperature": 0.6, "top_k": 4, "top_p": 0.8}},
            "user": {
                "emission": {"temperature": emit_temperature},
                "content": {
                    "temperature": 0.0 if mode == "argmax" else 0.65,
                    "top_k": None if mode == "sample" else 4,
                    "top_p": 0.8 if mode == "top_p" else None,
                },
            },
        }
    )
    return {"duplexio_working_state": SimpleNamespace(tool_call_constraint=None, sampling=sampling)}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sampling"),
        ),
    ],
)
@pytest.mark.parametrize("mode", ["argmax", "top_k", "top_p", "sample"])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_user_behavior_probabilities_include_waits_and_actual_truncation(mode, temperature, device):
    model = head_model(device)
    infos = [sampling_info(model, device=device, mode=mode, emit_temperature=temperature) for _ in range(64)]
    logits = torch.tensor([[100.0, 0.2, 0.4, 1.0, 1.2, 1.4]], device=device).expand(64, -1)
    emissions = torch.linspace(-2, 2, 64, device=device)
    ids, emit_logprobs, token_logprobs = sample_user(model, logits, emissions, infos)
    emitted = ids != 0
    assert emitted.any() and (~emitted).any()
    for row, (token, emit_logprob, token_logprob) in enumerate(zip(ids, emit_logprobs, token_logprobs, strict=True)):
        if temperature == 0:
            assert emit_logprob.item() == 0
        else:
            p = torch.sigmoid(emissions[row] / temperature)
            torch.testing.assert_close(emit_logprob, (p if emitted[row] else 1 - p).log(), rtol=1e-6, atol=1e-7)
        if not emitted[row] or mode == "argmax":
            assert token_logprob.item() == 0
        else:
            # Independent dense reference for temperature, silence exclusion,
            # top-k, then the same inclusive nucleus cutoff used for sampling.
            values, indices = (logits[row, 1:] / 0.65).topk(5 if mode == "sample" else 4)
            if mode == "top_p":
                remove = values.softmax(-1).cumsum(-1) > 0.8
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                values[remove] = -torch.inf
            probs = values.softmax(-1)
            position = (indices + 1 == token.item()).nonzero().item()
            torch.testing.assert_close(token_logprob, probs[position].log(), rtol=1e-6, atol=1e-7)


def test_saturated_bernoulli_probabilities_are_recorded_exactly():
    model = head_model()
    ids, emit_logprobs, _ = sample_user(
        model,
        torch.ones(2, 6),
        torch.tensor([100.0, -100.0]),
        [sampling_info(model), sampling_info(model)],
    )
    assert ids[0].item() != 0 and ids[1].item() == 0
    assert emit_logprobs.tolist() == [0.0, 0.0]


def test_checkpoint_loader_loads_both_user_heads_in_place():
    model = head_model()
    pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
    weights = [(name, torch.full_like(p, 0.25)) for name, p in model.named_parameters() if name.startswith("user_")]
    with torch.inference_mode():
        loaded = model.load_weights(weights)
    assert loaded == {name for name, _ in weights}
    assert len(loaded) == 4
    for name, expected in weights:
        actual = model.get_parameter(name)
        assert actual.data_ptr() == pointers[name]
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph required")
@torch.inference_mode()
def test_loaded_user_heads_update_the_captured_graph_and_publish_version():
    model = head_model("cuda")
    rows = torch.randn(3, 6, 32, device="cuda")
    pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.project_text(rows)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, emits = model.project_text(rows)
    graph.replay()
    old_logits = logits.clone()
    pushed = [(name, torch.full_like(p, 0.25)) for name, p in model.named_parameters() if name.startswith("user_")]
    model.load_weights(pushed)
    model.set_policy_version(7)
    graph.replay()
    expected_logits, expected_emits = model.project_text(rows)
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(emits, expected_emits)
    assert not torch.allclose(logits[:, 2], old_logits[:, 2])
    assert model.policy_version == 7
    assert {name: p.data_ptr() for name, p in model.named_parameters()} == pointers
    with pytest.raises(ValueError):
        model.set_policy_version(7)
