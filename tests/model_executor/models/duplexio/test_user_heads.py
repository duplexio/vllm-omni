# SPDX-License-Identifier: Apache-2.0
"""User policy heads share training's full-frame predictor and causal targets."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    DuplexIOLogitsProcessor,
)
from vllm_omni.model_executor.models.duplexio.text_sampling import content_distribution


class LocalVocabulary(nn.Module):
    """One-rank vocabulary projection, using the actual serving head kernel."""

    def forward(self, head, hidden):
        return DuplexIOLogitsProcessor._apply_head(self, head, hidden, None).float()


def head_model(device="cpu"):
    torch.manual_seed(93)
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    model.policy_version = 0
    model.silence_token_id = 0
    model.llm = nn.Module()
    model.llm.base_model = nn.Module()
    model.llm.base_model.lm_head = nn.Linear(32, 64, bias=False)
    model.llm.output_head_proj = nn.ModuleDict({name: nn.Linear(32, 32) for name in ("agent", "tool_call")})
    model.user_token_projection = nn.Linear(6 * 32, 32)
    model.user_emit_head = nn.Linear(6 * 32, 1)
    model.agent_emit_head = nn.Linear(6 * 32, 1)
    model.tool_call_emit_head = nn.Linear(6 * 32, 1)
    model.logits_processor = LocalVocabulary()
    model.content_distribution = content_distribution
    model.register_buffer("user_suppressed_token_ids", torch.tensor([0]), persistent=False)
    model.register_buffer("agent_suppressed_token_ids", torch.tensor([0]), persistent=False)
    model.register_buffer("tool_suppressed_token_ids", torch.tensor([0]), persistent=False)
    return model.to(device)


def sampling_info(model, device="cpu", mode="top_k", seed=17, emit_temperature=0.7):
    info = {
        "duplexio_working_state": SimpleNamespace(sampling_generator=torch.Generator(device=device).manual_seed(seed)),
        "duplex": {
            "runtime_config": {
                "duplexio_text_sampling": {"temperature": 0.6, "top_k": 4, "top_p": 0.8},
                "duplexio_user_sampling": {"content": {
                    "temperature": 0.0 if mode == "argmax" else 0.65,
                    "top_k": None if mode == "sample" else 4,
                    "top_p": 0.8 if mode == "top_p" else None,
                }},
                "duplexio_emit_temperatures": {"agent": 1.0, "tool_call": 1.0, "user": emit_temperature},
            }
        },
    }
    info["duplexio_working_state"].sampling = model.resolve_sampling(info["duplex"]["runtime_config"])
    return info


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
    infos = [sampling_info(model, device=device, mode=mode, seed=seed, emit_temperature=temperature) for seed in range(64)]
    logits = torch.tensor([[100.0, 0.2, 0.4, 1.0, 1.2, 1.4]], device=device).expand(64, -1)
    emissions = torch.linspace(-2, 2, 64, device=device)
    ids, emit_logprobs, token_logprobs = model.sample_stream_tokens(
        logits,
        emissions,
        infos,
        stream="user",
    )
    emitted = torch.cat(ids) != 0
    assert emitted.any() and (~emitted).any()
    for row, (token, emit_logprob, token_logprob) in enumerate(zip(ids, emit_logprobs, token_logprobs, strict=True)):
        if temperature == 0:
            assert emit_logprob.item() == 0
        else:
            p = torch.sigmoid(emissions[row] / temperature)
            torch.testing.assert_close(emit_logprob[0], (p if emitted[row] else 1 - p).log(), rtol=0, atol=0)
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
            torch.testing.assert_close(token_logprob[0], probs[position].log(), rtol=1e-6, atol=1e-7)


def test_saturated_bernoulli_probabilities_are_recorded_exactly():
    model = head_model()
    ids, emit_logprobs, _ = model.sample_stream_tokens(
        torch.ones(2, 6),
        torch.tensor([100.0, -100.0]),
        [sampling_info(model), sampling_info(model, seed=18)],
        stream="user",
    )
    assert ids[0].item() != 0 and ids[1].item() == 0
    assert torch.cat(emit_logprobs).tolist() == [0.0, 0.0]


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
def test_staged_user_heads_update_same_captured_graph_and_publish_version():
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
    old_logits, old_emits = logits.clone(), emits.clone()
    pushed = [(name, torch.full_like(p, 0.25)) for name, p in model.named_parameters() if name.startswith("user_")]
    source = iter(pushed)
    receiver = pytest.importorskip("duplexio.rollout_policy").PolicyWeightReceiver()
    receiver.device = rows.device
    receiver.model_runner = SimpleNamespace(model=model, get_model=lambda: model)
    receiver.policy_group = SimpleNamespace(broadcast=lambda tensor: tensor.copy_(next(source)[1]))
    receiver._policy_stream = torch.cuda.Stream()
    receiver._policy_stream.wait_stream(torch.cuda.current_stream())
    receiver._policy_pending = receiver._policy_plan = receiver._policy_version = None
    receiver._policy_buffer = []
    receiver._policy_commit_failed = False
    receiver.start_policy_weight_update(7, [[name, "float32", list(p.shape)] for name, p in pushed])
    receiver._policy_pending[1].result(timeout=10)
    graph.replay()
    torch.testing.assert_close(logits, old_logits)
    torch.testing.assert_close(emits, old_emits)
    assert model.policy_version == 0
    receiver.commit_policy_weight_update(7)
    graph.replay()
    expected_logits, expected_emits = model.project_text(rows)
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(emits, expected_emits)
    assert not torch.allclose(logits[:, 2], old_logits[:, 2])
    assert model.policy_version == 7
    assert {name: p.data_ptr() for name, p in model.named_parameters()} == pointers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA training parity")
@torch.inference_mode()
def test_training_user_timing_computation_and_next_frame_targets_match_serving():
    reference_path = os.environ.get("DUPLEXIO_USER_HEAD_REFERENCE")
    if reference_path is None:
        pytest.skip("Generate user_head_reference.py with the training environment first")
    torch.backends.cuda.matmul.allow_tf32 = False
    for case in torch.load(reference_path, weights_only=True):
        model = head_model("cuda")
        weights = [
            (name if name.startswith("user_") else "llm.base_model.lm_head.weight", tensor.cuda())
            for name, tensor in case["weights"].items()
        ]
        model.load_weights(weights)
        hidden = case["hidden"].cuda()
        dtype = getattr(torch, case["dtype"])
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            logits, emissions = model.project_text(hidden)
        torch.testing.assert_close(logits[:, 2].cpu(), case["logits"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(emissions[:, 2].float().cpu(), case["emit_logits"], rtol=1e-5, atol=1e-6)
        target_rows = case["token_rows"].cuda()
        ids = case["user_ids"].cuda()
        # Training user_timing_losses uses candidate row - 1 as its predictor.
        # linear_cross_entropy returns per-target values in the input dtype;
        # training takes their mean in that dtype, including BF16 rounding.
        ce = F.cross_entropy(logits[target_rows - 1, 2, 1:], ids[target_rows] - 1, reduction="none").to(dtype).mean()
        labels = ids[1:] != 0
        bce = F.binary_cross_entropy_with_logits(emissions[:-1, 2].float(), labels.float())
        torch.testing.assert_close(ce.cpu(), case["losses"]["user_token_ce"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bce.cpu(), case["losses"]["user_emit"], rtol=1e-5, atol=1e-6)
