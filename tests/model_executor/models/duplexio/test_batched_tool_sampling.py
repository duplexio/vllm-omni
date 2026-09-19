"""Batch tool-start reads without changing request-local sampling histories."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import xgrammar as xgr
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    _sample_emit,
    _sample_factorized_text_ids,
    sample_tool_token,
)
from vllm_omni.model_executor.models.duplexio.text_sampling import content_distribution
from vllm_omni.model_executor.models.duplexio.tool_calling import (
    ToolCallConstraintCompiler,
    ToolCallConstraintState,
    tool_call_grammar,
)


def fixture(device: str, *, mixed: bool):
    tools = [{"type": "function", "function": {
        "name": "ping", "parameters": {"type": "object", "properties": {}},
    }}]
    vocab = ["<silence>", *sorted(set("<function=ping>\n</function>"))]
    tokenizer = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=len(vocab))
    grammar = xgr.GrammarCompiler(tokenizer).compile_grammar(tool_call_grammar(tools, {"mode": "auto"}))
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.silence_token_id = 0
    model.agent_suppressed_token_ids = torch.tensor([0], device=device)
    model.user_suppressed_token_ids = model.agent_suppressed_token_ids
    model.tool_suppressed_token_ids = model.agent_suppressed_token_ids
    model.content_distribution = content_distribution
    model.tool_call_compiler = object.__new__(ToolCallConstraintCompiler)
    infos = []
    for index in range(8):
        constraint = ToolCallConstraintState(
            compiled_grammar=grammar, decoded_vocab=tuple(token.encode() for token in vocab), tools=tuple(tools),
        )
        if mixed:
            if index == 0:
                constraint = None
            elif index == 1:
                constraint = ToolCallConstraintState(compiled_grammar=None)
            elif index == 2:
                constraint.begin()
            elif index == 3:
                constraint.force_next_call = True
        infos.append({
            "duplexio_working_state": SimpleNamespace(
                sampling_generator=torch.Generator(device=device).manual_seed(17 + index),
                tool_call_constraint=constraint,
            ),
            "duplex": {"runtime_config": {
                "duplexio_text_sampling": {
                    "temperature": 0.8 if mixed and index % 3 else 0.0,
                    "top_k": len(vocab), "top_p": 0.9 if index % 3 == 2 else None,
                },
                "duplexio_emit_temperatures": {"user": 0.6, "agent": 0.7, "tool_call": 0.9},
                "duplexio_user_sampling": {"content": {
                    "temperature": 0.8 if mixed and index % 3 else 0.0,
                    "top_k": len(vocab), "top_p": 0.9 if index % 3 == 2 else None,
                }},
            }},
        })
        infos[-1]["duplexio_working_state"].sampling = model.resolve_sampling(infos[-1]["duplex"]["runtime_config"])
    return model, infos, len(vocab)


def serial_sample(model, logits, emissions, info):
    """Original per-request draw order, including the synchronous tool read."""
    state = info["duplexio_working_state"]
    sampling = state.sampling.tool
    temperatures = state.sampling.emission
    agent = _sample_factorized_text_ids(
        logits[:1], emissions[:1], silence_token_id=0,
        sampling=replace(sampling, suppressed_token_ids=model.agent_suppressed_token_ids),
        emit_temperature=temperatures.agent, generator=state.sampling_generator,
    )
    constraint = state.tool_call_constraint
    emit = False
    if constraint is not None and constraint.enabled and not constraint.active:
        emit = constraint.force_next_call or bool(_sample_emit(
            emissions[1:2], temperatures.tool_call,
            generator=state.sampling_generator,
        ).item())
    tool_sample = sample_tool_token(
        logits[1:2], constraint=constraint, emit=emit,
        sampling=sampling, generator=state.sampling_generator,
    )
    call = None
    if tool_sample is None:
        tool = logits.new_full((1,), 0, dtype=torch.long)
    else:
        tool = tool_sample.token_id
        if constraint.accept(tool.item()):
            call = model.tool_call_compiler.take_completed_call(constraint)
    user = _sample_factorized_text_ids(
        logits[2:3], emissions[2:3], silence_token_id=0,
        sampling=replace(sampling, suppressed_token_ids=model.user_suppressed_token_ids),
        emit_temperature=temperatures.user, generator=state.sampling_generator,
    )
    return torch.cat((user, agent, tool)), call


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)])
@pytest.mark.parametrize("compile_filter", [False, True])
def test_batched_tool_sampling_preserves_tokens_calls_noise_and_rng(device, compile_filter):
    if compile_filter and device == "cpu":
        pytest.skip("compiled deployment uses CUDA")
    model, infos, vocab = fixture(device, mixed=True)
    if compile_filter:
        model.content_distribution = torch.compile(
            content_distribution, fullgraph=True, dynamic=True,
            options={"emulate_precision_casts": True, "triton.cudagraphs": True},
        )
    reference, original, _ = fixture(device, mixed=True)
    inputs = torch.Generator(device=device).manual_seed(53)
    for _ in range(60):
        logits = torch.randn(8, 3, vocab, device=device, generator=inputs)
        emissions = torch.randn(8, 3, device=device, generator=inputs)
        expected = [serial_sample(reference, logits[row], emissions[row], info) for row, info in enumerate(original)]
        sampled = model.sample_text_batch(logits, emissions, infos)
        for row, (info, old) in enumerate(zip(infos, original, strict=True)):
            torch.testing.assert_close(sampled.text_ids[row], expected[row][0], rtol=0, atol=0)
            assert sampled.tool_calls[row] == expected[row][1]
            actual_rng = info["duplexio_working_state"].sampling_generator
            expected_rng = old["duplexio_working_state"].sampling_generator
            torch.testing.assert_close(
                torch.randn(1, 32, device=device, generator=actual_rng),
                torch.randn(1, 32, device=device, generator=expected_rng), rtol=0, atol=0,
            )
            assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


class ScalarReads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default and args[0].is_cuda:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("start_tool", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="checks GPU scalar readback")
def test_tools_do_not_read_gpu_scalars_per_request(start_tool):
    model, infos, vocab = fixture("cuda", mixed=False)
    logits = torch.zeros(8, 3, vocab, device="cuda")
    emissions = torch.full((8, 3), 100.0 if start_tool else -100.0, device="cuda")
    with ScalarReads() as serial:
        for row, info in enumerate(infos):
            serial_sample(model, logits[row], emissions[row], info)
    model, infos, _ = fixture("cuda", mixed=False)
    with ScalarReads() as batched:
        model.sample_text_batch(logits, emissions, infos)
    assert serial.count == (16 if start_tool else 8)
    assert batched.count == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)])
def test_agent_logprobs_match_the_distributions_actually_sampled(device):
    """Behaviour log-probs must equal the distributions the draws came from.

    Verified without replaying the generators: the truncated content distribution
    is deterministic given the logits, the emit outcome is recoverable from
    whether the returned id is silence, and a thresholded (greedy) decision has
    probability one. An importance ratio divides by these numbers, so a wrong
    value here is silently wrong training rather than a crash.
    """
    model, infos, vocab = fixture(device, mixed=True)
    inputs = torch.Generator(device=device).manual_seed(97)
    logits = torch.randn(8, 2, vocab, device=device, generator=inputs)
    emissions = torch.randn(8, 2, device=device, generator=inputs)

    ids, emit_logprobs, token_logprobs = model.sample_agent_tokens(
        logits[:, 0], emissions[:, 0], infos
    )

    checked_emitted = 0
    for row, info in enumerate(infos):
        sampling = info["duplexio_working_state"].sampling.agent
        emit_logprob, token_logprob = emit_logprobs[row], token_logprobs[row]
        assert emit_logprob.shape == token_logprob.shape == (1,)
        assert emit_logprob.dtype == token_logprob.dtype == torch.float32

        emitted = int(ids[row]) != model.silence_token_id
        scaled = emissions[row, 0].float() / info["duplexio_working_state"].sampling.emission.agent
        torch.testing.assert_close(
            emit_logprob.reshape(()),
            torch.nn.functional.logsigmoid(scaled if emitted else -scaled),
            atol=1e-5, rtol=1e-4,
        )
        assert float(emit_logprob) < 0.0

        if not emitted or sampling.temperature == 0:
            # A content draw discarded by a silent row took no action.
            assert float(token_logprob) == 0.0
            continue
        indices, probabilities = content_distribution(logits[row:row + 1, 0], sampling)
        position = (indices[0] == int(ids[row])).nonzero().flatten()
        assert position.numel() == 1, "sampled id must lie in the truncated support"
        torch.testing.assert_close(
            token_logprob.reshape(()),
            probabilities[0, int(position)].float().log(),
            atol=1e-5, rtol=1e-4,
        )
        checked_emitted += 1

    assert checked_emitted, "fixture produced no emitted agent row to verify"


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)])
@pytest.mark.parametrize("mode", ["argmax", "top_k", "top_p"])
def test_tool_emit_logprobs_score_all_decisions_including_forced_emit_and_wait(device, mode):
    model, infos, vocab = fixture(device, mixed=True)
    for info in infos:
        info["duplex"]["runtime_config"]["duplexio_text_sampling"]["temperature"] = 0.0 if mode == "argmax" else 0.8
    logits = torch.zeros(8, 3, vocab, device=device)
    emissions = torch.zeros(8, 3, device=device)
    emissions[:, 1] = torch.tensor([0, 0, -100, -100, -100, 100, -0.4, 0.4], device=device)
    sampled = model.sample_text_batch(logits, emissions, infos)
    for row in range(8):
        emitted = sampled.text_ids[row][2].item() != model.silence_token_id
        # Score raw logits, independent of sampling temperature or forced actions.
        signed_logit = emissions[row, 1] if emitted else -emissions[row, 1]
        expected = -torch.logaddexp(torch.zeros_like(signed_logit), -signed_logit)
        torch.testing.assert_close(sampled.tool_emit_logprobs[row][0], expected)
        if not emitted or mode == "argmax":
            assert sampled.tool_token_logprobs[row].item() == 0
        if row < 2:
            assert not emitted
        elif row < 4:
            assert emitted  # active continuation / forced start ignores the negative emit logit
            assert sampled.tool_emit_logprobs[row].item() == -100
