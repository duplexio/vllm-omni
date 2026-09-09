"""Batch tool-start reads without changing request-local sampling histories."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import xgrammar as xgr
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    _emit_temperatures,
    _sample_emit,
    _sample_factorized_text_ids,
    _text_sampling,
    sample_tool_token_id,
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
            "duplexio": {"user_token_id": torch.tensor(1, device=device)},
            "duplexio_working_state": SimpleNamespace(
                sampling_generator=torch.Generator(device=device).manual_seed(17 + index),
                tool_call_constraint=constraint,
            ),
            "duplex": {"runtime_config": {
                "duplexio_text_sampling": {
                    "mode": ("argmax", "top_k", "top_p")[index % 3] if mixed else "argmax",
                    "temperature": 0.8, "top_k": len(vocab), "top_p": 0.9,
                },
                "duplexio_emit_temperatures": {"user": 0.0, "agent": 0.7, "tool_call": 0.9},
            }},
        })
    return model, infos, len(vocab)


def serial_sample(model, logits, emissions, info):
    """Original per-request draw order, including the synchronous tool read."""
    state = info["duplexio_working_state"]
    sampling = _text_sampling(info, model.tool_suppressed_token_ids)
    temperatures = _emit_temperatures(info)
    agent = _sample_factorized_text_ids(
        logits[:1], emissions[:1], silence_token_id=0,
        sampling=replace(sampling, suppressed_token_ids=model.agent_suppressed_token_ids),
        emit_temperature=temperatures.agent, generator=state.sampling_generator,
    )
    constraint = state.tool_call_constraint
    emit = False
    if constraint is not None and constraint.enabled and not constraint.active:
        emit = constraint.force_next_call or bool(_sample_emit(
            emissions[1:2], 0.0 if sampling.mode == "argmax" else temperatures.tool_call,
            generator=state.sampling_generator,
        ).item())
    tool = sample_tool_token_id(
        logits[1:2], constraint=constraint, emit=emit,
        sampling=sampling, generator=state.sampling_generator,
    )
    call = None
    if tool is None:
        tool = logits.new_full((1,), 0, dtype=torch.long)
    elif constraint.accept(tool.item()):
        call = model.tool_call_compiler.take_completed_call(constraint)
    return torch.cat((info["duplexio"]["user_token_id"].view(1), agent, tool)), call


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
        logits = torch.randn(8, 2, vocab, device=device, generator=inputs)
        emissions = torch.randn(8, 2, device=device, generator=inputs)
        expected = [serial_sample(reference, logits[row], emissions[row], info) for row, info in enumerate(original)]
        texts, calls = model.sample_text_batch(logits, emissions, infos)
        for row, (info, old) in enumerate(zip(infos, original, strict=True)):
            torch.testing.assert_close(texts[row], expected[row][0], rtol=0, atol=0)
            assert calls[row] == expected[row][1]
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
    logits = torch.zeros(8, 2, vocab, device="cuda")
    emissions = torch.full((8, 2), 100.0 if start_tool else -100.0, device="cuda")
    with ScalarReads() as serial:
        for row, info in enumerate(infos):
            serial_sample(model, logits[row], emissions[row], info)
    model, infos, _ = fixture("cuda", mixed=False)
    with ScalarReads() as batched:
        model.sample_text_batch(logits, emissions, infos)
    assert serial.count == (16 if start_tool else 8)
    assert batched.count == 0
