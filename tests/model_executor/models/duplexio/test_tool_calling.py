# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
import xgrammar as xgr

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    TokenSamplingOptions,
    sample_tool_token,
)
from vllm_omni.model_executor.models.duplexio.tool_calling import (
    ToolCallCapture,
    ToolCallConstraintCompiler,
    ToolCallConstraintState,
    tool_call_grammar,
)


def test_tool_call_latches_grammar_until_complete_then_releases_stream() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    body = "<function=ping>\n</function>"
    vocab = ["<silence>", *sorted(set(body))]
    grammar = tool_call_grammar(tools, {"mode": "auto"})
    assert grammar is not None
    tokenizer_info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        vocab_size=len(vocab),
    )
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    state = ToolCallConstraintState(
        compiled_grammar=compiled,
        decoded_vocab=tuple(value.encode() for value in vocab),
        tools=tuple(tools),
    )
    sampling = TokenSamplingOptions(
        mode="argmax",
        temperature=1.0,
        top_k=len(vocab),
        top_p=1.0,
        suppressed_token_ids=torch.tensor([0], dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(0)
    generated = ""

    for frame in range(100):
        logits = torch.zeros(1, len(vocab))
        logits[0, 0] = 100
        sampled = sample_tool_token(
            logits,
            constraint=state,
            emit=frame == 0,
            sampling=sampling,
            generator=generator,
        )
        assert sampled is not None
        assert sampled.logprob.item() == 0
        sampled_id = sampled.token_id.item()
        complete = state.accept(sampled_id)
        assert sampled_id != 0
        generated += vocab[sampled_id]
        if complete:
            break

    assert generated == body
    assert complete
    assert not state.active

    compiler = object.__new__(ToolCallConstraintCompiler)
    assert compiler.take_completed_call(state) == {
        "name": "ping",
        "arguments": {},
    }
    state.begin()
    assert state.active


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)])
@pytest.mark.parametrize("mode", ["argmax", "top_k", "top_p"])
def test_forced_continuation_records_actual_constrained_token_probability(device, mode):
    vocab = ["<silence>", "a", "b", "c", "d"]
    tokenizer = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=len(vocab))
    grammar = xgr.GrammarCompiler(tokenizer).compile_grammar('root ::= "a" | "b" | "c"')
    state = ToolCallConstraintState(compiled_grammar=grammar)
    state.begin()
    sampling = TokenSamplingOptions(
        mode=mode, temperature=0.7, top_k=2, top_p=0.8,
        suppressed_token_ids=torch.tensor([0], device=device),
    )
    logits = torch.tensor([[100.0, 0.2, 0.6, 0.9, 90.0]], device=device)
    sampled = sample_tool_token(
        logits, constraint=state, emit=False, sampling=sampling,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    assert sampled is not None
    token = sampled.token_id.item()
    assert token in (2, 3)  # the grammar excludes the two highest raw logits; top-k excludes a
    if mode == "argmax":
        assert token == 3
        assert sampled.logprob.item() == 0
    else:
        # Independent reference: grammar leaves a/b/c, then top-k retains c/b.
        values = torch.tensor([0.9, 0.6], device=device) / 0.7
        probabilities = values.softmax(-1)
        if mode == "top_p" and probabilities[0] > 0.8:
            probabilities = torch.tensor([1.0, 0.0], device=device)
        expected = probabilities[0 if token == 3 else 1].log()
        torch.testing.assert_close(sampled.logprob[0], expected)
        assert sampled.logprob.item() < 0  # forcing emit does not force the content token


def test_constrained_sampler_captures_structured_call_while_accepting_bytes() -> None:
    tools = (
        {
            "type": "function",
            "function": {
                "name": "set_level",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer", "minimum": 0},
                        "label": {"type": "string"},
                    },
                    "required": ["level", "label"],
                },
            },
        },
    )
    text = (
        "<function=set_level>\n"
        "<parameter=level>\n3\n</parameter>\n"
        "<parameter=label>\nstille 🌙\n</parameter>\n"
        "</function>"
    )
    encoded = text.encode()
    capture = ToolCallCapture(tools)
    for byte in encoded:
        capture.accept_bytes(bytes([byte]))
    state = ToolCallConstraintState(
        compiled_grammar=None,
        completed_call=capture.complete(),
    )
    compiler = object.__new__(ToolCallConstraintCompiler)

    assert compiler.take_completed_call(state) == {
        "name": "set_level",
        "arguments": {"level": 3, "label": "stille 🌙"},
    }
    assert state.completed_call is None


def test_tool_grammar_rejects_schema_constraints_it_cannot_enforce() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "pattern": "^[a-z]+$"},
                    },
                },
            },
        }
    ]

    with pytest.raises(ValueError, match="unsupported constraints"):
        tool_call_grammar(tools, {"mode": "auto"})
