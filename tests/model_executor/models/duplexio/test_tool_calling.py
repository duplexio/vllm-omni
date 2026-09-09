# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
import xgrammar as xgr

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    TokenSamplingOptions,
    sample_tool_token_id,
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
        token_id = sample_tool_token_id(
            logits,
            constraint=state,
            emit=frame == 0,
            sampling=sampling,
            generator=generator,
        )
        assert token_id is not None
        sampled_id = token_id.item()
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
