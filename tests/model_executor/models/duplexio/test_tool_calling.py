# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import xgrammar as xgr

from tests.model_executor.models.duplexio.reference_sampling import ReferenceSampling, sample_tool_token
from vllm_omni.model_executor.models.duplexio.tool_calling import (
    ToolCallCapture,
    ToolCallConstraintCompiler,
    ToolCallConstraintState,
    tool_call_grammar,
)


@pytest.mark.parametrize(("opening_token_id", "closed"), [(1, True), (1, False), (-1, False)])
def test_tool_call_latches_grammar_until_complete_then_releases_stream(opening_token_id: int, closed: bool) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    body = "\n<function=ping>\n</function>\n</tool_call>" if closed else "<function=ping>\n</function>"
    vocab = ["<silence>", "<tool_call>", "</tool_call>", *sorted(set(body))]
    grammar = tool_call_grammar(tools, {"mode": "auto"}, closed=closed)
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
        opening_token_id=opening_token_id,
        closed=closed,
    )
    sampling = ReferenceSampling(
        temperature=0.0,
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

    # A v10+ export's call opens with <tool_call> as in training, then follows its
    # grammar; an older export's starts with the grammar. A v11 export's call is the
    # teacher chat's whole block and completes only at </tool_call>.
    assert generated == ("<tool_call>" if opening_token_id >= 0 else "") + body
    assert complete
    assert not state.active

    compiler = object.__new__(ToolCallConstraintCompiler)
    assert compiler.take_completed_call(state) == {
        "name": "ping",
        "arguments": {},
    }
    state.begin()
    assert state.active


def test_session_grammar_compiles_without_blocking_its_request() -> None:
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]
    vocab = ["<silence>", "<tool_call>", *sorted(set("<function=ping>\n</function>"))]
    compiler = object.__new__(ToolCallConstraintCompiler)
    compiler.decoded_vocab = tuple(value.encode() for value in vocab)
    compiler.opening_token_id = 1
    compiler.closed = True
    compiler.compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=len(vocab))
    )
    compiler.executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    compiler.executor.submit(release.wait)

    state = compiler.new_state(tools, {"mode": "auto"})
    assert state.enabled and not state.compiled_grammar.done()
    fork = state.fork()
    release.set()
    state.begin()

    assert state.active and not fork.active
    assert fork.grammar is state.grammar
    compiler.executor.shutdown()


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


def test_a_closed_call_completes_at_its_closing_tag() -> None:
    tools = (
        {
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object", "properties": {"level": {"type": "integer"}}, "required": ["level"]},
            },
        },
    )
    capture = ToolCallCapture(tools, closed=True)
    capture.accept_text("\n<function=ping>\n<parameter=level>\n3\n</parameter>\n</function>\n")
    assert capture.stage == "parameter"  # still waiting for </tool_call>
    capture.accept_text("</tool_call>")
    assert capture.complete() == {"name": "ping", "arguments": {"level": 3}}
