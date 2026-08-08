# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constrained sampling and semantic materialization for tool calls."""

from __future__ import annotations

import codecs
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import xgrammar as xgr
from jsonschema.validators import validator_for
from transformers import PreTrainedTokenizerBase

TOOL_CALL_STRING_LIMIT = 512
SCHEMA_ANNOTATION_KEYS = {
    "$comment",
    "$id",
    "$schema",
    "default",
    "deprecated",
    "description",
    "examples",
    "format",
    "readOnly",
    "title",
    "writeOnly",
}
STRING_SCHEMA_KEYS = {
    "type",
    "enum",
    "const",
    "minLength",
    "maxLength",
    *SCHEMA_ANNOTATION_KEYS,
}
TOOL_PARAMETER_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    *SCHEMA_ANNOTATION_KEYS,
}


def literal_grammar(text: str) -> xgr.Grammar:
    return xgr.Grammar.from_ebnf(f"root ::= {json.dumps(text)}")


def empty_grammar() -> xgr.Grammar:
    return literal_grammar("")


def string_value_grammar(schema: Mapping[str, Any]) -> xgr.Grammar:
    unsupported = {
        key
        for key in schema
        if key not in STRING_SCHEMA_KEYS and not key.startswith("x-")
    }
    if unsupported:
        raise ValueError(
            "DuplexIO string parameters use unsupported constraints: "
            f"{sorted(unsupported)}"
        )
    min_length = schema.get("minLength", 0)
    max_length = schema.get("maxLength", TOOL_CALL_STRING_LIMIT)
    if not isinstance(min_length, int) or not isinstance(max_length, int):
        raise ValueError("DuplexIO string lengths must be integers")
    max_length = min(max_length, TOOL_CALL_STRING_LIMIT)
    if min_length < 0 or max_length < min_length:
        raise ValueError("DuplexIO string length bounds are invalid")

    enum = schema.get("enum")
    if isinstance(enum, list):
        values = [
            value
            for value in enum
            if isinstance(value, str)
            and min_length <= len(value) <= max_length
            and ("const" not in schema or value == schema["const"])
            and "<" not in value
            and "\n" not in value
            and "\r" not in value
        ]
        if not values:
            raise ValueError("DuplexIO string enum has no representable value")
        return xgr.Grammar.union(*(literal_grammar(value) for value in values))

    if "const" in schema:
        const = schema["const"]
        if not isinstance(const, str):
            raise ValueError("DuplexIO string const values must be strings")
        if (
            not min_length <= len(const) <= max_length
            or "<" in const
            or "\n" in const
            or "\r" in const
        ):
            raise ValueError("DuplexIO string const is not representable")
        return literal_grammar(const)

    return xgr.Grammar.from_ebnf(
        f"root ::= [^<\\n\\r]{{{min_length},{max_length}}}"
    )


def parameter_value_grammar(schema: Mapping[str, Any]) -> xgr.Grammar:
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list) and alternatives:
        unsupported = {
            key
            for key in schema
            if key != "anyOf"
            and key not in SCHEMA_ANNOTATION_KEYS
            and not key.startswith("x-")
        }
        if unsupported:
            raise ValueError(
                "DuplexIO anyOf parameters cannot carry sibling constraints: "
                f"{sorted(unsupported)}"
            )
        return xgr.Grammar.union(
            *(
                parameter_value_grammar(alternative)
                for alternative in alternatives
                if isinstance(alternative, Mapping)
            )
        )
    schema_type = schema.get("type", "string")
    if "type" not in schema:
        raise ValueError("DuplexIO tool parameters must declare a JSON Schema type")
    if isinstance(schema_type, list):
        return xgr.Grammar.union(
            *(
                parameter_value_grammar({**schema, "type": value})
                for value in schema_type
                if isinstance(value, str)
            )
        )
    if schema_type == "string":
        return string_value_grammar(schema)
    return xgr.Grammar.from_json_schema(
        dict(schema),
        any_whitespace=False,
        strict_mode=True,
    )


def tool_function_grammar(tool: Mapping[str, Any]) -> xgr.Grammar:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("DuplexIO tools must contain a function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("DuplexIO tool names must be non-empty strings")
    if any(character in name for character in "<>=\n\r"):
        raise ValueError(f"DuplexIO tool name contains an XML delimiter: {name!r}")

    parameters = function.get("parameters") or {
        "type": "object",
        "properties": {},
    }
    if not isinstance(parameters, Mapping) or parameters.get("type", "object") != "object":
        raise ValueError(f"DuplexIO tool {name!r} parameters must be an object schema")
    validator_for(parameters).check_schema(dict(parameters))
    unsupported_parameter_keys = {
        key
        for key in parameters
        if key not in TOOL_PARAMETER_KEYS and not key.startswith("x-")
    }
    if unsupported_parameter_keys:
        raise ValueError(
            f"DuplexIO tool {name!r} uses unsupported object constraints: "
            f"{sorted(unsupported_parameter_keys)}"
        )
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise ValueError(f"DuplexIO tool {name!r} has invalid properties or required fields")
    required_names = set(required)
    if not required_names.issubset(properties):
        missing = sorted(required_names.difference(properties))
        raise ValueError(f"DuplexIO tool {name!r} requires unknown parameters: {missing}")

    parts = [literal_grammar(f"<function={name}>\n")]
    for parameter_name, parameter_schema in properties.items():
        if not isinstance(parameter_name, str) or not isinstance(parameter_schema, Mapping):
            raise ValueError(f"DuplexIO tool {name!r} has an invalid parameter schema")
        if any(character in parameter_name for character in "<>=\n\r"):
            raise ValueError(
                f"DuplexIO parameter name contains an XML delimiter: {parameter_name!r}"
            )
        parameter = xgr.Grammar.concat(
            literal_grammar(f"<parameter={parameter_name}>\n"),
            parameter_value_grammar(parameter_schema),
            literal_grammar("\n</parameter>\n"),
        )
        parts.append(
            parameter
            if parameter_name in required_names
            else xgr.Grammar.union(empty_grammar(), parameter)
        )
    parts.append(literal_grammar("</function>"))
    return xgr.Grammar.concat(*parts)


def selected_tools(
    tools: Sequence[Mapping[str, Any]],
    tool_choice: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    mode = tool_choice.get("mode")
    if mode == "none" or not tools:
        return []
    if mode != "named":
        return list(tools)
    name = tool_choice.get("name")
    selected = [tool for tool in tools if tool.get("function", {}).get("name") == name]
    if not selected:
        raise ValueError(f"DuplexIO tool_choice selected unknown function {name!r}")
    return selected


def tool_call_grammar(
    tools: Sequence[Mapping[str, Any]],
    tool_choice: Mapping[str, Any],
) -> xgr.Grammar | None:
    available = selected_tools(tools, tool_choice)
    if not available:
        return None
    return xgr.Grammar.union(*(tool_function_grammar(tool) for tool in available))


@dataclass
class ToolCallCapture:
    """Capture tool-call fields as grammar-constrained bytes are accepted."""

    tools: tuple[Mapping[str, Any], ...]
    decoder: Any = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")()
    )
    buffer: str = ""
    stage: str = "function"
    function_name: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    parameter_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

    def fork(self) -> ToolCallCapture:
        decoder = codecs.getincrementaldecoder("utf-8")()
        decoder.setstate(self.decoder.getstate())
        return ToolCallCapture(
            tools=self.tools,
            decoder=decoder,
            buffer=self.buffer,
            stage=self.stage,
            function_name=self.function_name,
            properties=self.properties,
            parameter_name=self.parameter_name,
            arguments=dict(self.arguments),
        )

    def accept_bytes(self, value: bytes) -> None:
        self.accept_text(self.decoder.decode(value, final=False))

    def accept_text(self, value: str) -> None:
        self.buffer += value
        while True:
            if self.stage == "function":
                header_end = self.buffer.find(">\n")
                if header_end < 0:
                    return
                prefix = "<function="
                assert self.buffer.startswith(prefix)
                self.function_name = self.buffer[len(prefix) : header_end]
                tool = next(
                    tool
                    for tool in self.tools
                    if tool.get("function", {}).get("name") == self.function_name
                )
                function = tool["function"]
                assert isinstance(function, Mapping)
                parameters = function.get("parameters") or {
                    "type": "object",
                    "properties": {},
                }
                assert isinstance(parameters, Mapping)
                properties = parameters.get("properties", {})
                assert isinstance(properties, Mapping)
                self.properties = properties
                self.buffer = self.buffer[header_end + 2 :]
                self.stage = "parameter"
                continue

            if self.stage == "parameter":
                closing_tag = "</function>"
                if closing_tag.startswith(self.buffer):
                    if len(self.buffer) < len(closing_tag):
                        return
                    self.buffer = self.buffer[len(closing_tag) :]
                    self.stage = "done"
                    continue
                parameter_prefix = "<parameter="
                if parameter_prefix.startswith(self.buffer):
                    return
                assert self.buffer.startswith(parameter_prefix)
                header_end = self.buffer.find(">\n")
                if header_end < 0:
                    return
                self.parameter_name = self.buffer[
                    len(parameter_prefix) : header_end
                ]
                assert self.parameter_name in self.properties
                self.buffer = self.buffer[header_end + 2 :]
                self.stage = "value"
                continue

            if self.stage == "value":
                value_end_marker = "\n</parameter>\n"
                value_end = self.buffer.find(value_end_marker)
                if value_end < 0:
                    return
                assert self.parameter_name is not None
                schema = self.properties[self.parameter_name]
                assert isinstance(schema, Mapping)
                self.arguments[self.parameter_name] = decode_constrained_parameter(
                    self.buffer[:value_end],
                    schema,
                )
                self.buffer = self.buffer[value_end + len(value_end_marker) :]
                self.parameter_name = None
                self.stage = "parameter"
                continue

            assert self.stage == "done"
            return

    def complete(self) -> dict[str, Any]:
        self.accept_text(self.decoder.decode(b"", final=True))
        assert self.stage == "done"
        assert not self.buffer
        assert self.function_name is not None
        return {
            "name": self.function_name,
            "arguments": self.arguments,
        }


@dataclass
class ToolCallConstraintState:
    compiled_grammar: xgr.CompiledGrammar | None
    decoded_vocab: tuple[bytes, ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    force_next_call: bool = False
    matcher: xgr.GrammarMatcher | None = None
    capture: ToolCallCapture | None = None
    completed_call: dict[str, Any] | None = None

    def fork(self) -> ToolCallConstraintState:
        return ToolCallConstraintState(
            compiled_grammar=self.compiled_grammar,
            decoded_vocab=self.decoded_vocab,
            tools=self.tools,
            force_next_call=self.force_next_call,
            matcher=self.matcher.fork() if self.matcher is not None else None,
            capture=self.capture.fork() if self.capture is not None else None,
            completed_call=(
                dict(self.completed_call)
                if self.completed_call is not None
                else None
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.compiled_grammar is not None

    @property
    def active(self) -> bool:
        return self.matcher is not None

    def begin(self) -> None:
        if self.compiled_grammar is None:
            raise RuntimeError("Cannot begin a DuplexIO tool call without tools")
        if self.matcher is not None:
            raise RuntimeError("DuplexIO tool call state is already active")
        self.matcher = xgr.GrammarMatcher(
            self.compiled_grammar,
            terminate_without_stop_token=True,
        )
        self.capture = ToolCallCapture(self.tools)
        self.completed_call = None

    def next_token_bitmask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        if self.matcher is None:
            raise RuntimeError("DuplexIO tool-call grammar is not active")
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self.matcher.fill_next_token_bitmask(bitmask)
        return bitmask.to(device=device, non_blocking=True)

    def accept(self, token_id: int) -> bool:
        if self.matcher is None or not self.matcher.accept_token(token_id):
            raise RuntimeError(f"DuplexIO tool grammar rejected sampled token {token_id}")
        assert self.capture is not None
        self.capture.accept_bytes(self.decoded_vocab[token_id])
        completed = self.matcher.is_completed()
        if completed:
            self.completed_call = self.capture.complete()
            self.matcher = None
            self.capture = None
            self.force_next_call = False
        return completed


class ToolCallConstraintCompiler:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: int) -> None:
        self.tokenizer = tokenizer
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size,
        )
        self.decoded_vocab = tuple(tokenizer_info.decoded_vocab)
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            cache_limit_bytes=64 * 1024 * 1024,
        )

    def new_state(
        self,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: Mapping[str, Any],
    ) -> ToolCallConstraintState:
        grammar = tool_call_grammar(tools, tool_choice)
        if grammar is None:
            return ToolCallConstraintState(compiled_grammar=None)
        return ToolCallConstraintState(
            compiled_grammar=self.compiler.compile_grammar(grammar),
            decoded_vocab=self.decoded_vocab,
            tools=tuple(tools),
            force_next_call=tool_choice.get("mode") in {"required", "named"},
        )

    def take_completed_call(
        self,
        state: ToolCallConstraintState,
    ) -> dict[str, Any]:
        if state.completed_call is None:
            raise RuntimeError("DuplexIO tool-call constraint has no completed call")
        tool_call = state.completed_call
        state.completed_call = None
        return tool_call


def decode_constrained_parameter(value: str, schema: Mapping[str, Any]) -> Any:
    if any(
        string_value_matches_schema(value, alternative)
        for alternative in string_schema_alternatives(schema)
    ):
        return value
    return json.loads(value)


def string_schema_alternatives(
    schema: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        return [
            string_schema
            for alternative in alternatives
            if isinstance(alternative, Mapping)
            for string_schema in string_schema_alternatives(alternative)
        ]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return (
            [{**schema, "type": "string"}]
            if "string" in schema_type
            else []
        )
    return [schema] if schema_type == "string" else []


def string_value_matches_schema(
    value: str,
    schema: Mapping[str, Any],
) -> bool:
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    min_length = schema.get("minLength", 0)
    max_length = schema.get("maxLength", TOOL_CALL_STRING_LIMIT)
    assert isinstance(min_length, int) and isinstance(max_length, int)
    return min_length <= len(value) <= max_length


__all__ = [
    "ToolCallConstraintCompiler",
    "ToolCallConstraintState",
    "tool_call_grammar",
]
