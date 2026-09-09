# SPDX-License-Identifier: Apache-2.0
"""Prepare Convogen scenario prompts for synchronous DuplexIO self-play.

The source file is the JSONL emitted by Convogen.  It contains separate
``user`` and ``assistant`` system-prompt metadata even though the training
conversation model only keeps the assistant prompt.  This adapter preserves
both prompts, renders each with Qwen's native system template, and leaves voice
selection explicit because dataset TTS voice IDs are not necessarily present
in a serving checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from convogen.scenarios import ToolDefinition, _tool_def_to_openai_tool
from convogen.simulator import AGENT_TURN_SUFFIX, USER_TURN_SUFFIX
from duplexio.duplexio_data import render_chat_system_content
from duplexio.multistream.tokenization import _encode_text
from vllm_omni.experimental.fullduplex.duplexio.self_play import (
    PreparedRole,
    PreparedScenarioPair,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import DuplexIOConfig


def _role_metadata(row: dict[str, Any], role: str) -> dict[str, Any]:
    value = row.get(role)
    return value if isinstance(value, dict) else {}


def _scenario_data(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("scenario")
    return value if isinstance(value, dict) else row


def _prompt(row: dict[str, Any], role: str) -> str:
    role_value = _role_metadata(row, role).get("system_prompt")
    if isinstance(role_value, str) and role_value.strip():
        return role_value
    scenario = _scenario_data(row)
    key = "agent_system_prompt" if role == "assistant" else "user_system_prompt"
    value = scenario.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Convogen row {row.get('id')!r} has no {key}")
    return value


def _tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tools = _scenario_data(row).get("tools", row.get("tools", []))
    if not isinstance(raw_tools, list):
        raise ValueError(f"Convogen row {row.get('id')!r} has invalid tools")
    return [
        _tool_def_to_openai_tool(ToolDefinition.model_validate(raw_tool))
        for raw_tool in raw_tools
    ]


def _simulator_prompt(row: dict[str, Any], role: str) -> str:
    prompt = _prompt(row, role)
    if role == "assistant":
        completion_condition = _scenario_data(row).get("completion_condition")
        if isinstance(completion_condition, str) and completion_condition.strip():
            prompt += (
                "\n\nCompletion objective: "
                + completion_condition
                + "\nWork toward every concrete part of this objective. "
                "When success requires user verification, ask them to test or "
                "confirm it rather than assuming an expected result worked."
            )
        return prompt + AGENT_TURN_SUFFIX
    return prompt + USER_TURN_SUFFIX


def prepare_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    agent_voice: str,
    user_voice: str,
) -> PreparedScenarioPair:
    conversation_id = row.get("id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("Convogen rows require a nonempty string id")
    tools = _tools(row)
    agent_prompt = _simulator_prompt(row, "assistant")
    user_prompt = _simulator_prompt(row, "user")
    agent_content = render_chat_system_content(tokenizer, agent_prompt, tools or None)
    user_content = render_chat_system_content(tokenizer, user_prompt, None)
    return PreparedScenarioPair(
        conversation_id=conversation_id,
        agent=PreparedRole(
            system_token_ids=_encode_text(tokenizer, agent_content),
            voice=agent_voice,
            tools=tools,
            metadata=_role_metadata(row, "assistant"),
        ),
        user=PreparedRole(
            system_token_ids=_encode_text(tokenizer, user_content),
            voice=user_voice,
            metadata=_role_metadata(row, "user"),
        ),
        metadata={
            "domain": row.get("domain"),
            "completion_condition": _scenario_data(row).get("completion_condition"),
            "tool_system_prompt": _scenario_data(row).get("tool_system_prompt"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("conversations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--agent-voice")
    parser.add_argument("--user-voice")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Prepared output already exists")
    config = DuplexIOConfig.from_pretrained(args.checkpoint, local_files_only=True)
    default_voice = config.default_voice
    agent_voice = args.agent_voice or default_voice
    user_voice = args.user_voice or default_voice
    if not agent_voice or not user_voice:
        parser.error("Pass --agent-voice and --user-voice when the checkpoint has no default voice")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )
    pairs = []
    with args.conversations.open() as lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                pairs.append(
                    prepare_row(
                        row,
                        tokenizer,
                        agent_voice=agent_voice,
                        user_voice=user_voice,
                    ).model_dump()
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid Convogen row {line_number}: {exc}") from exc
    torch.save(pairs, args.output)
    print(json.dumps({"prepared": len(pairs), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
