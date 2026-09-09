# SPDX-License-Identifier: Apache-2.0
"""Answer the student's tool calls during rollouts with a cheap LLM.

The result prompt is Convogen's tool-simulation prompt (duplexio's
`convogen/simulator.py`), so on-policy tool results have the same shape and
consistency rules as the training data's. Any OpenAI-compatible endpoint works;
the default is OpenRouter.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

TOOL_RESULT_PROMPT = """\
The tool being called is: {tool_name}
Tool description: {tool_description}

The arguments provided are:
{arguments}

The agent's conversation context so far:
{context_summary}

Prior tool calls and results in this conversation:
{tool_history}

Generate a realistic JSON response for this tool call.
- Prior tool results are authoritative facts. Never contradict them.
- Successful mutations persist. Later reads must reflect earlier writes.
- Interrupted mutations still persist even when the agent did not observe the
  result. Use the prior tool history as the canonical backend state.
- Reject duplicate creates and overlapping reservations or time blocks. Return
  a conflict or duplicate error unless the called tool explicitly updates or
  replaces the existing resource.
- Treat equivalent values such as "5 PM" and "17:00" as identical.
- Return an error object when required arguments are genuinely missing or the
  requested action conflicts with established state.
- Keep the response concise and include only fields this tool would return.

Respond with ONLY one complete JSON object, no markdown or explanation."""

SYSTEM_PROMPT = """\
You simulate the backend services behind a voice assistant's tools. The
assistant's own instructions, for context about the scenario, are:

{assistant_system_prompt}

Invent plausible, self-consistent data. Never mention that you are simulating."""


def decode_tool_calls(payload: Any) -> list[dict[str, Any]]:
    """Decode the engine's `tool_call_json` bytes: zero or more JSON objects."""
    if payload is None:
        return []
    data = bytes(payload.tolist()) if hasattr(payload, "tolist") else bytes(payload)
    text = data.decode("utf-8")
    decoder = json.JSONDecoder()
    calls, offset = [], 0
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        call, offset = decoder.raw_decode(text, offset)
        calls.append(call)
    return calls


class ToolSimulator:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    async def execute(
        self,
        call: dict[str, Any],
        *,
        tools: list[dict[str, Any]],
        assistant_system_prompt: str,
        transcript: str,
        history: list[dict[str, Any]],
    ) -> str:
        """Return canonical JSON for one call and record it in `history`."""
        name = call["name"]
        description = next(
            (tool["function"].get("description", "") for tool in tools if tool["function"]["name"] == name),
            "",
        )
        arguments = call.get("arguments", {})
        prompt = TOOL_RESULT_PROMPT.format(
            tool_name=name,
            tool_description=description,
            arguments=json.dumps(arguments, indent=2, ensure_ascii=False),
            context_summary=transcript[-4000:],
            tool_history=json.dumps(history, indent=2, ensure_ascii=False),
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(assistant_system_prompt=assistant_system_prompt)},
            {"role": "user", "content": prompt},
        ]
        result: dict[str, Any] | None = None
        for _ in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=0.3,
                )
                content = (response.choices[0].message.content or "").strip()
            except Exception as error:  # network or provider failure: the tool errors
                result = {"error": f"tool backend unavailable: {type(error).__name__}"}
                break
            if content.startswith("```"):
                content = "\n".join(content.splitlines()[1:-1]).strip()
            try:
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("tool result must be a JSON object")
                result = parsed
                break
            except (json.JSONDecodeError, ValueError) as error:
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That was invalid JSON ({error}). Return one complete JSON object only."},
                ]
        if result is None:
            result = {"error": "tool backend returned invalid JSON"}
        history.append({"name": name, "arguments": arguments, "result": result})
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
