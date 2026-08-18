# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import (
    NativeRealtimeSessionProtocol,
)


def test_session_assigns_tool_call_ids_and_accepts_result() -> None:
    session = DuplexSession(
        session_id="session-1",
        config=DuplexSessionConfig(),
    )

    call = session.register_tool_call(
        name="get_current_time",
        arguments={"timezone": "Europe/Copenhagen"},
    )

    assert call.call_id.startswith("call_")
    assert call.item_id.startswith("item_")
    assert session.pending_tool_call(call.call_id) == call
    assert session.history[-1]["tool_calls"][0]["id"] == call.call_id

    result = session.complete_tool_call(
        call_id=call.call_id,
        item_id="item-result",
        output='{"time":"18:30"}',
    )

    assert result == {
        "role": "tool",
        "tool_call_id": call.call_id,
        "content": '{"time":"18:30"}',
    }
    assert session.pending_tool_call(call.call_id) is None


def test_session_tracks_concurrent_tool_calls_and_out_of_order_results() -> None:
    session = DuplexSession(
        session_id="session-1",
        config=DuplexSessionConfig(),
    )
    first = session.register_tool_call(name="first", arguments={})
    second = session.register_tool_call(name="second", arguments={})

    session.complete_tool_call(
        call_id=second.call_id,
        item_id="second-result",
        output="second finished",
    )

    assert session.pending_tool_call(first.call_id) == first
    assert session.pending_tool_call(second.call_id) is None
    session.complete_tool_call(
        call_id=first.call_id,
        item_id="first-result",
        output="first finished",
    )
    assert session.pending_tool_call(first.call_id) is None


def test_realtime_projects_structured_tool_call_events() -> None:
    protocol = NativeRealtimeSessionProtocol({})
    protocol._from_duplex_event(
        {"type": "response.created", "response_id": "response-1"}
    )
    item = {
        "id": "item-1",
        "object": "realtime.item",
        "type": "function_call",
        "status": "completed",
        "call_id": "call-1",
        "name": "get_current_time",
        "arguments": '{"timezone":"Europe/Copenhagen"}',
    }

    events = protocol._from_duplex_event(
        {
            "type": "response.tool_call.done",
            "response_id": "response-1",
            "item": item,
        }
    )
    done = protocol._from_duplex_event(
        {"type": "response.done", "response_id": "response-1"}
    )

    by_type = {event["type"]: event for event in events}
    assert by_type["response.function_call_arguments.done"] == {
        "type": "response.function_call_arguments.done",
        "response_id": "response-1",
        "item_id": "item-1",
        "output_index": 1,
        "call_id": "call-1",
        "name": "get_current_time",
        "arguments": '{"timezone":"Europe/Copenhagen"}',
    }
    response_done = next(event for event in done if event["type"] == "response.done")
    assert response_done["response"]["output"][1] == item


@pytest.mark.asyncio
async def test_realtime_translates_function_output_for_duplex_runtime() -> None:
    protocol = NativeRealtimeSessionProtocol({})
    translated = await protocol._to_duplex_event(
        {
            "type": "conversation.item.create",
            "item": {
                "id": "result-1",
                "type": "function_call_output",
                "call_id": "call-1",
                "output": '{"time":"18:30"}',
            },
        }
    )

    assert translated == {
        "type": "turn.signal",
        "event": "conversation.item.create",
        "payload": {
            "item": {
                "id": "result-1",
                "object": "realtime.item",
                "type": "function_call_output",
                "status": "completed",
                "call_id": "call-1",
                "output": '{"time":"18:30"}',
                "content": [],
            }
        },
    }
