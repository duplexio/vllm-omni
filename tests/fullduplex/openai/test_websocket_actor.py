# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm_omni.experimental.fullduplex.openai.websocket import (
    DuplexWebSocketActor,
    normalize_duplex_input_event,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(dict(payload))


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"type": "signal_turn", "event": "barge_in"}, {"type": "turn.signal", "event": "barge_in"}),
        ({"type": "close_session"}, {"type": "session.close"}),
        ({"type": "audio.playback_ack", "played_ms": 1}, {"type": "playback.ack", "played_ms": 1}),
        ({"type": "input_text.append", "text": "a"}, {"type": "input.text.append", "text": "a"}),
        ({"type": "push_text", "text": "b"}, {"type": "input.text.append", "text": "b"}),
        (
            {"type": "input.audio.append", "audio": "wav"},
            {"type": "input_audio_buffer.append", "audio": "wav", "format": "wav"},
        ),
        (
            {"type": "push_chunk", "audio": "wav", "format": "pcm_f32le"},
            {"type": "input_audio_buffer.append", "audio": "wav", "format": "pcm_f32le"},
        ),
    ],
)
def test_input_aliases_normalize_once_at_mailbox_boundary(event, expected):
    assert normalize_duplex_input_event(event) == expected


def test_actor_uses_one_fifo_mailbox_for_inbound_events():
    actor = DuplexWebSocketActor(FakeWebSocket())

    assert isinstance(actor.mailbox, asyncio.Queue)
    assert not isinstance(actor.mailbox, asyncio.PriorityQueue)
    assert not hasattr(actor, "control_queue")
    assert not hasattr(actor, "input_queue")
    assert not hasattr(actor, "event_queue")
    assert not hasattr(actor, "lifecycle_state")
    assert not hasattr(actor, "last_response_id")
    assert not hasattr(actor, "overlap_speech_ms")
    assert not hasattr(actor, "transition")
    assert not hasattr(actor, "output_generation_in_flight")
    assert not hasattr(actor, "session")
    assert not hasattr(actor, "runtime_opened")
    assert not hasattr(actor, "runtime_closed")
    assert not hasattr(actor, "drain_input_queue")
    assert not hasattr(actor, "control_events_seen")
    assert not hasattr(actor, "input_events_seen")
    assert not hasattr(actor, "cancel_count")


@pytest.mark.asyncio
async def test_control_event_does_not_overtake_earlier_audio_input():
    actor = DuplexWebSocketActor(FakeWebSocket())
    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "pcm"})
    await actor.enqueue_event({"type": "response.cancel"})

    assert await actor.next_event() == {"type": "input_audio_buffer.append", "audio": "pcm"}
    assert await actor.next_event() == {"type": "response.cancel"}


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["input.cancel", "session.close", "close_session"])
async def test_terminal_control_preserves_wire_order(event_type: str):
    actor = DuplexWebSocketActor(FakeWebSocket())
    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "pcm"})
    await actor.enqueue_event({"type": event_type})

    assert await actor.next_event() == {"type": "input_audio_buffer.append", "audio": "pcm"}
    assert await actor.next_event() == {"type": event_type}


@pytest.mark.asyncio
async def test_mailbox_delivers_every_enqueued_event_exactly_once():
    actor = DuplexWebSocketActor(FakeWebSocket())
    events = [
        {"type": "input_audio_buffer.append", "audio": str(index)}
        if index % 2
        else {"type": "response.created", "response_id": f"resp-{index}"}
        for index in range(20)
    ]

    await asyncio.gather(*(actor.enqueue_event(dict(event)) for event in events))
    received = [await actor.next_event() for _ in events]

    assert sorted(received, key=repr) == sorted(events, key=repr)


@pytest.mark.asyncio
async def test_writer_is_single_owner_of_websocket_send():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket)
    writer = asyncio.create_task(actor.writer_loop())

    await actor.send_json({"type": "one"})
    await actor.send_json({"type": "two"})
    await actor.close_writer()
    await writer

    assert websocket.sent == [{"type": "one"}, {"type": "two"}]


@pytest.mark.asyncio
async def test_stale_fence_payload_is_dropped_before_websocket_send():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket, current_epoch=lambda: 2)
    writer = asyncio.create_task(actor.writer_loop())

    await actor.send_json({"type": "response.audio.delta", "epoch": 1})
    await actor.send_json({"type": "response.audio.delta", "epoch": 2})
    await actor.close_writer()
    await writer

    assert websocket.sent == [{"type": "response.audio.delta", "epoch": 2}]
    assert actor.stale_output_dropped == 1


@pytest.mark.asyncio
async def test_writer_does_not_revoke_accepted_terminal_after_close_starts():
    websocket = FakeWebSocket()
    actor = DuplexWebSocketActor(websocket, current_epoch=lambda: 1)
    await actor.send_json({"type": "response.done", "epoch": 1, "response_id": "resp-1"})
    actor.closing = True
    await actor.close_writer()

    await actor.writer_loop()

    assert websocket.sent == [{"type": "response.done", "epoch": 1, "response_id": "resp-1"}]
    assert actor.stale_output_dropped == 0


@pytest.mark.asyncio
async def test_mailbox_backpressure_preserves_audio_order():
    actor = DuplexWebSocketActor(FakeWebSocket())
    assert actor.mailbox.maxsize > 0
    for index in range(actor.mailbox.maxsize):
        await actor.enqueue_event({"type": "input_audio_buffer.append", "index": index})
    blocked = asyncio.create_task(actor.enqueue_event({"type": "session.close"}))
    await asyncio.sleep(0)
    assert not blocked.done()
    assert (await actor.next_event())["index"] == 0
    await asyncio.wait_for(blocked, timeout=1)
    for index in range(1, actor.mailbox.maxsize):
        assert (await actor.next_event())["index"] == index
    assert (await actor.next_event())["type"] == "session.close"


@pytest.mark.asyncio
async def test_output_backpressure_preserves_audio_order():
    socket = FakeWebSocket()
    actor = DuplexWebSocketActor(socket)
    assert actor.output_queue.maxsize > 0
    for index in range(actor.output_queue.maxsize):
        await actor.send_json({"type": "response.audio.delta", "index": index})
    blocked = asyncio.create_task(actor.send_json({"type": "response.done"}))
    await asyncio.sleep(0)
    assert not blocked.done()
    writer = asyncio.create_task(actor.writer_loop())
    await asyncio.wait_for(blocked, timeout=1)
    await actor.close_writer()
    await asyncio.wait_for(writer, timeout=1)
    assert [event["index"] for event in socket.sent[:-1]] == list(range(actor.output_queue.maxsize))
    assert socket.sent[-1]["type"] == "response.done"


@pytest.mark.asyncio
async def test_mailbox_byte_budget_applies_before_model_input_buffer():
    actor = DuplexWebSocketActor(FakeWebSocket(), mailbox_byte_limit=8)
    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "aaaaaa"}, wire_bytes=6)
    blocked = asyncio.create_task(actor.enqueue_event({"type": "session.update"}, wire_bytes=4))
    await asyncio.sleep(0)
    assert not blocked.done()
    assert actor.mailbox_bytes == 6
    assert (await actor.next_event())["audio"] == "aaaaaa"
    await asyncio.wait_for(blocked, timeout=1)
    assert actor.mailbox_bytes == 4
    await actor.next_event()
    assert actor.mailbox_bytes == 0


@pytest.mark.asyncio
async def test_slow_receiver_closes_transport_and_wakes_session():
    class SlowWebSocket(FakeWebSocket):
        closed = None

        async def send_json(self, payload):
            raise TimeoutError

        async def close(self, *, code, reason):
            self.closed = (code, reason)

    socket = SlowWebSocket()
    actor = DuplexWebSocketActor(socket)
    await actor.send_json({"type": "response.audio.delta"})
    await asyncio.wait_for(actor.writer_loop(), timeout=1)
    assert socket.closed == (1013, "output_backpressure")
    assert actor.closing
    assert (await actor.next_event())["type"] == "__disconnect__"
