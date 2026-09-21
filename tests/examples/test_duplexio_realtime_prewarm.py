import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import websockets

PREWARM_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/online_serving/duplexio/realtime_web/prewarm.py"
)
spec = importlib.util.spec_from_file_location(
    "duplexio_realtime_prewarm_test",
    PREWARM_PATH,
)
assert spec is not None and spec.loader is not None
prewarm_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prewarm_module
spec.loader.exec_module(prewarm_module)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prewarm_waits_for_live_audio_before_closing() -> None:
    messages: list[dict[str, object]] = []

    async def scenario() -> None:
        async def handler(websocket) -> None:
            messages.append(json.loads(await websocket.recv()))
            await websocket.send(json.dumps({"type": "session.created"}))
            await websocket.send(json.dumps({"type": "session.updated"}))
            await websocket.send(json.dumps({"type": "response.audio.delta"}))
            for _ in range(3):
                messages.append(json.loads(await websocket.recv()))
                await websocket.send(json.dumps({"type": "response.audio.delta"}))
            messages.append(json.loads(await websocket.recv()))
            await websocket.send(json.dumps({"type": "session.closed"}))

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            await prewarm_module.prewarm(
                f"ws://127.0.0.1:{port}",
                "checkpoint",
                base64.b64encode(bytes(1_920 * 4)).decode(),
                [],
                timeout_seconds=5,
                warmup_frames=3,
            )

    asyncio.run(scenario())

    assert messages[0]["type"] == "session.update"
    for message in messages[1:4]:
        assert message["type"] == "input_audio_buffer.append"
        assert base64.b64decode(message["audio"]) == bytes(1_920 * 4)
        assert message["sample_rate_hz"] == 24_000
        assert message["format"] == "pcm_f32le"
    assert messages[4] == {"type": "session.close"}


def test_prewarm_supplies_reference_pcm_instead_of_a_voice_name() -> None:
    reference = base64.b64encode(bytes(1_920 * 4)).decode()
    session = prewarm_module.session_update("checkpoint", reference, [])["session"]
    assert "voice" not in session
    assert session["extra_body"]["ref_audio_data"] == reference
    assert session["extra_body"]["ref_audio_format"] == "pcm_f32le"
    assert session["extra_body"]["ref_audio_sample_rate"] == 24_000
