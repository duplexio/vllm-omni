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


def test_prewarm_waits_for_one_model_frame_and_clean_session_close() -> None:
    messages: list[dict[str, object]] = []

    async def scenario() -> None:
        async def handler(websocket) -> None:
            messages.append(json.loads(await websocket.recv()))
            await websocket.send(json.dumps({"type": "session.created"}))
            await websocket.send(json.dumps({"type": "session.updated"}))
            messages.append(json.loads(await websocket.recv()))
            await websocket.send(json.dumps({"type": "response.listen"}))
            messages.append(json.loads(await websocket.recv()))
            await websocket.send(json.dumps({"type": "session.closed"}))

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            await prewarm_module.prewarm(
                f"ws://127.0.0.1:{port}",
                "checkpoint",
                "voice-333",
                [],
                timeout_seconds=5,
            )

    asyncio.run(scenario())

    assert messages[0]["type"] == "session.update"
    assert messages[1]["type"] == "input_audio_buffer.append"
    assert len(base64.b64decode(messages[1]["audio"])) == 1_920 * 4
    assert messages[2] == {"type": "session.close"}
