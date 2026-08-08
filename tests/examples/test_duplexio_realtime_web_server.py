import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/online_serving/duplexio/realtime_web/server.py"
)
spec = importlib.util.spec_from_file_location(
    "duplexio_realtime_web_server_test",
    SERVER_PATH,
)
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_voice_options_follow_checkpoint_tensor_numbers(tmp_path: Path) -> None:
    manifest_path = tmp_path / "voices.json"
    manifest_path.write_text(
        json.dumps(
            {
                "voices": {
                    "custom": {"tensor": "voice.332"},
                    "first": {"tensor": "voice.0"},
                }
            }
        ),
        encoding="utf-8",
    )

    assert server.load_voice_options(manifest_path) == [
        {"id": "first", "label": "Voice 1 — first"},
        {"id": "custom", "label": "Voice 333 — custom"},
    ]


def test_index_exposes_voice_options_and_default() -> None:
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        voice="custom",
        voices=[{"id": "custom", "label": "Voice 333 — custom"}],
        sampling={
            "text": {
                "mode": "top_p",
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
            },
            "audio": {"temperature": 0.8, "top_k": 250},
            "emit": {"user": 0.0, "agent": 0.6, "tool_call": 0.6},
        },
    )

    index = TestClient(app).get("/")

    assert index.status_code == 200
    assert '"voice": "custom"' in index.text
    assert '"label": "Voice 333 \\u2014 custom"' in index.text
    assert '"sampling": {"text": {"mode": "top_p"' in index.text


def test_sampling_defaults_follow_checkpoint_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "rollout_sampling_config": {
                    "mode": "top_p",
                    "temperature": 0.6,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                "depth_transformer_config": {
                    "sampling_temperature": 0.8,
                    "sampling_top_k": 250,
                },
            }
        ),
        encoding="utf-8",
    )

    assert server.load_sampling_defaults(config_path) == {
        "text": {
            "mode": "top_p",
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        },
        "audio": {"temperature": 0.8, "top_k": 250},
        "emit": {"user": 0.0, "agent": 0.6, "tool_call": 1.0},
    }
