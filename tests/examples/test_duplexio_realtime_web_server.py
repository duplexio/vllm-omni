import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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


class SingleMessageBackend:
    def __init__(self) -> None:
        self.message_sent = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self.message_sent:
            raise StopAsyncIteration
        self.message_sent = True
        return '{"type":"session.ready"}'

    async def close(self) -> None:
        return None


class BackendConnection:
    def __init__(self, backend: SingleMessageBackend) -> None:
        self.backend = backend

    async def __aenter__(self) -> SingleMessageBackend:
        return self.backend

    async def __aexit__(self, *args: object) -> None:
        return None


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


def test_sampling_defaults_apply_serving_temperatures(tmp_path: Path) -> None:
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
        "audio": {"temperature": 0.7, "top_k": 250},
        "emit": {"user": 0.0, "agent": 1.0, "tool_call": 1.0},
    }


def test_sampling_defaults_omit_audio_for_continuous_checkpoint(
    tmp_path: Path,
) -> None:
    """A flow-map audio head takes no client sampling overrides."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "audio_representation": "continuous",
                "rollout_sampling_config": {
                    "mode": "top_p",
                    "temperature": 0.6,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                "flowmap_config": {"sampling_temperature": 0.3},
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
        "emit": {"user": 0.0, "agent": 1.0, "tool_call": 1.0},
    }


@pytest.mark.parametrize(
    ("healthy", "status_code", "body"),
    [(True, 200, "ok"), (False, 503, "unhealthy")],
)
def test_healthz_reports_backend_health(
    healthy: bool,
    status_code: int,
    body: str,
) -> None:
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        voice="custom",
        voices=[{"id": "custom", "label": "Voice 333 — custom"}],
        sampling={},
        health_check=lambda: healthy,
    )

    response = TestClient(app).get("/healthz")

    assert response.status_code == status_code
    assert response.text == body


def test_access_password_protects_page_and_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server,
        "password_matches",
        lambda password, password_hash: (
            password == "test-password" and password_hash == "test-verifier"
        ),
    )
    backend_is_ready = False

    def prepare_backend() -> None:
        nonlocal backend_is_ready
        backend_is_ready = True

    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        voice="custom",
        voices=[{"id": "custom", "label": "Voice 333 — custom"}],
        sampling={},
        password_hash="test-verifier",
        backend_headers={
            "Modal-Key": "test-key",
            "Modal-Secret": "test-secret",
        },
        backend_open_timeout=900,
        backend_ready=prepare_backend,
    )
    client = TestClient(app, base_url="https://testserver")

    index = client.get("/", follow_redirects=False)
    assert index.status_code == 303
    assert index.headers["location"] == "/login"

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/realtime"):
            pass
    assert exc_info.value.code == 1008

    rejected = client.post(
        "/login",
        data={"password": "wrong-password"},
        follow_redirects=False,
    )
    assert rejected.status_code == 401
    assert "Incorrect password." in rejected.text

    accepted = client.post(
        "/login",
        data={"password": "test-password"},
        follow_redirects=False,
    )
    assert accepted.status_code == 303
    assert accepted.headers["location"] == "/"
    assert "HttpOnly" in accepted.headers["set-cookie"]
    assert "SameSite=strict" in accepted.headers["set-cookie"]
    assert "Secure" in accepted.headers["set-cookie"]
    assert client.get("/").status_code == 200

    backend = SingleMessageBackend()

    def connect(*args: object, **kwargs: object) -> BackendConnection:
        assert backend_is_ready
        assert kwargs["additional_headers"] == {
            "Modal-Key": "test-key",
            "Modal-Secret": "test-secret",
        }
        assert kwargs["open_timeout"] == 900
        return BackendConnection(backend)

    monkeypatch.setattr(
        server.websockets,
        "connect",
        connect,
    )
    session_cookie = client.cookies.get(server.SESSION_COOKIE_NAME)
    with client.websocket_connect(
        "/v1/realtime",
        headers={"cookie": f"{server.SESSION_COOKIE_NAME}={session_cookie}"},
    ) as websocket:
        assert websocket.receive_text() == '{"type":"session.ready"}'


def test_healthz_remains_public_when_access_password_is_set() -> None:
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        voice="custom",
        voices=[{"id": "custom", "label": "Voice 333 — custom"}],
        sampling={},
        password_hash="test-verifier",
    )

    response = TestClient(app).get("/healthz")

    assert response.status_code == 200
    assert response.text == "ok"


def test_session_cookie_survives_frontend_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "password_matches", lambda password, password_hash: True)
    app_args = {
        "ws_backend": "wss://backend.example",
        "model": "checkpoint",
        "voice": "custom",
        "voices": [{"id": "custom", "label": "Voice 333 — custom"}],
        "sampling": {},
        "password_hash": "test-verifier",
    }
    first_client = TestClient(server.build_app(**app_args), base_url="https://testserver")
    first_client.post("/login", data={"password": "test-password"})
    session_cookie = first_client.cookies.get(server.SESSION_COOKIE_NAME)

    second_client = TestClient(server.build_app(**app_args), base_url="https://testserver")
    response = second_client.get(
        "/",
        headers={"cookie": f"{server.SESSION_COOKIE_NAME}={session_cookie}"},
        follow_redirects=False,
    )

    assert response.status_code == 200
