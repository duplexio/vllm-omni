import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

REALTIME_WEB = Path(__file__).resolve().parents[2] / "examples/online_serving/duplexio/realtime_web"
SERVER_PATH = REALTIME_WEB / "server.py"
# server.py imports the relay from beside itself, as it does when run as a script.
sys.path.insert(0, str(REALTIME_WEB))
import realtime_relay  # noqa: E402

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


def test_sample_clips_are_listed_for_the_upload_control(tmp_path: Path) -> None:
    (tmp_path / "bravo_voice.wav").write_bytes(b"RIFF")
    (tmp_path / "alpha_voice.flac").write_bytes(b"fLaC")
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")

    assert server.list_sample_clips(tmp_path) == [
        {
            "id": "alpha_voice.flac",
            "label": "alpha voice",
            "url": "/voices/alpha_voice.flac",
        },
        {
            "id": "bravo_voice.wav",
            "label": "bravo voice",
            "url": "/voices/bravo_voice.wav",
        },
    ]
    # A deployment may ship none: the page still uploads a clip.
    assert server.list_sample_clips(None) == []


def test_voices_serves_only_listed_clips(tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "alice.wav").write_bytes(b"RIFF")
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "alice.wav").symlink_to(tmp_path / "shared" / "alice.wav")
    (tmp_path / "export" / "model.safetensors").write_bytes(b"weights")
    client = TestClient(
        server.build_app(
            ws_backend="ws://127.0.0.1:8099",
            model="checkpoint",
            sample_clips=server.list_sample_clips(tmp_path / "export"),
            sample_clip_dir=tmp_path / "export",
            sampling={},
        )
    )

    assert client.get("/voices/alice.wav").content == b"RIFF"
    assert client.get("/voices/model.safetensors").status_code == 404


def test_index_exposes_sample_clips() -> None:
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        sample_clips=[{"id": "alice.wav", "label": "alice", "url": "/voices/alice.wav"}],
        sample_clip_dir=None,
        default_voice="alice.wav",
        sampling={
            "text": {
                "mode": "top_p",
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
            },
            "audio": {"temperature": 0.8},
            "emit": {"agent": 0.6, "tool_call": 0.6},
        },
    )

    index = TestClient(app).get("/")

    assert index.status_code == 200
    assert '"sampleClips"' in index.text
    assert '"url": "/voices/alice.wav"' in index.text
    assert '"defaultVoice": "alice.wav"' in index.text
    assert '"sampling": {"text": {"mode": "top_p"' in index.text
    assert '"realtimePath": "v1/realtime"' in index.text
    assert '"realtimeTicketPath": null' in index.text


def test_index_points_the_page_at_a_direct_relay() -> None:
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        sample_clips=[],
        sample_clip_dir=None,
        sampling={},
        realtime_url="wss://engine.example/v1/realtime",
    )

    assert '"realtimePath": "wss://engine.example/v1/realtime"' in TestClient(app).get("/").text


def test_sampling_defaults_are_the_engine_defaults_and_the_checkpoint_audio_temperature(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "flowmap_config": {"sampling_temperature": 0.3},
            }
        ),
        encoding="utf-8",
    )

    assert server.load_sampling_defaults(config_path) == {
        "agent": {"emission": {"temperature": 1.0}, "content": {"temperature": 0.6, "top_k": 20, "top_p": 0.95}},
        "user": {"emission": {"temperature": 0.0}, "content": {"temperature": 0.0, "top_k": None, "top_p": None}},
        "audio": {"temperature": 0.3},
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
        sample_clips=[{"id": "alice.wav", "label": "alice", "url": "/voices/alice.wav"}],
        sample_clip_dir=None,
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
        lambda password, password_hash: password == "test-password" and password_hash == "test-verifier",
    )
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        sample_clips=[{"id": "alice.wav", "label": "alice", "url": "/voices/alice.wav"}],
        sample_clip_dir=None,
        sampling={},
        password_hash="test-verifier",
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


def test_signed_in_pages_get_tickets_for_a_direct_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "password_matches", lambda password, password_hash: True)
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        sample_clips=[],
        sample_clip_dir=None,
        sampling={},
        password_hash="test-verifier",
        realtime_url="wss://engine.example/v1/realtime",
    )
    client = TestClient(app, base_url="https://testserver")

    assert client.get("/v1/realtime/ticket").status_code == 401
    client.post("/login", data={"password": "test-password"}, follow_redirects=False)
    assert '"realtimeTicketPath": "v1/realtime/ticket"' in client.get("/").text
    ticket = client.get("/v1/realtime/ticket").json()["ticket"]

    key = realtime_relay.ticket_key("test-verifier")
    assert realtime_relay.ticket_is_valid(key, ticket)
    assert not realtime_relay.ticket_is_valid(realtime_relay.ticket_key("other-verifier"), ticket)


def test_tickets_expire_and_resist_forgery() -> None:
    key = realtime_relay.ticket_key("test-verifier")
    ticket = realtime_relay.issue_ticket(key, now=1_000)
    expiry, _, signature = ticket.partition(".")

    assert realtime_relay.ticket_is_valid(key, ticket, now=1_000 + realtime_relay.TICKET_LIFETIME_SECONDS - 1)
    assert not realtime_relay.ticket_is_valid(key, ticket, now=1_000 + realtime_relay.TICKET_LIFETIME_SECONDS)
    assert not realtime_relay.ticket_is_valid(key, f"{int(expiry) + 3600}.{signature}", now=1_000)
    for forged in (None, "", "garbage", f"{expiry}."):
        assert not realtime_relay.ticket_is_valid(key, forged, now=1_000)


def test_tickets_and_codec_stay_out_of_the_engine_query() -> None:
    query, opus = realtime_relay.backend_query([("duplex", "1"), ("codec", "opus"), ("ticket", "1.abc")])

    assert query == "duplex=1"
    assert opus is not None


def test_healthz_remains_public_when_access_password_is_set() -> None:
    app = server.build_app(
        ws_backend="ws://127.0.0.1:8099",
        model="checkpoint",
        sample_clips=[{"id": "alice.wav", "label": "alice", "url": "/voices/alice.wav"}],
        sample_clip_dir=None,
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
        "sample_clips": [{"id": "alice.wav", "label": "alice", "url": "/voices/alice.wav"}],
        "sample_clip_dir": None,
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


def opus_packets(samples: np.ndarray, frame_ms: int) -> list[bytes]:
    encoder = av.CodecContext.create("libopus", "w")
    encoder.sample_rate = realtime_relay.INPUT_SAMPLE_RATE
    encoder.layout = "mono"
    encoder.format = "s16"
    encoder.bit_rate = realtime_relay.OPUS_BITRATE
    encoder.options = {"frame_duration": str(frame_ms)}
    encoder.open()
    packets = []
    for offset in range(0, samples.size, encoder.frame_size):
        frame = av.AudioFrame.from_ndarray(
            samples[None, offset : offset + encoder.frame_size], format="s16", layout="mono"
        )
        frame.sample_rate = realtime_relay.INPUT_SAMPLE_RATE
        frame.pts = offset
        packets += [bytes(packet) for packet in encoder.encode(frame)]
    return packets


def decode_opus(packets: list[bytes]) -> np.ndarray:
    decoder = av.CodecContext.create("opus", "r")
    resampler = av.AudioResampler(format="flt", layout="mono", rate=realtime_relay.INPUT_SAMPLE_RATE)
    return np.concatenate(
        [
            frame.to_ndarray().reshape(-1)
            for packet in packets
            for decoded in decoder.decode(av.Packet(packet))
            for frame in resampler.resample(decoded)
        ]
    )


def tone(seconds: float) -> np.ndarray:
    time = np.arange(int(seconds * realtime_relay.INPUT_SAMPLE_RATE)) / realtime_relay.INPUT_SAMPLE_RATE
    return (0.3 * 32767 * np.sin(2 * np.pi * 440 * time)).astype(np.int16)


def matches_tone(pcm: np.ndarray, reference: np.ndarray) -> bool:
    # Opus delays audio by its lookahead; compare after aligning on the best lag.
    lags = range(0, 400)
    scores = [np.corrcoef(pcm[lag : lag + 4_000], reference[:4_000])[0, 1] for lag in lags]
    return max(scores) > 0.99


def test_opus_microphone_packets_reach_the_backend_as_pcm() -> None:
    transcoder = realtime_relay.OpusTranscoder()
    reference = tone(0.4)
    appends = [
        json.loads(
            transcoder.to_backend(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(packet).decode(),
                        "format": "opus",
                    }
                )
            )
        )
        for packet in opus_packets(reference, frame_ms=20)
    ]

    assert {(event["format"], event["sample_rate_hz"]) for event in appends} == {("pcm_f32le", 24_000)}
    pcm = np.concatenate([np.frombuffer(base64.b64decode(event["audio"]), "<f4") for event in appends])
    assert abs(pcm.size - reference.size) < 480
    assert matches_tone(pcm, reference / 32768)
    other = '{"type":"session.update"}'
    assert transcoder.to_backend(other) == other


def test_model_audio_reaches_the_page_as_opus_packets() -> None:
    transcoder = realtime_relay.OpusTranscoder()
    reference = tone(0.4)
    events = []
    for offset in range(0, 1_920 * 4, 1_920):
        events += transcoder.to_client(
            json.dumps(
                {
                    "type": "response.output_audio.delta",
                    "response_id": "resp_1",
                    "delta": base64.b64encode(reference[offset : offset + 1_920].tobytes()).decode(),
                    "format": "pcm_s16le",
                    "sample_rate_hz": 24_000,
                    "metadata": {"playback": {"generated_ms": 80}},
                }
            )
        )
    # A trailing partial frame is flushed, padded, ahead of the done event.
    events += transcoder.to_client(
        json.dumps(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_1",
                "delta": base64.b64encode(reference[1_920 * 4 :].tobytes()).decode(),
            }
        )
    )
    events += transcoder.to_client('{"type":"response.output_audio.done","response_id":"resp_1"}')

    deltas = [json.loads(event) for event in events[:-1]]
    assert [delta["type"] for delta in deltas] == ["response.output_audio.delta"] * 5
    assert {(delta["format"], delta["response_id"]) for delta in deltas} == {("opus", "resp_1")}
    assert not any("metadata" in delta for delta in deltas)
    assert json.loads(events[-1])["type"] == "response.output_audio.done"
    pcm = decode_opus([base64.b64decode(delta["delta"]) for delta in deltas])
    assert matches_tone(pcm, reference / 32768)


def test_proxy_transcodes_only_sessions_that_ask_for_opus(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingBackend(SingleMessageBackend):
        def __init__(self) -> None:
            super().__init__()
            self.sent: list[str] = []

        async def __anext__(self) -> str:
            if self.message_sent:
                raise StopAsyncIteration
            self.message_sent = True
            await asyncio.sleep(0.2)
            return json.dumps(
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(tone(0.08).tobytes()).decode(),
                }
            )

        async def send(self, message: str) -> None:
            self.sent.append(message)

    backend = RecordingBackend()
    urls = []

    def connect(url: str, **kwargs: object) -> BackendConnection:
        urls.append(url)
        return BackendConnection(backend)

    monkeypatch.setattr(server.websockets, "connect", connect)
    app = server.build_app(
        ws_backend="ws://backend", model="checkpoint", sample_clips=[], sample_clip_dir=None, sampling={}
    )
    packet = opus_packets(tone(0.02), frame_ms=20)[0]
    with TestClient(app).websocket_connect("/v1/realtime?duplex=1&codec=opus") as websocket:
        websocket.send_text(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(packet).decode(),
                    "format": "opus",
                }
            )
        )
        delta = json.loads(websocket.receive_text())

    assert urls == ["ws://backend/v1/realtime?duplex=1"]
    assert json.loads(backend.sent[0])["format"] == "pcm_f32le"
    assert delta["format"] == "opus"


def test_index_versions_scripts_by_their_current_content(tmp_path, monkeypatch) -> None:
    # Pulling new page code must reach browsers without restarting the server.
    static = tmp_path / "static"
    for name in ("app.js", "capture_worklet.js", "playback_worklet.js", "recording_worklet.js"):
        static.mkdir(exist_ok=True)
        (static / name).write_text(f"// {name}\n")
    monkeypatch.setattr(server, "STATIC_DIR", static)
    client = TestClient(
        server.build_app(
            ws_backend="ws://127.0.0.1:8099",
            model="checkpoint",
            sample_clips=[],
            sample_clip_dir=None,
            sampling={},
        )
    )

    def script_version() -> str:
        return client.get("/").text.split("static/app.js?v=", 1)[1].split('"', 1)[0]

    before = script_version()
    (static / "app.js").write_text("// updated\n")
    assert script_version() != before
