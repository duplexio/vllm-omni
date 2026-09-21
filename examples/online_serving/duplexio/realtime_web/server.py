"""Small local web host and same-origin Realtime WebSocket proxy for DuplexIO."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).parent / "app"
STATIC_DIR = APP_DIR / "static"
INPUT_SAMPLE_RATE = 24_000
DEFAULT_TOOLS_PATH = Path(__file__).parent / "tools.json"
SESSION_COOKIE_NAME = "__Host-duplexio_session"
DEFAULT_SAMPLING = {
    "agent": {
        "emission": {"temperature": 1.0},
        "content": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
    },
    "user": {
        "emission": {"temperature": 0.0},
        "content": {"temperature": 0.0, "top_k": None, "top_p": None},
    },
}




class DepthSamplingDefaults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sampling_temperature: float = Field(gt=0)
    sampling_top_k: int = Field(ge=1)


class CheckpointSamplingDefaults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Only a discrete (Mimi depth-transformer) checkpoint has a client-tunable
    # audio sampler. A continuous flow-map head samples at the temperature
    # baked into its checkpoint, so it exports no depth transformer config.
    depth_transformer_config: DepthSamplingDefaults | None = None


def join_ws_url(base: str, path: str, query: str) -> str:
    return base.rstrip("/") + path + (("?" + query) if query else "")


def list_sample_clips(directory: Path | None) -> list[dict[str, str]]:
    """Offer the sample reference clips a browser can pick instead of uploading.

    A voice is audio the client sends, so these are only convenience assets: the
    page reads whichever clip is chosen, resamples it, and puts it in the session
    as reference audio. No clips is a valid deployment — the page still uploads.
    """
    if directory is None:
        return []
    if not directory.is_dir():
        raise ValueError(f"DuplexIO sample clip directory not found: {directory}")
    return [
        {"id": path.name, "label": path.stem.replace("_", " "), "url": f"/voices/{path.name}"}
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
    ]


def load_sampling_defaults(config_path: Path) -> dict[str, object]:
    checkpoint = CheckpointSamplingDefaults.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    depth = checkpoint.depth_transformer_config
    return {
        **DEFAULT_SAMPLING,
        **(
            {}
            if depth is None
            else {"audio": {"temperature": 0.7, "top_k": depth.sampling_top_k}}
        ),
    }


async def pump_client_to_backend(client: WebSocket, backend) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                await backend.close()
                return
            if message.get("text") is not None:
                await backend.send(message["text"])
            elif message.get("bytes") is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def pump_backend_to_client(client: WebSocket, backend) -> None:
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                await client.send_text(message)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        return
    except RuntimeError as exc:
        if "websocket.send" in str(exc) and "websocket.close" in str(exc):
            return
        raise


def expected_proxy_close(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed, asyncio.CancelledError)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.send" in str(exc) and "websocket.close" in str(exc)


def password_matches(password: str, password_hash: str) -> bool:
    import bcrypt

    return bcrypt.checkpw(password.encode(), password_hash.encode())


def build_app(
    *,
    ws_backend: str,
    model: str,
    sample_clips: list[dict[str, str]],
    sample_clip_dir: Path | None,
    sampling: dict[str, object],
    tools: list[dict[str, object]] | None = None,
    health_check: Callable[[], bool] | None = None,
    password_hash: str | None = None,
    backend_headers: dict[str, str] | None = None,
    backend_open_timeout: float | None = 10,
    backend_ready: Callable[[], None] | None = None,
) -> FastAPI:
    app = FastAPI(title="duplexio")
    index_path = APP_DIR / "index.html"
    login_path = APP_DIR / "login.html"
    session_token = (
        hmac.digest(
            password_hash.encode(),
            b"duplexio browser session v1",
            "sha256",
        ).hex()
        if password_hash is not None
        else None
    )
    app_version_hash = hashlib.sha256()
    for asset_path in (
        STATIC_DIR / "app.js",
        STATIC_DIR / "capture_worklet.js",
        STATIC_DIR / "playback_worklet.js",
        STATIC_DIR / "recording_worklet.js",
    ):
        app_version_hash.update(asset_path.read_bytes())
    app_version = app_version_hash.hexdigest()[:12]

    def is_authenticated(cookie: str | None) -> bool:
        return session_token is None or (
            cookie is not None and hmac.compare_digest(cookie, session_token)
        )

    def login_page_response(*, invalid_password: bool = False) -> HTMLResponse:
        error = "Incorrect password." if invalid_password else ""
        html = login_path.read_text(encoding="utf-8").replace(
            "__DUPLEXIO_LOGIN_ERROR__",
            error,
        )
        return HTMLResponse(
            html,
            status_code=401 if invalid_password else 200,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        if not is_authenticated(request.cookies.get(SESSION_COOKIE_NAME)):
            return RedirectResponse("/login", status_code=303)
        config = json.dumps(
            {
                "model": model,
                "sampleClips": sample_clips,
                "sampling": sampling,
                "inputSampleRate": INPUT_SAMPLE_RATE,
                "realtimePath": "v1/realtime",
                "appVersion": app_version,
                "tools": tools or [],
            },
            ensure_ascii=True,
        )
        html = (
            index_path.read_text(encoding="utf-8")
            .replace("__DUPLEXIO_CONFIG__", config)
            .replace("__DUPLEXIO_APP_VERSION__", app_version)
        )
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request) -> Response:
        if is_authenticated(request.cookies.get(SESSION_COOKIE_NAME)):
            return RedirectResponse("/", status_code=303)
        return login_page_response()

    @app.post("/login")
    async def login(request: Request) -> Response:
        if password_hash is None or session_token is None:
            return RedirectResponse("/", status_code=303)
        form = parse_qs((await request.body()).decode("utf-8", errors="replace"))
        supplied_password = form.get("password", [""])[0]
        if not await asyncio.to_thread(
            password_matches,
            supplied_password,
            password_hash,
        ):
            return login_page_response(invalid_password=True)

        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_token,
            secure=True,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return response

    @app.get("/healthz")
    def healthz() -> Response:
        healthy = health_check is None or health_check()
        return Response(
            content="ok" if healthy else "unhealthy",
            media_type="text/plain",
            status_code=200 if healthy else 503,
        )

    @app.websocket("/v1/realtime")
    async def realtime_proxy(websocket: WebSocket) -> None:
        if not is_authenticated(websocket.cookies.get(SESSION_COOKIE_NAME)):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        query = urlencode(websocket.query_params.multi_items())
        backend_url = join_ws_url(ws_backend, "/v1/realtime", query)
        logger.info("Proxying Realtime WebSocket to %s", backend_url)
        try:
            if backend_ready is not None:
                await asyncio.to_thread(backend_ready)
            async with websockets.connect(
                backend_url,
                additional_headers=backend_headers,
                max_size=64 * 1024 * 1024,
                open_timeout=backend_open_timeout,
            ) as backend:
                tasks = {
                    asyncio.create_task(pump_client_to_backend(websocket, backend)),
                    asyncio.create_task(pump_backend_to_client(websocket, backend)),
                }
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    if isinstance(result, BaseException) and not expected_proxy_close(result):
                        raise result
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            return
        except Exception:
            logger.exception("Realtime WebSocket proxy failed")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if sample_clip_dir is not None:
        app.mount(
            "/voices",
            StaticFiles(directory=str(sample_clip_dir)),
            name="voices",
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--ws-backend", default="ws://127.0.0.1:8099")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--sample-clips",
        type=Path,
        default=None,
        help="Directory of reference clips the page offers beside its upload control.",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        default=DEFAULT_TOOLS_PATH,
        help="JSON file containing OpenAI function tool definitions",
    )
    args = parser.parse_args()

    tools = json.loads(args.tools.read_text(encoding="utf-8"))
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        parser.error("--tools must contain a JSON list of tool definitions")
    sample_clips = list_sample_clips(args.sample_clips)
    sampling = load_sampling_defaults(Path(args.model) / "config.json")

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        build_app(
            ws_backend=args.ws_backend,
            model=args.model,
            sample_clips=sample_clips,
            sample_clip_dir=args.sample_clips,
            sampling=sampling,
            tools=tools,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
