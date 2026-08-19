"""Deploy the DuplexIO realtime server and web client on Modal."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

# Staging variant (DUPLEXIO_MODAL_STAGING=1): separate app + web labels,
# experimenting with memory snapshots of a genuinely SLEPT engine. The live
# app is completely unaffected unless the env var is set at deploy time.
STAGING = os.environ.get("DUPLEXIO_MODAL_STAGING") == "1"
APP_NAME = "duplexio-vllm-omni-staging" if STAGING else "duplexio-vllm-omni"
MODEL_VOLUME_NAME = "duplexio-vllm-models"
MODEL_NAME = "duplexio-489780-checkpoint-1-vllm-v3"
MODEL_PATH = Path("/models") / MODEL_NAME
# Default voice for prewarm + demo; must exist in the checkpoint voice
# pool (v3 ships the VoxCeleb id100xx set; id10014 is the eval voice).
VOICE = "id10014"
APP_ROOT = Path("/app/vllm-omni")
FRONTEND_ROOT = Path("/app/realtime_web")
DEPLOY_CONFIG_NAME = "duplexio.yaml"
KV_CACHE_MEMORY_BYTES = 19_947_344_692
MODEL_STARTUP_TIMEOUT_SECONDS = 15 * 60
BACKEND_PORT = 8099
BACKEND_HTTP_URL = f"http://127.0.0.1:{BACKEND_PORT}"
LOCAL_BACKEND_WEBSOCKET_URL = f"ws://127.0.0.1:{BACKEND_PORT}"
SNAPSHOT_STAGE_IDS = [0]
AUTH_SECRET_NAME = "DEMO_PASSWORD_HASH"
AUTH_PASSWORD_HASH_ENV = "DEMO_PASSWORD_HASH"
BACKEND_AUTH_SECRET_NAME = "duplexio-demo-backend-auth"
BACKEND_KEY_ENV = "DUPLEXIO_BACKEND_MODAL_KEY"
BACKEND_SECRET_ENV = "DUPLEXIO_BACKEND_MODAL_SECRET"
MODEL_WEB_LABEL = "model-snapshot-staging" if STAGING else "model-snapshot"
DEMO_WEB_LABEL = "demo-staging" if STAGING else "demo"
BACKEND_WEBSOCKET_URL = f"wss://duplexio--{MODEL_WEB_LABEL}.modal.run"
BACKEND_HEALTH_URL = f"https://duplexio--{MODEL_WEB_LABEL}.modal.run/healthz"

repo_root = Path(__file__).resolve().parents[3] if modal.is_local() else APP_ROOT
model_image = modal.Image.from_dockerfile(
    repo_root / "docker" / "Dockerfile.cuda",
    context_dir=repo_root,
    build_args={"BASE_IMAGE": "vllm/vllm-openai:v0.26.0"},
    ignore=[
        ".venv",
        "wandb",
        "examples/online_serving/duplexio/modal_app.py",
        "**/__pycache__",
        "**/.pytest_cache",
    ],
).env(
    {
        "PYTHONPATH": str(APP_ROOT),
        **(
            {
                "DUPLEXIO_MODAL_STAGING": "1",
                # TCP connections do not survive snapshot restore; the NCCL
                # heartbeat monitor then spams "Broken pipe" against the dead
                # rank-0 TCPStore forever. World-size-1 inference only loses
                # the flight recorder, so silence the monitor rather than
                # mask real errors under the spam.
                "TORCH_NCCL_ENABLE_MONITORING": "0",
            }
            if STAGING
            else {}
        ),
    }
)
frontend_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "bcrypt==4.3.0",
        "fastapi==0.136.3",
        "uvicorn==0.52.1",
        "websockets==17.0.1",
    )
    .add_local_dir(
        repo_root / "examples" / "online_serving" / "duplexio" / "realtime_web",
        remote_path=str(FRONTEND_ROOT),
        copy=True,
    )
    .env({"PYTHONPATH": "/app"})
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def backend_is_healthy(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return False
    try:
        with urllib.request.urlopen(f"{BACKEND_HTTP_URL}/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def wait_for_backend(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 12 * 60
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"vLLM exited during startup with code {exit_code}")
        if backend_is_healthy(process):
            return
        time.sleep(1)
    process.terminate()
    raise TimeoutError("vLLM did not become healthy within 12 minutes")


def backend_command(*, enable_sleep_mode: bool) -> list[str]:
    command = [
        "vllm-omni",
        "serve",
        str(MODEL_PATH),
        "--omni",
        "--deploy-config",
        str(APP_ROOT / "vllm_omni" / "deploy" / DEPLOY_CONFIG_NAME),
        "--stage-overrides",
        json.dumps({"0": {"kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES}}),
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(BACKEND_PORT),
    ]
    if enable_sleep_mode:
        command.append("--enable-sleep-mode")
    return command


def start_backend(*, enable_sleep_mode: bool) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join(
        filter(None, (str(APP_ROOT), environment.get("PYTHONPATH")))
    )
    environment["VLLM_USE_AOT_COMPILE"] = "0"
    if enable_sleep_mode:
        environment["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    backend = subprocess.Popen(
        backend_command(enable_sleep_mode=enable_sleep_mode),
        cwd=APP_ROOT,
        env=environment,
    )
    wait_for_backend(backend)
    return backend


def stop_backend(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def control_backend(path: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"{BACKEND_HTTP_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(
        request,
        timeout=MODEL_STARTUP_TIMEOUT_SECONDS,
    ) as response:
        result = json.load(response)
    if result.get("status") not in {"SUCCESS", "SKIPPED"}:
        raise RuntimeError(f"Unexpected response from {path}: {result!r}")
    return result


def prewarm_backend() -> None:
    subprocess.run(
        [
            sys.executable,
            str(
                APP_ROOT
                / "examples"
                / "online_serving"
                / "duplexio"
                / "realtime_web"
                / "prewarm.py"
            ),
            "--backend",
            LOCAL_BACKEND_WEBSOCKET_URL,
            "--model",
            str(MODEL_PATH),
            "--voice",
            VOICE,
            "--tools",
            str(
                APP_ROOT
                / "examples"
                / "online_serving"
                / "duplexio"
                / "realtime_web"
                / "tools.json"
            ),
            "--timeout-seconds",
            "300",
        ],
        check=True,
        timeout=6 * 60,
    )


def wait_for_remote_backend(url: str, headers: dict[str, str]) -> None:
    deadline = time.monotonic() + MODEL_STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=MODEL_STARTUP_TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    return
                last_error = RuntimeError(f"Backend health check returned HTTP {response.status}")
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Backend did not become ready within {MODEL_STARTUP_TIMEOUT_SECONDS}s") from last_error


def build_model_app(process: subprocess.Popen[bytes]) -> object:
    from examples.online_serving.duplexio.realtime_web.server import (
        build_app,
        load_sampling_defaults,
        load_voice_options,
    )

    tools_path = (
        APP_ROOT
        / "examples"
        / "online_serving"
        / "duplexio"
        / "realtime_web"
        / "tools.json"
    )
    return build_app(
        ws_backend=LOCAL_BACKEND_WEBSOCKET_URL,
        model=str(MODEL_PATH),
        voice=VOICE,
        voices=load_voice_options(MODEL_PATH / "voices.json"),
        sampling=load_sampling_defaults(MODEL_PATH / "config.json"),
        tools=json.loads(tools_path.read_text(encoding="utf-8")),
        health_check=lambda: backend_is_healthy(process),
    )


def check_model_present() -> None:
    if not MODEL_PATH.is_dir():
        raise RuntimeError(
            f"Model checkpoint not found at {MODEL_PATH}. Upload {MODEL_NAME} "
            f"to the {MODEL_VOLUME_NAME!r} Modal Volume."
        )


if STAGING:

    @app.cls(
        image=model_image,
        gpu="H100",
        volumes={"/models": model_volume.with_mount_options(read_only=True)},
        # Sleep level 1 offloads weights (~13 GB) + the kv_cache pool
        # (~20 GB) into host RAM before the snapshot captures it.
        memory=96_000,
        timeout=24 * 60 * 60,
        startup_timeout=MODEL_STARTUP_TIMEOUT_SECONDS,
        scaledown_window=5 * 60,
        max_containers=1,
        # The ORIGINAL (2026-08-11, 58c2721) snapshot design, verbatim:
        # sleep-mode CuMem pools engage at weight load, prewarm exercises the
        # full path, sleep level 1 offloads weights to CPU pools and discards
        # the rest of GPU memory, THEN memory + GPU snapshot capture the slept
        # engine. That deployment served correct audio through restores; the
        # live app's garbled restores coincided with workers logging "Sleep
        # Mode DISABLED" (snapshot of a fully LIVE engine). The flag plumbing
        # is code-identical 58c2721..HEAD and engages on the cluster via this
        # exact CLI, so staging first re-tests the proven design end-to-end.
        enable_memory_snapshot=True,
        experimental_options={"enable_gpu_snapshot": True},
    )
    @modal.concurrent(max_inputs=100)
    class SnapshotModelServer:
        @modal.enter(snap=True)
        def start_and_sleep(self) -> None:
            check_model_present()
            self.backend = start_backend(enable_sleep_mode=True)
            prewarm_backend()
            control_backend(
                "/v1/omni/sleep",
                {"stage_ids": SNAPSHOT_STAGE_IDS, "level": 1},
            )
            print("[staging] engine slept (weights offloaded); snapshotting")

        @modal.enter(snap=False)
        def wake(self) -> None:
            wake_started = time.monotonic()
            control_backend(
                "/v1/omni/wakeup",
                {"stage_ids": SNAPSHOT_STAGE_IDS},
            )
            wait_for_backend(self.backend)
            print(
                f"[staging] engine woke in {time.monotonic() - wake_started:.1f}s"
            )

        @modal.exit()
        def stop(self) -> None:
            stop_backend(self.backend)

        # Staging is non-production: no proxy auth so the remote
        # agent-speech gate can drive it directly.
        @modal.asgi_app(label=MODEL_WEB_LABEL, requires_proxy_auth=False)
        def web(self) -> object:
            return build_model_app(self.backend)

else:

    @app.cls(
        image=model_image,
        gpu="H100",
        volumes={"/models": model_volume.with_mount_options(read_only=True)},
        memory=32_768,
        timeout=24 * 60 * 60,
        startup_timeout=MODEL_STARTUP_TIMEOUT_SECONDS,
        scaledown_window=10 * 60,
        max_containers=1,
        # Snapshots stay OFF: A/B-confirmed (2026-08-19) that GPU-snapshot
        # restore garbles agent audio (fluent mumble, text unaffected). The
        # pre-snapshot sleep also silently no-ops (workers log "Sleep Mode
        # DISABLED"), so the snapshot captured a fully live engine. Do not
        # re-enable without passing a remote agent-speech gate on a restored
        # container.
        enable_memory_snapshot=False,
    )
    @modal.concurrent(max_inputs=100)
    class SnapshotModelServer:
        # Plain start (no snapshot, no sleep/wakeup) — see the snapshot note
        # on the class decorator for why.
        @modal.enter()
        def start(self) -> None:
            check_model_present()
            self.backend = start_backend(enable_sleep_mode=False)
            prewarm_backend()
            wait_for_backend(self.backend)

        @modal.exit()
        def stop(self) -> None:
            stop_backend(self.backend)

        @modal.asgi_app(label="model-snapshot", requires_proxy_auth=True)
        def web(self) -> object:
            return build_model_app(self.backend)


@app.function(
    image=frontend_image,
    volumes={"/models": model_volume.with_mount_options(read_only=True)},
    memory=1024,
    timeout=24 * 60 * 60,
    scaledown_window=60,
    max_containers=1,
    secrets=[
        modal.Secret.from_name(AUTH_SECRET_NAME),
        modal.Secret.from_name(BACKEND_AUTH_SECRET_NAME),
    ],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label=DEMO_WEB_LABEL)
def demo() -> object:
    """Serve authentication and proxy authenticated streams to the GPU."""
    from realtime_web.server import (
        build_app,
        load_sampling_defaults,
        load_voice_options,
    )

    tools_path = FRONTEND_ROOT / "tools.json"
    backend_headers = {
        "Modal-Key": os.environ[BACKEND_KEY_ENV],
        "Modal-Secret": os.environ[BACKEND_SECRET_ENV],
    }
    return build_app(
        ws_backend=BACKEND_WEBSOCKET_URL,
        model=str(MODEL_PATH),
        voice=VOICE,
        voices=load_voice_options(MODEL_PATH / "voices.json"),
        sampling=load_sampling_defaults(MODEL_PATH / "config.json"),
        tools=json.loads(tools_path.read_text(encoding="utf-8")),
        password_hash=os.environ[AUTH_PASSWORD_HASH_ENV],
        backend_headers=backend_headers,
        backend_open_timeout=MODEL_STARTUP_TIMEOUT_SECONDS,
        backend_ready=lambda: wait_for_remote_backend(
            BACKEND_HEALTH_URL,
            backend_headers,
        ),
    )
