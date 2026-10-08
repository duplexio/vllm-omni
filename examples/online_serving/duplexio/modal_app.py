"""Deploy the DuplexIO realtime server and web client on Modal.

Upload the exported checkpoint and a mono 24 kHz ``prewarm.wav`` into
``MODEL_NAME`` on the ``duplexio-vllm-models`` volume. The startup clip warms
inference. Clips in its ``voices/`` subdirectory are offered in the browser
(``DEFAULT_VOICE`` preselected); users can also upload their own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

# A variant name deploys a parallel app under suffixed web labels
# (DUPLEXIO_MODAL_VARIANT=paged serves https://duplexio--demo-paged.eu-west.modal.run).
# The live demo is unaffected unless the variable is unset at deploy time.
VARIANT = os.environ.get("DUPLEXIO_MODAL_VARIANT", "")
SUFFIX = f"-{VARIANT}" if VARIANT else ""
# Every image that imports this module has to agree with the deploying client on
# the variant: web labels and the frontend's backend URL are recomputed from it
# when a container imports the module.
VARIANT_ENV = {"DUPLEXIO_MODAL_VARIANT": VARIANT} if VARIANT else {}
APP_NAME = f"duplexio-vllm-omni{SUFFIX}"
MODEL_VOLUME_NAME = "duplexio-vllm-models"
MODEL_NAME = "run514428_step5000_v7_steps8"
MODEL_PATH = Path("/models") / MODEL_NAME
# Upload a mono 24 kHz clip alongside the model for startup warmup.
# Browser sessions send their own uploaded reference audio.
PREWARM_AUDIO_PATH = MODEL_PATH / "prewarm.wav"
# Voices sit in their own directory so the page offers them and not prewarm.wav.
VOICE_DIR = MODEL_PATH / "voices"
DEFAULT_VOICE = "maya.wav"
APP_ROOT = Path("/app/vllm-omni")
FRONTEND_ROOT = Path("/app/realtime_web")
RELAY_DIR = APP_ROOT / "examples" / "online_serving" / "duplexio" / "realtime_web"
DEPLOY_CONFIG_NAME = "duplexio-realtime-h100.yaml"
MODEL_STARTUP_TIMEOUT_SECONDS = 15 * 60
# The GPU and the page proxy run next to each other in Europe, where the demo's
# users are: unpinned, audio crossed the Atlantic twice a frame.
REGION = "eu"
ROUTING_REGION = "eu-west"
BACKEND_PORT = 8099
BACKEND_HTTP_URL = f"http://127.0.0.1:{BACKEND_PORT}"
LOCAL_BACKEND_WEBSOCKET_URL = f"ws://127.0.0.1:{BACKEND_PORT}"
SNAPSHOT_STAGE_IDS = [0]
AUTH_SECRET_NAME = "DEMO_PASSWORD_HASH"
AUTH_PASSWORD_HASH_ENV = "DEMO_PASSWORD_HASH"
MODEL_WEB_LABEL = f"model-snapshot{SUFFIX}"
DEMO_WEB_LABEL = f"demo{SUFFIX}"
# A routing region puts the region in every web endpoint's hostname.
BACKEND_HOST = f"duplexio--{MODEL_WEB_LABEL}.{ROUTING_REGION}.modal.run"
BACKEND_WEBSOCKET_URL = f"wss://{BACKEND_HOST}"

repo_root = Path(__file__).resolve().parents[3] if modal.is_local() else APP_ROOT


# Serving runs the stack the cluster and the RTX 5090 demo serve on: torch
# 2.11+cu130, the PyPI vLLM 0.26.0 wheel (a CUDA 13 build pinned to that torch),
# and the quack/FLA kernels the model calls directly.
model_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "ninja-build")
    .uv_pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install("vllm==0.26.0")
    .add_local_dir(
        repo_root,
        str(APP_ROOT),
        copy=True,
        # Serving needs the source tree only. The working copy also holds many
        # gigabytes of local artifacts (exports, run logs, captured tensors)
        # that would otherwise be uploaded into every image layer.
        ignore=[
            ".venv",
            "wandb",
            "checkpoints",
            "logs",
            "probe_dumps",
            "rollout_layers_v9",
            "**/*.pt",
            "**/__pycache__",
            "**/.pytest_cache",
            # Serving never reads these, and any change to the image rebuilds
            # its GPU snapshot (a ~20 minute cold start).
            "tests",
            "docs",
            "benchmarks",
            # The page and its proxy run in the demo app. Leaving them out keeps
            # page changes from rebuilding this image and its GPU snapshot. The
            # relay they share with this container stays.
            "examples/online_serving/duplexio/realtime_web/app",
            "examples/online_serving/duplexio/realtime_web/server.py",
        ],
    )
    # Installed from inside the image so the requirements file resolves relative
    # to itself, and so vllm-omni's own pins cannot move torch. common.txt, not
    # cuda.txt: the cuda extras (fa3-fwd, onnxruntime) serve the diffusion and
    # other-model paths, and are absent from the validated training venv.
    .run_commands(
        f"python -m pip install --no-cache-dir -r {APP_ROOT}/requirements/common.txt",
    )
    # Last, so the kernel and transformers pins win over any resolution above.
    .uv_pip_install(
        "nvidia-cutlass-dsl==4.6.0",
        "quack-kernels==0.6.3",
        "flash-linear-attention==0.5.1",
        "fla-core==0.5.1",
        "transformers==5.14.1",
    )
    .env(
        {
            "PYTHONPATH": f"{APP_ROOT}:{RELAY_DIR}",
            # TCP connections do not survive snapshot restore (the container IP
            # changes), so the NCCL monitor thread's periodic flight-recorder
            # dump-flag poll spams "Broken pipe" against the dead rank-0
            # TCPStore forever. The monitor thread always runs regardless of
            # ENABLE_MONITORING=0 (that only disables its kill action, verified
            # on a restored container); DUMP_ON_TIMEOUT=0 stops the TCPStore
            # polling. World-size-1 inference only loses the flight recorder.
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
            "TORCH_NCCL_PROPAGATE_ERROR": "0",
            **VARIANT_ENV,
        }
    )
)
frontend_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "av==17.0.1",
        "bcrypt==4.3.0",
        "fastapi==0.136.3",
        "numpy==2.2.6",
        "uvicorn==0.52.1",
        "websockets==17.0.1",
    )
    .add_local_dir(
        repo_root / "examples" / "online_serving" / "duplexio" / "realtime_web",
        remote_path=str(FRONTEND_ROOT),
        copy=True,
    )
    .env({"PYTHONPATH": f"/app:{FRONTEND_ROOT}", **VARIANT_ENV})
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
        # Run the CLI as a module off PYTHONPATH, exactly as the cluster does:
        # vllm-omni is not pip-installed here, so there is no console script and
        # no second source of truth for which tree serves.
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        str(MODEL_PATH),
        "--omni",
        "--deploy-config",
        str(APP_ROOT / "vllm_omni" / "deploy" / DEPLOY_CONFIG_NAME),
        "--trust-remote-code",
        # A fresh image loads weights and compiles for longer than the engine's
        # 600 s default; the container's startup budget is the real limit.
        "--init-timeout",
        str(MODEL_STARTUP_TIMEOUT_SECONDS),
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
    environment["PYTHONPATH"] = ":".join(filter(None, (str(APP_ROOT), environment.get("PYTHONPATH"))))
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
            str(APP_ROOT / "examples" / "online_serving" / "duplexio" / "realtime_web" / "prewarm.py"),
            "--backend",
            LOCAL_BACKEND_WEBSOCKET_URL,
            "--model",
            str(MODEL_PATH),
            "--ref-audio",
            str(PREWARM_AUDIO_PATH),
            "--tools",
            str(APP_ROOT / "examples" / "online_serving" / "duplexio" / "realtime_web" / "tools.json"),
            "--timeout-seconds",
            "300",
        ],
        check=True,
        timeout=6 * 60,
    )


def build_model_app(process: subprocess.Popen[bytes], password_hash: str) -> object:
    """Expose engine health, and relay realtime sessions to the engine.

    Only these two routes are reachable: the engine's own server also serves
    sleep and wakeup controls. The page connects here directly, skipping a hop
    through the demo app, which serves only the page and its login. A session
    needs a ticket the demo app issues to signed-in pages.
    """
    import contextlib

    import websockets
    from realtime_relay import backend_query, relay, ticket_is_valid, ticket_key
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    def healthz(request) -> PlainTextResponse:
        healthy = backend_is_healthy(process)
        return PlainTextResponse("ok" if healthy else "unhealthy", status_code=200 if healthy else 503)

    key = ticket_key(password_hash)

    async def realtime(websocket) -> None:
        if not ticket_is_valid(key, websocket.query_params.get("ticket")):
            # Refuses the handshake (HTTP 403).
            await websocket.close(code=1008)
            return
        await websocket.accept()
        query, opus = backend_query(websocket.query_params.multi_items())
        url = f"{LOCAL_BACKEND_WEBSOCKET_URL}/v1/realtime?{query}"
        async with websockets.connect(url, max_size=64 * 1024 * 1024) as backend:
            await relay(websocket, backend, opus)
        # The page may have gone first.
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()

    return Starlette(routes=[Route("/healthz", healthz), WebSocketRoute("/v1/realtime", realtime)])


def check_model_present() -> None:
    if not MODEL_PATH.is_dir():
        raise RuntimeError(
            f"Model checkpoint not found at {MODEL_PATH}. Upload {MODEL_NAME} "
            f"to the {MODEL_VOLUME_NAME!r} Modal Volume."
        )
    if not PREWARM_AUDIO_PATH.is_file():
        raise RuntimeError(f"Upload a mono 24 kHz startup reference clip to {PREWARM_AUDIO_PATH}")


@app.cls(
    image=model_image,
    gpu="H100",
    # The default single core runs out at a few sessions: the engine core, API
    # server and relay each keep one busy, and every session's audio passes
    # through them.
    cpu=8,
    region=REGION,
    routing_region=ROUTING_REGION,
    volumes={"/models": model_volume.with_mount_options(read_only=True)},
    # The page's password secret signs the session tickets this relay checks.
    secrets=[modal.Secret.from_name(AUTH_SECRET_NAME)],
    # Sleep level 1 offloads weights (~13 GB) + the kv_cache pool
    # (~15 GB) into host RAM before the snapshot captures it.
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
    # exact CLI.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class SnapshotModelServer:
    @modal.enter(snap=True)
    def start_and_sleep(self) -> None:
        check_model_present()
        self.backend = start_backend(enable_sleep_mode=True)
        # Decode frames compile and initialize lazily on first use; run them now
        # so every restore starts from a warm snapshot.
        prewarm_backend()
        control_backend(
            "/v1/omni/sleep",
            {"stage_ids": SNAPSHOT_STAGE_IDS, "level": 1},
        )
        print("[snapshot] engine slept (weights offloaded); snapshotting")

    @modal.enter(snap=False)
    def wake(self) -> None:
        wake_started = time.monotonic()
        control_backend(
            "/v1/omni/wakeup",
            {"stage_ids": SNAPSHOT_STAGE_IDS},
        )
        wait_for_backend(self.backend)
        print(f"[snapshot] engine woke in {time.monotonic() - wake_started:.1f}s")

    @modal.exit()
    def stop(self) -> None:
        stop_backend(self.backend)

    # No proxy auth, which a browser cannot send: the relay checks the
    # demo app's session tickets instead.
    @modal.asgi_app(label=MODEL_WEB_LABEL, requires_proxy_auth=False)
    def web(self) -> object:
        return build_model_app(self.backend, os.environ[AUTH_PASSWORD_HASH_ENV])


@app.function(
    image=frontend_image,
    region=REGION,
    routing_region=ROUTING_REGION,
    volumes={"/models": model_volume.with_mount_options(read_only=True)},
    memory=1024,
    timeout=24 * 60 * 60,
    scaledown_window=60,
    max_containers=1,
    secrets=[modal.Secret.from_name(AUTH_SECRET_NAME)],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label=DEMO_WEB_LABEL)
def demo() -> object:
    """Serve the page and its login; the page streams to the GPU directly."""
    from realtime_web.server import (
        build_app,
        list_sample_clips,
        load_sampling_defaults,
    )

    tools_path = FRONTEND_ROOT / "tools.json"
    return build_app(
        ws_backend=BACKEND_WEBSOCKET_URL,
        model=str(MODEL_PATH),
        sample_clips=list_sample_clips(VOICE_DIR),
        sample_clip_dir=VOICE_DIR,
        default_voice=DEFAULT_VOICE,
        sampling=load_sampling_defaults(MODEL_PATH / "config.json"),
        tools=json.loads(tools_path.read_text(encoding="utf-8")),
        password_hash=os.environ[AUTH_PASSWORD_HASH_ENV],
        realtime_url=f"{BACKEND_WEBSOCKET_URL}/v1/realtime",
    )
