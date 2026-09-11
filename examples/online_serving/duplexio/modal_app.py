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

# A variant name deploys a parallel app under suffixed web labels
# (DUPLEXIO_MODAL_VARIANT=paged serves https://duplexio--demo-paged.modal.run).
# The live demo is unaffected unless the variable is unset at deploy time.
VARIANT = os.environ.get("DUPLEXIO_MODAL_VARIANT", "")
SUFFIX = f"-{VARIANT}" if VARIANT else ""
# Every image that imports this module has to agree with the deploying client on
# the variant: web labels and the frontend's backend URL are recomputed from it
# when a container imports the module.
VARIANT_ENV = {"DUPLEXIO_MODAL_VARIANT": VARIANT} if VARIANT else {}
APP_NAME = f"duplexio-vllm-omni{SUFFIX}"
MODEL_VOLUME_NAME = "duplexio-vllm-models"
MODEL_NAME = "duplexio-opd-warm23k-vllm-v5"
MODEL_PATH = Path("/models") / MODEL_NAME
# Default voice for prewarm + demo; must exist in the checkpoint voice pool
# (VoxCeleb id100xx set plus the two custom voices, boxlyx and maya).
VOICE = "69f67cf0ae15b9e491cd6b21"
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
MODEL_WEB_LABEL = f"model-snapshot{SUFFIX}"
DEMO_WEB_LABEL = f"demo{SUFFIX}"
BACKEND_WEBSOCKET_URL = f"wss://duplexio--{MODEL_WEB_LABEL}.modal.run"
BACKEND_HEALTH_URL = f"https://duplexio--{MODEL_WEB_LABEL}.modal.run/healthz"

repo_root = Path(__file__).resolve().parents[3] if modal.is_local() else APP_ROOT


def vllm_wheel() -> Path:
    """Locate the cluster-built vLLM wheel named by DUPLEXIO_VLLM_WHEEL."""
    setting = os.environ.get("DUPLEXIO_VLLM_WHEEL")
    if not setting:
        raise RuntimeError(
            "Set DUPLEXIO_VLLM_WHEEL to the vLLM wheel built against the "
            "training compiler stack (duplexio-modal-demo/wheel.sbatch)."
        )
    wheel = Path(setting)
    if not wheel.is_file():
        raise RuntimeError(f"DUPLEXIO_VLLM_WHEEL is not a file: {wheel}")
    return wheel


# Serving must run the compiler stack training parity was measured on: torch
# 2.13+cu130 (older torch has no flex-attention AuxRequest), vLLM built from
# upstream 568afb3a1, and the quack/FLA kernels the model calls directly. vLLM
# arrives as a wheel because compiling it on a Modal builder takes hours.
# uv reads the version from the wheel's filename, so the copy keeps its name.
# Only a local deploy builds the image, so the container-side value is unused.
VLLM_WHEEL_PATH = f"/wheels/{vllm_wheel().name}" if modal.is_local() else "/wheels"
model_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.13"
    )
    .apt_install("git", "ninja-build")
    .uv_pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "torchaudio==2.11.0",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_file(
        vllm_wheel() if modal.is_local() else VLLM_WHEEL_PATH,
        VLLM_WHEEL_PATH,
        copy=True,
    )
    .uv_pip_install(VLLM_WHEEL_PATH)
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
        "nvidia-cutlass-dsl==4.6.0.dev0",
        "quack-kernels==0.5.3",
        "flash-linear-attention==0.5.1",
        "fla-core==0.5.1",
        "transformers @ git+https://github.com/huggingface/transformers.git"
        "@b3d7e8c9d4e078a5e6c09a9d67e22dcadc2df4b8",
        pre=True,
    )
    .env(
        {
            "PYTHONPATH": str(APP_ROOT),
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
    .env({"PYTHONPATH": "/app", **VARIANT_ENV})
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
        print(
            f"[snapshot] engine woke in {time.monotonic() - wake_started:.1f}s"
        )

    @modal.exit()
    def stop(self) -> None:
        stop_backend(self.backend)

    # No proxy auth: the remote agent-speech gate drives this
    # backend directly.
    @modal.asgi_app(label=MODEL_WEB_LABEL, requires_proxy_auth=False)
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
