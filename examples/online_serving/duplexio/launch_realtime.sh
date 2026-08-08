#!/usr/bin/env bash
# Launch the native DuplexIO server and its local realtime web interface.
#
# Usage:
#   ./examples/online_serving/duplexio/launch_realtime.sh [checkpoint] [voice]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../.." && pwd)"

checkpoint="${1:-/home/anders/repos/duplexio/checkpoints/duplexio-475001-checkpoint-7-vllm}"
voice="${2:-69f67cf0ae15b9e491cd6b21}"
deploy_config="$repo_dir/vllm_omni/deploy/duplexio.yaml"
python="$repo_dir/.venv/bin/python"
vllm_omni="$repo_dir/.venv/bin/vllm-omni"
web_server="$script_dir/realtime_web/server.py"
prewarm="$script_dir/realtime_web/prewarm.py"
tools="$script_dir/realtime_web/tools.json"

asr_cudnn_lib="/home/anders/repos/duplexio/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib"
cuda_lib="/usr/local/cuda-13.0/lib64"
export LD_LIBRARY_PATH="$asr_cudnn_lib:$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

for port in 8099 7862; do
    if ss -H -ltn "sport = :$port" | rg -q .; then
        echo "Port $port is already in use; stop the existing realtime server first" >&2
        exit 1
    fi
done

backend_pid=""
cleanup() {
    if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
        kill "$backend_pid"
        wait "$backend_pid" || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting DuplexIO from $checkpoint"
echo "Using voice $voice and deploy config $deploy_config"
"$vllm_omni" serve "$checkpoint" \
    --omni \
    --deploy-config "$deploy_config" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 8099 &
backend_pid=$!

for _ in {1..180}; do
    if curl -fsS http://127.0.0.1:8099/health >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        wait "$backend_pid"
        exit 1
    fi
    sleep 1
done

if ! curl -fsS http://127.0.0.1:8099/health >/dev/null; then
    echo "DuplexIO did not become healthy within 180 seconds" >&2
    exit 1
fi

echo "Prewarming the DuplexIO realtime path"
"$python" "$prewarm" \
    --backend ws://127.0.0.1:8099 \
    --model "$checkpoint" \
    --voice "$voice" \
    --tools "$tools"

echo "DuplexIO is ready; opening the web interface on http://127.0.0.1:7862"
"$python" "$web_server" \
    --host 127.0.0.1 \
    --port 7862 \
    --ws-backend ws://127.0.0.1:8099 \
    --model "$checkpoint" \
    --voice "$voice" \
    --voice-manifest "$checkpoint/voices.json" \
    --tools "$tools"
