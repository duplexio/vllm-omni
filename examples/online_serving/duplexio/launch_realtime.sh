#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
serving_dir="$(cd "$script_dir/../../.." && pwd)"
checkpoint="${1:-$HOME/duplexio/run508946_step31000_v7}"
deploy_config="${DEPLOY_CONFIG:-$serving_dir/vllm_omni/deploy/duplexio-multistream.yaml}"
python="$serving_dir/.venv/bin/python"
vllm_omni="$serving_dir/.venv/bin/vllm-omni"
script_dir="$serving_dir/examples/online_serving/duplexio"
tools="$script_dir/realtime_web/tools.json"
prewarm="$script_dir/realtime_web/prewarm.py"
web_server="$script_dir/realtime_web/server.py"
engine_port="${ENGINE_PORT:-8099}"
web_port="${WEB_PORT:-7862}"
web_host="${WEB_HOST:-0.0.0.0}"

asr_cudnn_lib="$HOME/repos/duplexio/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib"
cuda_lib="/usr/local/cuda-13.0/lib64"
export LD_LIBRARY_PATH="$asr_cudnn_lib:$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$serving_dir${PYTHONPATH:+:$PYTHONPATH}"

for port in "$engine_port" "$web_port"; do
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

echo "Starting DuplexIO v7 from $checkpoint"
echo "Using deploy config $deploy_config"
"$vllm_omni" serve "$checkpoint" \
    --omni \
    --deploy-config "$deploy_config" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$engine_port" &
backend_pid=$!

for _ in {1..600}; do
    if curl -fsS "http://127.0.0.1:$engine_port/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        wait "$backend_pid"
        exit 1
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:$engine_port/health" >/dev/null || {
    echo "DuplexIO did not become healthy in time" >&2
    exit 1
}

echo "Prewarming prefix and live audio for two sessions"
"$python" "$prewarm" \
    --backend "ws://127.0.0.1:$engine_port" \
    --model "$checkpoint" \
    --ref-audio "$checkpoint/prewarm.wav" \
    --tools "$tools" \
    --timeout-seconds 300 \
    --sessions 2

echo "Web interface on http://$web_host:$web_port"
"$python" "$web_server" \
    --host "$web_host" \
    --port "$web_port" \
    --ws-backend "ws://127.0.0.1:$engine_port" \
    --model "$checkpoint" \
    --sample-clips "$checkpoint" \
    --tools "$tools"
