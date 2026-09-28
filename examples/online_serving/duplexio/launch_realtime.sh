#!/usr/bin/env bash
# Launch the native DuplexIO server and its realtime web interface.
#
# Usage:
#   ./examples/online_serving/duplexio/launch_realtime.sh CHECKPOINT [VOICE]
#
# CHECKPOINT is a native export. Its prewarm.wav (mono 24 kHz) warms the first
# session, and its audio clips are offered as voices in the page; VOICE names the
# clip selected on load. DEPLOY_CONFIG,
# WEB_HOST (default 127.0.0.1) and WEB_PORT (default 7862) override the defaults.
# Ctrl-C stops the web interface and the server.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../.." && pwd)"

checkpoint="${1:?usage: $0 CHECKPOINT [VOICE]}"
voice="${2:-}"
deploy_config="${DEPLOY_CONFIG:-$repo_dir/vllm_omni/deploy/duplexio-realtime.yaml}"
web_host="${WEB_HOST:-127.0.0.1}"
web_port="${WEB_PORT:-7862}"
python="$repo_dir/.venv/bin/python"
vllm_omni="$repo_dir/.venv/bin/vllm-omni"
web_server="$script_dir/realtime_web/server.py"
prewarm="$script_dir/realtime_web/prewarm.py"
tools="$script_dir/realtime_web/tools.json"

asr_cudnn_lib="$HOME/repos/duplexio/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib"
cuda_lib="/usr/local/cuda-13.0/lib64"
export LD_LIBRARY_PATH="$asr_cudnn_lib:$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

for port in 8099 "$web_port"; do
    if ss -H -ltn "sport = :$port" | grep -q .; then
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
echo "Using deploy config $deploy_config"
"$vllm_omni" serve "$checkpoint" \
    --omni \
    --deploy-config "$deploy_config" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 8099 &
backend_pid=$!

# Every serving variant is compiled and captured before the server reports healthy.
for _ in {1..600}; do
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
    echo "DuplexIO did not become healthy within 600 seconds" >&2
    exit 1
fi

echo "Prewarming the DuplexIO realtime path"
"$python" "$prewarm" \
    --backend ws://127.0.0.1:8099 \
    --model "$checkpoint" \
    --ref-audio "$checkpoint/prewarm.wav" \
    --tools "$tools" \
    --timeout-seconds 300

echo "DuplexIO is ready; web interface on http://$web_host:$web_port"
"$python" "$web_server" \
    --host "$web_host" \
    --port "$web_port" \
    --ws-backend ws://127.0.0.1:8099 \
    --model "$checkpoint" \
    --sample-clips "$checkpoint" \
    ${voice:+--default-voice "$voice"} \
    --tools "$tools"
