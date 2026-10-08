#!/usr/bin/env bash
# Start DuplexIO, ask ten spoken questions, and save inputs and answers.
# Usage: ./test_inference.sh [output-directory] [questions.json] [run_questions.py options...]
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:-outputs/duplexio-inference/$(date +%Y%m%d-%H%M%S)}
questions=${2:-$script_dir/questions.json}
model=${DUPLEXIO_MODEL:-duplexio/duo-4b}
revision=${DUPLEXIO_REVISION:-80de418a31334c737100afe767a08b1a27f6ee1a}
port=${DUPLEXIO_PORT:-8099}
healthy() {
    python - "$port" <<'PYTHON'
import sys
from urllib.request import urlopen
try:
    with urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except (OSError, TimeoutError):
    sys.exit(1)
PYTHON
}
if [[ -e "$out" ]]; then
    echo "Output directory already exists: $out" >&2
    exit 1
fi
if healthy; then
    echo "A server already uses port $port. Set DUPLEXIO_PORT to a free port." >&2
    exit 1
fi
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
server_args=(serve "$model" --omni --revision "$revision" --host 127.0.0.1 --port "$port"
             --init-timeout 900 --stage-init-timeout 900)
printf 'vllm' > "$out/server-command.txt"
printf ' %q' "${server_args[@]}" >> "$out/server-command.txt"
printf '\n' >> "$out/server-command.txt"
echo "Starting server; log: $out/server.log"
setsid vllm "${server_args[@]}" > "$out/server.log" 2>&1 &
server=$!
cleanup() {
    kill -- "-$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=false
for ((attempt=0; attempt<450; attempt++)); do
    if ! kill -0 "$server" 2>/dev/null; then
        tail -40 "$out/server.log" >&2
        exit 1
    fi
    if healthy; then
        ready=true
        break
    fi
    sleep 2
done
if [[ "$ready" != true ]]; then
    echo "Server startup timed out; inspect $out/server.log" >&2
    exit 1
fi
echo "Server healthy. Running spoken questions."
python "$script_dir/run_questions.py" --model "$model" --url "ws://127.0.0.1:$port" \
    --questions "$questions" --output-dir "$out" "${@:3}"
