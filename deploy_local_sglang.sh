#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash deploy_local_sglang.sh start --model-path PATH --gpu ID --port PORT --run-dir DIR [OPTIONS]
  bash deploy_local_sglang.sh stop --run-dir DIR

This script only manages the process group recorded in DIR/service/runtime.json.
It never scans for or stops unrelated model/training processes.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
ACTION="$1"
shift

MODEL_PATH=""
GPU=""
PORT=""
RUN_DIR=""
PYTHON_BIN="${SGLANG_PYTHON:-$(command -v python)}"
TOOLCHAIN_BIN="${SGLANG_TOOLCHAIN_BIN:-$(dirname "$(command -v gcc)")}"
CONTEXT_LENGTH=131072
MEM_FRACTION=0.8
HOST=127.0.0.1
READY_TIMEOUT=900

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --mem-fraction-static) MEM_FRACTION="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --ready-timeout) READY_TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf '[ERROR] unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ -n "$RUN_DIR" ]] || { echo "[ERROR] --run-dir is required" >&2; exit 2; }
SERVICE_DIR="${RUN_DIR}/service"
RUNTIME_FILE="${SERVICE_DIR}/runtime.json"
LOG_FILE="${SERVICE_DIR}/sglang.log"

read_runtime() {
    python - "$RUNTIME_FILE" "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)[sys.argv[2]])
PY
}

validate_owned_process() {
    [[ -f "$RUNTIME_FILE" ]] || return 1
    local pid pgid starttime model_path port cmdline current_pgid current_start
    pid="$(read_runtime pid)"
    pgid="$(read_runtime pgid)"
    starttime="$(read_runtime starttime)"
    model_path="$(read_runtime model_path)"
    port="$(read_runtime port)"
    [[ -r "/proc/${pid}/stat" && -r "/proc/${pid}/cmdline" ]] || return 1
    current_pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    current_start="$(python - "$pid" <<'PY'
import sys
print(open(f"/proc/{sys.argv[1]}/stat", encoding="utf-8").read().split()[21])
PY
)"
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "$current_pgid" == "$pgid" && "$current_start" == "$starttime" ]] || return 1
    [[ "$cmdline" == *"sglang.launch_server"* && "$cmdline" == *"$model_path"* && "$cmdline" == *"--port $port"* ]] || return 1
    [[ "$pgid" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]] || return 1
}

if [[ "$ACTION" == "stop" ]]; then
    if ! validate_owned_process; then
        echo "[ERROR] runtime ownership validation failed; refusing to stop any process" >&2
        exit 1
    fi
    PID="$(read_runtime pid)"
    PGID="$(read_runtime pgid)"
    echo "[INFO] stopping owned SGLang process group pgid=${PGID}"
    kill -TERM -- "-${PGID}"
    for _ in $(seq 1 30); do
        [[ ! -r "/proc/${PID}/stat" ]] && { echo "[INFO] owned SGLang stopped"; exit 0; }
        sleep 1
    done
    if validate_owned_process; then
        echo "[WARN] owned process group did not exit after 30s; sending KILL"
        kill -KILL -- "-${PGID}"
    else
        echo "[ERROR] ownership changed during shutdown; refusing KILL" >&2
        exit 1
    fi
    exit 0
fi

[[ "$ACTION" == "start" ]] || { usage >&2; exit 2; }
[[ -n "$MODEL_PATH" && -n "$GPU" && -n "$PORT" ]] || {
    echo "[ERROR] start requires --model-path, --gpu and --port" >&2
    exit 2
}
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "[ERROR] invalid model path: $MODEL_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] Python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -x "${TOOLCHAIN_BIN}/gcc" && -x "${TOOLCHAIN_BIN}/g++" ]] || {
    echo "[ERROR] C++20 toolchain not found in $TOOLCHAIN_BIN" >&2
    exit 1
}
if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
    echo "[ERROR] port ${PORT} is already in use; refusing to stop its owner" >&2
    exit 1
fi
if [[ -f "$RUNTIME_FILE" ]] && validate_owned_process; then
    echo "[ERROR] an owned SGLang process is already active for this run directory" >&2
    exit 1
fi

mkdir -p "$SERVICE_DIR"
echo "[INFO] starting Qwen service on GPU ${GPU}, ${HOST}:${PORT}"
PATH="${TOOLCHAIN_BIN}:${PATH}" \
CC="${TOOLCHAIN_BIN}/gcc" \
CXX="${TOOLCHAIN_BIN}/g++" \
TVM_FFI_CACHE_DIR="${SERVICE_DIR}/cache/tvm-ffi" \
CUDA_VISIBLE_DEVICES="$GPU" setsid "$PYTHON_BIN" -u -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size 1 \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION" \
    --served-model-name default \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    >"$LOG_FILE" 2>&1 &
PID=$!
PGID="$(ps -o pgid= -p "$PID" | tr -d ' ')"
STARTTIME="$(python - "$PID" <<'PY'
import sys, time
path = f"/proc/{sys.argv[1]}/stat"
for _ in range(50):
    try:
        print(open(path, encoding="utf-8").read().split()[21])
        break
    except FileNotFoundError:
        time.sleep(0.1)
else:
    raise SystemExit("launcher exited before metadata capture")
PY
)"
python - "$RUNTIME_FILE" "$PID" "$PGID" "$STARTTIME" "$MODEL_PATH" "$PORT" "$GPU" "$HOST" "$CONTEXT_LENGTH" "$MEM_FRACTION" <<'PY'
import datetime, json, sys
keys = ["pid", "pgid", "starttime", "model_path", "port", "gpu", "host", "context_length", "mem_fraction_static"]
values = sys.argv[2:]
data = dict(zip(keys, values))
for key in ("pid", "pgid", "port", "gpu", "context_length"):
    data[key] = int(data[key])
data["mem_fraction_static"] = float(data["mem_fraction_static"])
data["started_at"] = datetime.datetime.now().astimezone().isoformat()
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

for ((elapsed=0; elapsed<READY_TIMEOUT; elapsed+=5)); do
    if ! validate_owned_process; then
        echo "[ERROR] owned SGLang process exited before readiness; see $LOG_FILE" >&2
        exit 1
    fi
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && \
       curl -fsS "http://${HOST}:${PORT}/v1/models" | grep -q 'default'; then
        echo "[INFO] SGLang ready: http://${HOST}:${PORT}/v1"
        exit 0
    fi
    sleep 5
done

echo "[ERROR] SGLang readiness timed out after ${READY_TIMEOUT}s; process remains owned and recorded" >&2
exit 1
