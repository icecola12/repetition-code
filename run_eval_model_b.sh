#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DEFAULT_EVAL_CONFIG="${SCRIPT_DIR}/config/runtime/eval_model_b.yaml"

usage() {
    cat <<'EOF'
Usage: bash run_eval_model_b.sh API_URL SERVED_MODEL BENCHMARK_LIST [OPTIONS]

Runs the shared evaluator with model label "model_b" and defaults from
config/runtime/eval_model_b.yaml. CLI options override config values.
Example:
  bash run_eval_model_b.sh http://host:8001/v1 default all

Override the default config:
  bash run_eval_model_b.sh http://host:8001/v1 default all \
    --eval-config config/runtime/eval_model_b.example.yaml
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
[[ $# -ge 3 ]] || { usage >&2; exit 2; }

API_URL="$1"
SERVED_MODEL="$2"
BENCHMARK_LIST="$3"
shift 3

EVAL_CONFIG="$DEFAULT_EVAL_CONFIG"
if [[ " ${*} " == *" --eval-config "* ]]; then
    EVAL_CONFIG=""
fi

CONFIG_ARGS=()
[[ -n "$EVAL_CONFIG" ]] && CONFIG_ARGS=(--eval-config "$EVAL_CONFIG")

exec python "${SCRIPT_DIR}/run_eval.py" \
    --api-url "$API_URL" \
    --served-model "$SERVED_MODEL" \
    --benchmark-list "$BENCHMARK_LIST" \
    --model-label model_b \
    "${CONFIG_ARGS[@]}" \
    "$@"
