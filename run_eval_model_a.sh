#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DEFAULT_JUDGE_CONFIG="${SCRIPT_DIR}/config/runtime/judge_api.yaml"
DEFAULT_JUDGE_BACKEND="deepseek_397b"

usage() {
    cat <<'EOF'
Usage: bash run_eval_model_a.sh API_URL SERVED_MODEL BENCHMARK_LIST [OPTIONS]

Runs the shared evaluator with model label "model_a".
By default it uses config/runtime/judge_api.yaml and backend deepseek_397b.
Explicit --judge-api-config / --judge-backend options override these defaults.
Example:
  bash run_eval_model_a.sh http://host:8001/v1 default all \
    --comparison-id experiment_001 \
    --benchmark-config config/runtime/eval_benchmarks.yaml
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
[[ $# -ge 3 ]] || { usage >&2; exit 2; }

API_URL="$1"
SERVED_MODEL="$2"
BENCHMARK_LIST="$3"
shift 3

DEFAULT_ARGS=(
    --judge-api-config "$DEFAULT_JUDGE_CONFIG"
    --judge-backend "$DEFAULT_JUDGE_BACKEND"
)
for arg in "$@"; do
    case "$arg" in
        --judge-api-config) DEFAULT_ARGS=(--judge-backend "$DEFAULT_JUDGE_BACKEND") ;;
        --judge-backend)
            if [[ " ${DEFAULT_ARGS[*]} " == *" --judge-api-config "* ]]; then
                DEFAULT_ARGS=(--judge-api-config "$DEFAULT_JUDGE_CONFIG")
            else
                DEFAULT_ARGS=()
            fi
            ;;
    esac
done

exec python "${SCRIPT_DIR}/run_eval.py" \
    --api-url "$API_URL" \
    --served-model "$SERVED_MODEL" \
    --benchmark-list "$BENCHMARK_LIST" \
    --model-label model_a \
    "${DEFAULT_ARGS[@]}" \
    "$@"
