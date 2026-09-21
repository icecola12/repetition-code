#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN=false
LIMIT=30
OUTPUT_DIR="pipeline_results/quickstart_$(date +%Y%m%d_%H%M)"

usage() {
    cat <<'EOF'
Usage: bash quickstart.sh [OPTIONS]

Validates the default 5K dataset and runtime configuration without network calls.
Use --run only after rollout and judge endpoints are configured and reachable.

Options:
  --run                 execute the pipeline (default is dry-run)
  --limit N             number of input rows (default: 30)
  --output-dir PATH     pipeline output directory
  -h, --help            show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ARGS=(
    --config config/runtime/pipeline.yaml
    --rollout-api-config config/runtime/rollout_api.yaml
    --judge-api-config config/runtime/judge_api.yaml
    --dataset data/pipeline_default_5k.jsonl
    --limit "$LIMIT"
    --output-dir "$OUTPUT_DIR"
    --run-id "$(basename "$OUTPUT_DIR")"
)

[[ "$RUN" == true ]] || ARGS+=(--dry-run)

exec bash "$SCRIPT_DIR/full_pipeline.sh" "${ARGS[@]}"
