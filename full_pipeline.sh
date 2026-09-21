#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"
CONFIG="${WORK_DIR}/config/runtime/pipeline.yaml"
ROLLOUT_API_CONFIG="${WORK_DIR}/config/runtime/rollout_api.yaml"
JUDGE_API_CONFIG="${WORK_DIR}/config/runtime/judge_api.yaml"
DATASET="${WORK_DIR}/data/pipeline_default_5k.jsonl"
OUTPUT_DIR=""
RUN_ID="run"
LIMIT=""
ROLLOUT_WORKERS=""
JUDGE_WORKERS=""
REROLLOUT_WORKERS=""
TURN3_REFORM_WORKERS=""
STATS_INTERVAL=60
RETRY_TURN3_INCORRECT=false
AGGREGATE=false
DRY_RUN=false

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: bash full_pipeline.sh --dataset PATH [OPTIONS]

Required API configuration:
  --rollout-api-config PATH    rollout-only API YAML
  --judge-api-config PATH      judge-only API YAML

Options:
  --config PATH                pipeline YAML (default: config/runtime/pipeline.yaml)
  --dataset PATH               input JSONL (default: data/pipeline_default_5k.jsonl)
  --output-dir PATH            output directory (default: pipeline_results/YYYYmmdd_HHMM)
  --run-id ID                  run id (default: run)
  --limit N                    process at most N input rows
  --rollout-workers N          override config.rollout.max_workers
  --judge-workers N            override config.judge.max_workers
  --rerollout-workers N        override config.rerollout.max_workers
  --turn3-reform-workers N     override config.turn3_reform.max_workers
  --stats-interval N           progress interval in seconds (default: 60)
  --retry-turn3-incorrect      retry existing Stage 3 incorrect records
  --aggregate                  aggregate runs after Stage 3
  --dry-run                    validate and print commands without writing or calling APIs
  -h, --help                   show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --rollout-api-config) ROLLOUT_API_CONFIG="$2"; shift 2 ;;
        --judge-api-config) JUDGE_API_CONFIG="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --rollout-workers) ROLLOUT_WORKERS="$2"; shift 2 ;;
        --judge-workers) JUDGE_WORKERS="$2"; shift 2 ;;
        --rerollout-workers) REROLLOUT_WORKERS="$2"; shift 2 ;;
        --turn3-reform-workers) TURN3_REFORM_WORKERS="$2"; shift 2 ;;
        --stats-interval) STATS_INTERVAL="$2"; shift 2 ;;
        --retry-turn3-incorrect) RETRY_TURN3_INCORRECT=true; shift ;;
        --aggregate) AGGREGATE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$DATASET" ]] || die "dataset path cannot be empty"
[[ -f "$CONFIG" ]] || die "pipeline config not found: $CONFIG"
[[ -f "$ROLLOUT_API_CONFIG" ]] || die "rollout API config not found: $ROLLOUT_API_CONFIG"
[[ -f "$JUDGE_API_CONFIG" ]] || die "judge API config not found: $JUDGE_API_CONFIG"
[[ -f "$DATASET" ]] || die "dataset not found: $DATASET"
[[ -f "${WORK_DIR}/streaming_pipeline.py" ]] || die "streaming_pipeline.py not found"
[[ -f "${WORK_DIR}/turn3_reform.py" ]] || die "turn3_reform.py not found"
[[ -f "${WORK_DIR}/aggregate_to_data_all.py" ]] || die "aggregate_to_data_all.py not found"

python - "$CONFIG" "$ROLLOUT_API_CONFIG" "$JUDGE_API_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

paths = [Path(value).resolve() for value in sys.argv[1:]]
configs = []
for path in paths:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"configuration must be a YAML mapping: {path}")
    configs.append(data)

pipeline, rollout, judge = configs
if "qwen_apex" not in rollout:
    raise SystemExit("rollout API config missing qwen_apex")
if not any(judge.get(name) for name in ("kimi", "glm", "deepseek_397b")):
    raise SystemExit("judge API config has no supported backend")

config_dir = paths[0].parent
project_root = config_dir.parent.parent if config_dir.name == "runtime" else config_dir.parent
def resolve_prompt(value):
    path_value = Path(value).expanduser()
    candidates = [path_value] if path_value.is_absolute() else [
        project_root / path_value,
        config_dir / path_value,
        config_dir / path_value.name,
        Path.cwd() / path_value,
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise SystemExit(f"prompt template not found: {value}")
    return path

def validate_prompt(value, field):
    path = resolve_prompt(value)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or field not in data:
        raise SystemExit(f"prompt file missing {field}: {path}")

prompts = pipeline.get("prompts") or {}
for key, field in (("rollout", "rollout"), ("repetition", "judge"), ("correctness", "judge")):
    value = prompts.get(key)
    if not value:
        raise SystemExit(f"pipeline config missing prompts.{key}")
    validate_prompt(value, field)

rerollout = pipeline.get("rerollout") or {}
validate_prompt(rerollout.get("prompt", "config/prompts/turn2_rerollout_v4.yaml"), "rerollout")
turn3 = pipeline.get("turn3_reform") or {}
validate_prompt(turn3.get("prompt", "config/prompts/turn3_reform_v3.yaml"), "reform")
PY

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${WORK_DIR}/pipeline_results/$(date +%Y%m%d_%H%M)"
fi

PIPELINE_ARGS=(
    --config "$CONFIG"
    --rollout-api-config "$ROLLOUT_API_CONFIG"
    --judge-api-config "$JUDGE_API_CONFIG"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --run-id "$RUN_ID"
    --stats-interval "$STATS_INTERVAL"
)
[[ -n "$LIMIT" ]] && PIPELINE_ARGS+=(--limit "$LIMIT")
[[ -n "$ROLLOUT_WORKERS" ]] && PIPELINE_ARGS+=(--rollout-workers "$ROLLOUT_WORKERS")
[[ -n "$JUDGE_WORKERS" ]] && PIPELINE_ARGS+=(--judge-workers "$JUDGE_WORKERS")
[[ -n "$REROLLOUT_WORKERS" ]] && PIPELINE_ARGS+=(--rerollout-workers "$REROLLOUT_WORKERS")

STAGE2_CORRECT="${OUTPUT_DIR}/stage2/repetition_correct.jsonl"
STAGE3_DIR="${OUTPUT_DIR}/stage3"
TURN3_ARGS=(
    --config "$CONFIG"
    --rollout-api-config "$ROLLOUT_API_CONFIG"
    --judge-api-config "$JUDGE_API_CONFIG"
    --input-file "$STAGE2_CORRECT"
    --output-dir "$STAGE3_DIR"
    --skip-existing
)
[[ -n "$TURN3_REFORM_WORKERS" ]] && TURN3_ARGS+=(--max-workers "$TURN3_REFORM_WORKERS")
[[ "$RETRY_TURN3_INCORRECT" == true ]] && TURN3_ARGS+=(--retry-incorrect)

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY-RUN] Configuration validated. No files will be created and no API calls will run."
    echo "[DRY-RUN] Stage 1+2:"
    print_command python "${WORK_DIR}/streaming_pipeline.py" "${PIPELINE_ARGS[@]}"
    echo "[DRY-RUN] Stage 3 (when Stage 2 has records):"
    print_command python "${WORK_DIR}/turn3_reform.py" "${TURN3_ARGS[@]}"
    if [[ "$AGGREGATE" == true ]]; then
        echo "[DRY-RUN] Aggregate:"
        print_command python "${WORK_DIR}/aggregate_to_data_all.py" \
            --output-base-dir "$(dirname -- "$OUTPUT_DIR")" \
            --data-all-dir "$(dirname -- "$OUTPUT_DIR")/data_all"
    fi
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
PIPELINE_LOG="${OUTPUT_DIR}/full_pipeline.log"
exec > >(while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$PIPELINE_LOG") 2>&1

LOCK_FILE="${OUTPUT_DIR}/.pipeline.lock"
exec {LOCK_FD}>"$LOCK_FILE"
flock -n "$LOCK_FD" || die "another pipeline is using output directory: $OUTPUT_DIR"

echo "[INFO] Stage 1+2 starting"
print_command python "${WORK_DIR}/streaming_pipeline.py" "${PIPELINE_ARGS[@]}"
python "${WORK_DIR}/streaming_pipeline.py" "${PIPELINE_ARGS[@]}"

echo "[INFO] Stage 1+2 completed"
if [[ -s "$STAGE2_CORRECT" ]]; then
    mkdir -p "$STAGE3_DIR"
    echo "[INFO] Stage 3 starting"
    print_command python "${WORK_DIR}/turn3_reform.py" "${TURN3_ARGS[@]}"
    python "${WORK_DIR}/turn3_reform.py" "${TURN3_ARGS[@]}"
    echo "[INFO] Stage 3 completed"
else
    echo "[INFO] Stage 3 skipped: no records in $STAGE2_CORRECT"
fi

if [[ "$AGGREGATE" == true ]]; then
    OUTPUT_BASE_DIR="$(dirname -- "$OUTPUT_DIR")"
    DATA_ALL_DIR="${OUTPUT_BASE_DIR}/data_all"
    echo "[INFO] Aggregation starting"
    python "${WORK_DIR}/aggregate_to_data_all.py" \
        --output-base-dir "$OUTPUT_BASE_DIR" \
        --data-all-dir "$DATA_ALL_DIR"
    echo "[INFO] Aggregation completed: $DATA_ALL_DIR"
fi

echo "[INFO] Full pipeline completed: $OUTPUT_DIR"
