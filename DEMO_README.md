# End-to-End Demo

This guide shows the default input, local model service, three-stage pipeline, benchmark evaluation, and the record shape produced at every layer.

## 1. Repository layout

```text
repetition-code/
├── config/
│   ├── prompts/                  # committed prompt templates
│   └── runtime/                  # runtime configs and non-secret examples
├── data/
│   ├── README.md                 # source/provenance for the local subset
│   └── pipeline_default_5k.jsonl # ignored local 5K subset
├── pipeline_results/             # ignored Stage 1/2/3 output
└── eval_results/                 # ignored benchmark output
```

## 2. Raw input example

`data/pipeline_default_5k.jsonl` contains the first 5,000 valid records copied verbatim from the source listed in `data/README.md`. `data/example_input.jsonl` is a real one-row, three-field example extracted from that source and can be passed directly to `--dataset`.

A shortened full-source shape is:

```json
{
  "uuid": "example-uuid",
  "problem": "Solve the mathematical problem ...",
  "expected_answer": "42",
  "messages": [
    {"role": "user", "content": "source conversation"}
  ],
  "changed_answer_to_majority": false,
  "data_source": "source dataset name",
  "license": "source license",
  "tools": [],
  "tool_usage": "",
  "url": "source URL",
  "used_in": [],
  "user_name": "source author",
  "user_url": "source author URL"
}
```

The pipeline reads only `uuid`, `problem`, and `expected_answer`. It generates new `messages`; it does not reuse the source `messages` as its rollout.

## 3. Runtime configuration

Create local configs from examples:

```bash
cp config/runtime/pipeline.example.yaml config/runtime/pipeline.yaml
cp config/runtime/rollout_api.example.yaml config/runtime/rollout_api.yaml
cp config/runtime/judge_api.example.yaml config/runtime/judge_api.yaml
cp config/runtime/eval_benchmarks.example.yaml config/runtime/eval_benchmarks.yaml
cp config/runtime/eval_model_b.example.yaml config/runtime/eval_model_b.yaml
```

Responsibilities:

```text
config/prompts/   text templates only
config/runtime/pipeline.yaml      data fields, prompts, sampling, workers, retries
config/runtime/rollout_api.yaml   rollout endpoints only
config/runtime/judge_api.yaml     judge endpoints only
config/runtime/eval_model_b.yaml  model-B benchmark hyperparameters
```

Real endpoints and keys stay in ignored runtime files.

## 4. Quickstart: validate the pipeline

The shortest demo command is:

```bash
bash quickstart.sh
```

It validates the default 5K data and runtime configs in dry-run mode. To execute after configuring reachable endpoints:

```bash
bash quickstart.sh --run --limit 30 --output-dir pipeline_results/demo_30
```

`full_pipeline.sh` also uses `data/pipeline_default_5k.jsonl` when `--dataset` is omitted.

```bash
bash full_pipeline.sh \
  --config config/runtime/pipeline.yaml \
  --rollout-api-config config/runtime/rollout_api.yaml \
  --judge-api-config config/runtime/judge_api.yaml \
  --limit 30 \
  --output-dir pipeline_results/demo_30 \
  --dry-run
```

Remove `--dry-run` only after the endpoint configs are valid.

## 5. Quickstart: local Qwen checkpoint

Start one owned SGLang process on an explicitly available GPU:

```bash
bash deploy_local_sglang.sh start \
  --model-path /path/to/Qwen3.5-4B \
  --gpu 0 \
  --port 8000 \
  --run-dir pipeline_results/demo_30
```

Point `config/runtime/rollout_api.yaml` at `http://127.0.0.1:8000/v1`, then run the pipeline command above without `--dry-run`.

Stop only that recorded service:

```bash
bash deploy_local_sglang.sh stop \
  --run-dir pipeline_results/demo_30
```

The supervisor validates PID, PGID, process start time, model path and port before signaling anything.

## 6. Pipeline data flow

```text
input JSONL
  │
  ├─ Stage 1: rollout + repetition judge
  │    ├─ trajectory_raw.jsonl      all successful rollout attempts
  │    ├─ no_repetition.jsonl       positive Stage 1 candidates
  │    └─ has_repetition.jsonl      Stage 2 queue
  │
  ├─ Stage 2: rerollout + correctness judge
  │    ├─ repetition_correct.jsonl  positive Stage 2 candidates / Stage 3 queue
  │    └─ repetition_incorrect.jsonl
  │
  └─ Stage 3: reform + correctness judge
       ├─ turn3_reform_correct.jsonl positive Stage 3 candidates
       └─ turn3_reform_incorrect.jsonl
```

## 7. Output examples

### Stage 1 raw attempt

```json
{
  "uuid": "example-uuid",
  "problem": "...",
  "expected_answer": "42",
  "messages": [
    {"role": "user", "content": "configured rollout prompt"},
    {"role": "assistant", "reasoning_content": "...", "content": "... \\boxed{42}"}
  ],
  "attempt": 1,
  "rollout_count": 2
}
```

### Stage 1 classified record

```json
{
  "uuid": "example-uuid",
  "problem": "...",
  "expected_answer": "42",
  "messages": ["user and assistant messages"],
  "rollout_count": 2,
  "repeat_threshold": 1,
  "stage1_judge": {
    "is_repetition": false,
    "repeat_count": 0,
    "attempts": [],
    "analysis": "judge explanation",
    "predicted_answer": "42",
    "raw_content": "judge response",
    "raw_reasoning": "judge reasoning"
  }
}
```

### Stage 2 correct record

```json
{
  "uuid": "example-uuid",
  "stage1_judge": {"is_repetition": true},
  "stage2_judge": {
    "is_correct": true,
    "attempts": [
      {
        "attempt": 1,
        "predicted_answer": "42",
        "is_correct": true,
        "assistant_message": {
          "role": "assistant",
          "reasoning_content": "rerollout reasoning",
          "content": "rerollout answer"
        }
      }
    ]
  },
  "final_predicted_answer": "42",
  "final_messages": ["configured user prompt", "selected assistant message"]
}
```

### Stage 3 correct record

```json
{
  "uuid": "example-uuid",
  "turn2_messages": ["former Stage 2 final_messages"],
  "final_messages": [
    {"role": "user", "content": "original problem"},
    {"role": "assistant", "reasoning_content": "clean reform reasoning", "content": "clean final answer"}
  ],
  "stage3_judge": {
    "is_correct": true,
    "attempts": [
      {
        "attempt": 1,
        "reform_reasoning": "...",
        "reform_content": "...",
        "is_correct": true
      }
    ]
  }
}
```

For every audit field, see `README.md` and `PIPELINE_OUTPUT_SCHEMA.md`.

## 8. Compact training format

The reference `model_iterations_correct` format contains exactly three top-level fields:

```json
{
  "uuid": "example-uuid",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "reasoning_content": "...", "content": "..."}
  ],
  "source_stage": "stage1"
}
```

Mapping:

```text
Stage 1 no_repetition.messages              → messages, source_stage=stage1
Stage 2 repetition_correct.final_messages   → messages, source_stage=stage2
Stage 3 turn3_reform_correct.final_messages → messages, source_stage=stage3
```

Pipeline audit records are intentionally richer than this compact training format.

## 9. Benchmark evaluation

Model A accepts hyperparameters directly:

```bash
bash run_eval_model_a.sh http://model-a:8001/v1 default all \
  --comparison-id demo_compare \
  --benchmark-config config/runtime/eval_benchmarks.yaml \
  --judge-api-config config/runtime/judge_api.yaml \
  --judge-backend deepseek_397b
```

Model B reads separate defaults from `config/runtime/eval_model_b.yaml`:

```bash
bash run_eval_model_b.sh http://model-b:8001/v1 default all
```

CLI flags still override the model-B config. Results are written to:

```text
eval_results/<comparison_id>/model_a/...
eval_results/<comparison_id>/model_b/...
```
