# Pipeline Output Schema

This document separates the pipeline's resumable audit records from the compact training records derived from successful stages. Paths are relative to one run directory, for example `pipeline_results/qwen35_4b_smoke_30/`.

## Output tree and stage flow

```text
input JSONL
  │
  ├─ Stage 1: rollout + repetition judge
  │    ├─ stage1/trajectory_raw.jsonl
  │    ├─ stage1/no_repetition.jsonl        → Stage 1 training candidates
  │    └─ stage1/has_repetition.jsonl       → Stage 2 input
  │
  ├─ Stage 2: rerollout + correctness judge
  │    ├─ stage2/repetition_correct.jsonl   → Stage 3 input / Stage 2 candidates
  │    └─ stage2/repetition_incorrect.jsonl → terminal audit records
  │
  └─ Stage 3: reform + correctness judge
       ├─ stage3/turn3_reform_correct.jsonl → Stage 3 training candidates
       └─ stage3/turn3_reform_incorrect.jsonl
```

All JSONL checkpoints are append-only during a run. Resume identity is `uuid` for classified records. `trajectory_raw.jsonl` can contain multiple attempts for one UUID.

## Pipeline audit records

### Stage 1 raw trajectory

`stage1/trajectory_raw.jsonl` stores one line per successful rollout attempt:

```text
uuid: string
problem: string
expected_answer: any
messages: ChatMessage[]
attempt: integer, 1-based
rollout_count: integer
```

`messages` contains the configured rollout prompt and the assistant message:

```text
messages[0]: {role: "user", content: string}
messages[1]: {role: "assistant", reasoning_content: string, content: string}
```

This file does not contain the repetition verdict; it is an audit trail for generation.

### Stage 1 classified records

Both `stage1/no_repetition.jsonl` and `stage1/has_repetition.jsonl` use:

```text
uuid
problem
expected_answer
messages
rollout_count
repeat_threshold
stage1_judge:
  is_repetition: boolean
  repeat_count: integer
  attempts: Stage1Attempt[]
  analysis: string | null
  predicted_answer: string
  raw_content: string
  raw_reasoning: string
```

Each `Stage1Attempt` records the attempt number, parsed verdict/analysis, predicted answer and raw judge response. It omits the duplicate `messages` value stored at record level.

Semantics:

- `no_repetition.jsonl`: the aggregated repetition count is below the threshold. Every successful rollout for the UUID can become a separate training candidate, so UUID is not necessarily unique inside this file.
- `has_repetition.jsonl`: the threshold is met. One representative repeated trajectory is retained and submitted to Stage 2.

Judge output can be empty or unparseable after failures; audit fields therefore should not be treated as non-empty schema constraints.

### Stage 2 records

Both Stage 2 files retain the complete Stage 1 record and add:

```text
stage2_judge:
  is_correct: boolean
  attempts: Stage2Attempt[]
```

Each Stage 2 attempt may contain:

```text
attempt: integer
predicted_answer: string
is_correct: boolean | null
assistant_message:
  role: "assistant"
  reasoning_content: string
  content: string
error: string  # only on a failed request/attempt
```

Successful records in `stage2/repetition_correct.jsonl` additionally contain:

```text
final_predicted_answer: string
final_messages:
  - {role: "user", content: configured rollout prompt}
  - {role: "assistant", reasoning_content: string, content: string}
```

`repetition_incorrect.jsonl` is terminal for Stage 2. `repetition_correct.jsonl` is consumed by Stage 3.

### Stage 3 records

Both Stage 3 files retain the Stage 2 record and add/replace:

```text
turn2_messages: ChatMessage[]  # the former Stage 2 final_messages
final_messages: ChatMessage[] # the reform output
stage3_judge:
  is_correct: boolean
  attempts: Stage3Attempt[]
```

`Stage3Attempt` contains attempt number, reform reasoning/content and correctness verdict, or an `error` value such as `parse_failed` / `judge_failed`.

The reform output has exactly one user and one assistant message. Correct records are training candidates; incorrect records remain audit output.

## Compact training record

The reference dataset under `00_research/raw_data/output/model_iterations_correct` projects successful pipeline records into one compact schema:

```text
uuid: string
messages:
  - role: "user"
    content: string
  - role: "assistant"
    reasoning_content: string
    content: string
source_stage: "stage1" | "stage2" | "stage3"
```

Mapping:


| Training stage | Source file                         | Source messages  |
| ---------------- | ------------------------------------- | ------------------ |
| `stage1`       | `stage1/no_repetition.jsonl`        | `messages`       |
| `stage2`       | `stage2/repetition_correct.jsonl`   | `final_messages` |
| `stage3`       | `stage3/turn3_reform_correct.jsonl` | `final_messages` |

Only positive outcomes enter training data. Judge evidence, expected answers, retry attempts, `turn2_messages` and other audit fields are intentionally excluded.

Important constraints inherited from the reference data:

- Stage 1 can contain multiple different trajectories for one UUID.
- Stage 2 and Stage 3 are normally deduplicated by UUID when building training files.
- Concatenated training files are ordered Stage 1 → Stage 2 → Stage 3; they are not guaranteed to be shuffled or globally unique by UUID.
- `reasoning_content` can be tens of thousands of characters.
- Historical data can contain an empty assistant `content`; consumers that require non-empty final answers need an explicit quality filter.

## Smoke-run measurements

Two 30-row runs were used to verify the implementation:

### Natural routing: `pipeline_results/qwen35_4b_smoke_30/`

| File | Rows | Unique UUIDs | JSON errors |
|---|---:|---:|---:|
| `stage1/trajectory_raw.jsonl` | 60 | 30 | 0 |
| `stage1/no_repetition.jsonl` | 60 | 30 | 0 |
| `stage1/has_repetition.jsonl` | 0 | 0 | 0 |
| Stage 2 outputs | 0 | 0 | 0 |
| Stage 3 outputs | 0 | 0 | 0 |

All 30 UUIDs were naturally classified as non-repetitive, so Stage 2 and Stage 3 correctly had no input.

### Forced end-to-end routing: `pipeline_results/qwen35_4b_forced_all_stages_30/`

| File | Rows | Unique UUIDs | JSON errors |
|---|---:|---:|---:|
| `stage1/has_repetition.jsonl` | 30 | 30 | 0 |
| `stage2/repetition_correct.jsonl` | 30 | 30 | 0 |
| `stage3/turn3_reform_correct.jsonl` | 30 | 30 | 0 |

For this isolated validation run, the user explicitly authorized forced routing. Stage 1 records state that repetition was forced rather than judged. The configured judge endpoint returned HTTP 402 (quota exhausted), so Stage 2 and Stage 3 attempts contain `forced_correct: true` and an explicit `force_reason`. Rollout/rerollout/reform text was still generated by the real Qwen3.5-4B service. These records validate data flow and schema; they must not be presented as genuine correctness-judge labels.
