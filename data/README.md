# Default Pipeline Data

`pipeline_default_5k.jsonl` is the local default input used by `full_pipeline.sh` when `--dataset` is not provided.

`example_input.jsonl` is a tracked one-record example derived from the first source row. It keeps only the three fields consumed by the pipeline (`uuid`, `problem`, `expected_answer`) so documentation and dry-run examples do not duplicate unrelated source metadata.

## Source

- Original file: a local Nemotron-SFT-Math-v3 reformed no-tools JSONL supplied by the dataset owner (the absolute source path is intentionally not committed)
- Source dataset family: Nemotron-SFT-Math-v3, reformed no-tools training data
- Extraction rule: preserve source order and copy the first 5,000 non-empty, valid JSONL records
- Source physical lines scanned: 5,000
- Output records: 5,000
- Output size at extraction time: 217,714,065 bytes
- SHA-256: `ca6e9182bfb65366d7d35fb32deb1224191cd203272c6141facbc665de2433f3`

No field was renamed, removed, or generated during extraction. Each output line is copied verbatim from the corresponding source line.

## Fields present in the source records

| Field | Type | Pipeline use |
|---|---|---|
| `uuid` | string | Stable resume and classification key. |
| `problem` | string | Mathematical problem passed into the configured rollout prompt. |
| `expected_answer` | string | Reference answer used by Stage 2 and Stage 3 correctness judging. |
| `messages` | array | Existing source conversation; retained in the source subset but not used as the pipeline's generated trajectory. |
| `changed_answer_to_majority` | boolean | Source metadata; not consumed by the current pipeline. |
| `data_source` | string | Source metadata; not consumed by the current pipeline. |
| `license` | string | Source license metadata. |
| `tools` | array | Source tool schema metadata; not consumed by the no-tools pipeline. |
| `tool_usage` | string | Source tool-use metadata; not consumed by the current pipeline. |
| `url` | string | Source provenance metadata. |
| `used_in` | array | Source usage metadata. |
| `user_name` | string | Source attribution metadata. |
| `user_url` | string | Source attribution metadata. |

The configured pipeline reads only `uuid`, `problem`, and `expected_answer`. See [`../README.md`](../README.md) and [`../PIPELINE_OUTPUT_SCHEMA.md`](../PIPELINE_OUTPUT_SCHEMA.md) for generated Stage 1/2/3 fields.

## Repository policy

The JSONL file is intentionally ignored by Git because it is a generated local data subset and is about 218 MB. This README is tracked so the subset can be reproduced from the source path and extraction rule.

Override the default input at runtime with:

```bash
bash full_pipeline.sh --dataset /path/to/another.jsonl [other options]
```
