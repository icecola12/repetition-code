# Math Reasoning Data Pipeline

A compact research pipeline for building supervised fine-tuning data from mathematical reasoning rollouts.

## Pipeline

1. Generate multiple independent reasoning rollouts for each problem.
2. Judge each rollout for repetition and aggregate the verdicts.
3. Reroll repetitive samples and retain corrected trajectories.
4. Reform successful trajectories into final SFT examples.
5. Aggregate completed runs into categorized JSONL datasets.

The project uses OpenAI-compatible inference endpoints. Endpoint URLs, credentials, model names, worker counts, and local data paths are intentionally not included in this repository; provide them in a local YAML configuration.

## Core files

- `streaming_pipeline.py` — concurrent rollout, repetition judging, and rerollout pipeline
- `share_utils.py` — shared clients, parsing, prompts, and trajectory helpers
- `turn3_reform.py` — final trajectory reform stage
- `aggregate_to_data_all.py` — aggregate completed runs by outcome
- `rollout_only.py` — standalone rollout utility
- `judge_only.py` — standalone judge utility
- `rerollout.py` — standalone rerollout utility
- `config/` — prompt templates only; runtime endpoint configurations are excluded

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

Create a local inference configuration compatible with the fields consumed by the scripts, then run:

```bash
python streaming_pipeline.py \
  --config path/to/inference_config.yaml \
  --dataset path/to/input.jsonl \
  --output-dir path/to/output
```

Run `python streaming_pipeline.py --help` for all options. Generated datasets and logs are ignored by Git.

## Data schema

Input rows should provide a unique `uuid`, a math `problem`, and an `expected_answer`. Output trajectories use OpenAI-style `messages`, with assistant reasoning in `reasoning_content` and the final response in `content`.
