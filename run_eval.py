#!/usr/bin/env python3
"""Evaluate one OpenAI-compatible model across configured math benchmarks."""

import argparse
import datetime as dt
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

from share_utils import (
    ChatClient,
    extract_boxed,
    load_yaml_field,
    parse_correctness_judgement,
    render_template,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_TRAJ_FIELDS = {"example_id", "rollout_id", "score", "answer", "prompt", "response"}
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def load_yaml_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (config_path.parent / path).resolve()
    return candidate if candidate.exists() else path.resolve()


def load_benchmark(path: Path, spec: dict) -> list:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark not found: {path}")
    if path.suffix.lower() == ".jsonl" or spec.get("format") == "jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    else:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"benchmark must be a JSON array or JSONL file: {path}")

    problem_field = spec.get("problem_field", "problem")
    answer_field = spec.get("answer_field", "answer")
    id_field = spec.get("id_field")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"benchmark row {index} is not an object: {path}")
        missing = [field for field in (problem_field, answer_field) if field not in row]
        if missing:
            raise ValueError(f"benchmark row {index} missing {missing}: {path}")
        normalized.append({
            "example_id": row.get(id_field, index) if id_field else index,
            "problem": str(row[problem_field]),
            "answer": str(row[answer_field]),
        })
    return normalized


def select_benchmarks(config: dict, requested: str) -> list:
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("benchmark config missing non-empty 'benchmarks' mapping")
    names = list(benchmarks) if requested == "all" else [name.strip() for name in requested.split(",")]
    if not names or any(not name for name in names):
        raise ValueError("benchmark list cannot be empty")
    unknown = [name for name in names if name not in benchmarks]
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")
    return names


def validate_name(value: str, label: str) -> str:
    if not SAFE_NAME_RE.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' or '-'")
    return value


def trajectory_path(model_dir: Path, benchmark: str, run_index: int, item_index: int) -> Path:
    return model_dir / benchmark / f"run_{run_index}" / f"traj_{item_index}.json"


def brief_path(model_dir: Path, benchmark: str, run_index: int) -> Path:
    return model_dir / benchmark / f"run_{run_index}" / "brief.json"


def is_valid_trajectory(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or not REQUIRED_TRAJ_FIELDS.issubset(data):
        return False
    response = data.get("response")
    return (
        isinstance(response, list)
        and bool(response)
        and isinstance(response[0], dict)
        and "content" in response[0]
        and "reasoning_content" in response[0]
    )


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def safe_api_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))


def load_judge_config(path: Path, backend: str) -> dict:
    data = load_yaml_mapping(path)
    config = data.get(backend)
    if not isinstance(config, dict):
        raise ValueError(f"judge backend '{backend}' not found in {path}")
    urls = config.get("base_urls") or ([config["base_url"]] if config.get("base_url") else [])
    if not urls or not config.get("model"):
        raise ValueError(f"judge backend '{backend}' requires base_url(s) and model")
    result = dict(config)
    result["base_urls"] = urls
    return result


def write_brief(model_dir: Path, benchmark: str, run_index: int, total: int,
                started_at: float) -> bool:
    run_dir = model_dir / benchmark / f"run_{run_index}"
    paths = [run_dir / f"traj_{index}.json" for index in range(total)]
    if not all(is_valid_trajectory(path) for path in paths):
        return False
    resolved = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            resolved += float(json.load(handle).get("score", 0.0)) >= 0.5
    write_json(brief_path(model_dir, benchmark, run_index), {
        "evaluation_id": f"eval_{int(started_at)}",
        "environment": benchmark,
        "resolved": resolved,
        "total": total,
        "score": resolved / total if total else 0.0,
        "status": "completed",
        "timestamp": dt.datetime.fromtimestamp(started_at).isoformat(),
        "duration": str(dt.timedelta(seconds=time.time() - started_at)),
        "analysis": {},
    })
    return True


def build_parser(defaults=None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    sampling = defaults.get("sampling") or {}
    parser = argparse.ArgumentParser(description="Evaluate one model across math benchmarks")
    parser.add_argument("--eval-config")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY", "dummy"))
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--benchmark-list", required=True)
    parser.add_argument("--benchmark-config", default=defaults.get("benchmark_config"))
    parser.add_argument("--judge-api-config", default=defaults.get("judge_api_config"))
    parser.add_argument("--judge-backend", default=defaults.get("judge_backend"))
    parser.add_argument("--judge-prompt", default=defaults.get(
        "judge_prompt", str(SCRIPT_DIR / "config/prompts/judge_prompt.yaml")))
    parser.add_argument("--comparison-id", default=defaults.get("comparison_id", "comparison"))
    parser.add_argument("--output-root", default=defaults.get("output_root", str(SCRIPT_DIR / "eval_results")))
    parser.add_argument("--passk", type=int, default=sampling.get("pass_k"))
    parser.add_argument("--temperature", type=float, default=float(sampling.get("temperature", 0.6)))
    parser.add_argument("--top-p", type=float, default=float(sampling.get("top_p", 1.0)))
    parser.add_argument("--max-tokens", type=int, default=int(sampling.get("max_tokens", 32768)))
    parser.add_argument("--max-concurrent", type=int, default=int(sampling.get("max_concurrent", 40)))
    parser.add_argument("--no-thinking", action="store_true", default=bool(sampling.get("no_thinking", False)))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--eval-config")
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = {}
    if config_args.eval_config:
        defaults = load_yaml_mapping(Path(config_args.eval_config).resolve())
    args = build_parser(defaults).parse_args(argv)
    if not args.benchmark_config or not args.judge_api_config or not args.judge_backend:
        raise ValueError("benchmark_config, judge_api_config and judge_backend are required via CLI or --eval-config")
    validate_name(args.model_label, "model label")
    validate_name(args.comparison_id, "comparison id")
    if args.passk is not None and args.passk < 1:
        raise ValueError("--passk must be at least 1")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")

    benchmark_config_path = Path(args.benchmark_config).resolve()
    benchmark_config = load_yaml_mapping(benchmark_config_path)
    selected_names = select_benchmarks(benchmark_config, args.benchmark_list)
    judge_config = load_judge_config(Path(args.judge_api_config).resolve(), args.judge_backend)
    judge_template = load_yaml_field(Path(args.judge_prompt).resolve(), "judge")

    loaded = []
    for name in selected_names:
        spec = benchmark_config["benchmarks"][name]
        if not isinstance(spec, dict) or not spec.get("path"):
            raise ValueError(f"benchmark '{name}' requires path")
        path = resolve_path(str(spec["path"]), benchmark_config_path)
        problems = load_benchmark(path, spec)
        pass_k = args.passk if args.passk is not None else int(spec.get("pass_k", 1))
        if pass_k < 1:
            raise ValueError(f"benchmark '{name}' pass_k must be at least 1")
        loaded.append((name, problems, pass_k, path))

    output_root = Path(args.output_root).resolve()
    model_dir = output_root / args.comparison_id / args.model_label
    total_tasks = sum(len(problems) * pass_k for _, problems, pass_k, _ in loaded)
    if args.dry_run:
        print(f"[DRY-RUN] model={args.model_label} served_model={args.served_model}")
        print(f"[DRY-RUN] api_url={safe_api_url(args.api_url)} judge={args.judge_backend}")
        for name, problems, pass_k, path in loaded:
            print(f"[DRY-RUN] benchmark={name} path={path} problems={len(problems)} pass_k={pass_k}")
        print(f"[DRY-RUN] tasks={total_tasks} output={model_dir}")
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = model_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()

    def log(message: str, benchmark: str = "evaluation") -> None:
        line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with log_lock:
            print(line, flush=True)
            with (logs_dir / f"{benchmark}.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    rollout_client = ChatClient(
        base_urls=[args.api_url], model=args.served_model,
        api_key=args.api_key, max_concurrency=args.max_concurrent,
    )
    judge_client = ChatClient(
        base_urls=judge_config["base_urls"], model=judge_config["model"],
        api_key=judge_config.get("api_key", "dummy"),
        max_concurrency=judge_config.get("max_concurrency"),
    )
    judge_max_tokens = int(judge_config.get("judge_max_tokens", 8192))
    judge_temperature = float(judge_config.get("judge_temperature", 0.1))

    write_json(model_dir / "manifest.json", {
        "schema_version": 1,
        "comparison_id": args.comparison_id,
        "model_label": args.model_label,
        "served_model": args.served_model,
        "api_url": safe_api_url(args.api_url),
        "judge_backend": args.judge_backend,
        "benchmarks": {
            name: {"problems": len(problems), "pass_k": pass_k}
            for name, problems, pass_k, _ in loaded
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "no_thinking": args.no_thinking,
        },
        "updated_at": dt.datetime.now().astimezone().isoformat(),
    })

    for benchmark, problems, pass_k, _ in loaded:
        started_at = time.time()
        todo = []
        for run_index in range(1, pass_k + 1):
            for item_index in range(len(problems)):
                path = trajectory_path(model_dir, benchmark, run_index, item_index)
                if not is_valid_trajectory(path):
                    todo.append((run_index, item_index))
        log(f"start problems={len(problems)} pass_k={pass_k} todo={len(todo)}", benchmark)

        def evaluate_one(run_index: int, item_index: int) -> bool:
            problem = problems[item_index]
            messages = [{"role": "user", "content": problem["problem"]}]
            try:
                content, reasoning = rollout_client.chat_with_retry(
                    messages,
                    max_retries=3,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=1800,
                    no_thinking=args.no_thinking,
                )
                predicted = extract_boxed(content)
                score = 0.0
                if predicted:
                    judge_prompt = render_template(
                        judge_template,
                        problem=problem["problem"],
                        predicted_answer=predicted,
                        expected_answer=problem["answer"],
                    )
                    judge_content, judge_reasoning = judge_client.chat_with_retry(
                        [{"role": "user", "content": judge_prompt}],
                        max_retries=3,
                        max_tokens=judge_max_tokens,
                        temperature=judge_temperature,
                        timeout=600,
                    )
                    verdict = parse_correctness_judgement(judge_content, judge_reasoning)
                    if verdict is None:
                        raise ValueError("unparseable judge response")
                    score = 1.0 if verdict else 0.0
                write_json(trajectory_path(model_dir, benchmark, run_index, item_index), {
                    "example_id": problem["example_id"],
                    "rollout_id": run_index,
                    "score": score,
                    "answer": problem["answer"],
                    "prompt": messages,
                    "response": [{
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                    }],
                })
                return True
            except Exception as exc:
                log(f"failed run={run_index} item={item_index}: {type(exc).__name__}: {exc}", benchmark)
                return False

        completed = 0
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
            futures = [executor.submit(evaluate_one, run_index, item_index)
                       for run_index, item_index in todo]
            for future in as_completed(futures):
                completed += bool(future.result())

        complete_runs = 0
        for run_index in range(1, pass_k + 1):
            complete_runs += write_brief(
                model_dir, benchmark, run_index, len(problems), started_at
            )
        log(f"done completed_tasks={completed}/{len(todo)} complete_runs={complete_runs}/{pass_k}", benchmark)
        if complete_runs != pass_k:
            log("benchmark remains incomplete; rerun the same command to resume", benchmark)
            return 1

    log(f"all benchmarks completed; output={model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
