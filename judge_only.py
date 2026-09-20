#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_only.py — standalone Kimi Judge (no Qwen rollout).

Reads trajectory_raw.jsonl (from rollout_only.py), runs both repetition and
correctness judgment via Kimi, with CRASH RECOVERY: items already judged
(uuid exists in result_{run_id}.jsonl) are auto-skipped on restart.

Usage:
    python judge_only.py \
        --config config/inference_config.yaml \
        --trajectory-file ./output/<ts>/trajectory_raw.jsonl \
        --output-dir ./output/<ts> \
        --run-id run \
        --max-workers 64 \
        --stats-interval 300
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Import shared utilities
from share_utils import (
    ChatClient,
    extract_boxed,
    load_prompt_template,
    parse_correctness_judgement,
    parse_repetition_judgement,
    render_template,
)

# ---------- batch-size marker / buffers ----------
RESULT_BATCH = 50
STATS_INTERVAL = 300


# ---------- stats (thread-safe) ----------
class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, delta=1):
        with self._lock:
            self._value += delta

    def read(self):
        with self._lock:
            return self._value


# ---------- cras recovery: read already-judged uuids from result file ----------
def load_completed_uuids(path):
    uuids = set()
    if not path.exists():
        return uuids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                uuids.add(json.loads(line)["uuid"])
            except Exception:
                pass
    return uuids


def main():
    parser = argparse.ArgumentParser(description="Kimi Judge (standalone)")
    parser.add_argument("--config", required=True, help="Path to inference_config.yaml")
    parser.add_argument("--trajectory-file", required=True, help="Path to trajectory_raw.jsonl from rollout phase")
    parser.add_argument("--output-dir", required=True, help="Output directory for judged results")
    parser.add_argument("--run-id", default="run", help="run id for result filenames")
    parser.add_argument("--max-workers", type=int, default=64, help="Concurrent Kimi judge workers")
    parser.add_argument("--stats-interval", type=int, default=300, help="stats log interval (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N items")
    args = parser.parse_args()

    # ---------- suppress proxy ----------
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)

    # ---------- load config ----------
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---------- output dir ----------
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load prompt templates ----------
    cfg_dir = cfg_path.parent
    prompts_cfg = cfg.get("prompts", {})

    rep_rel = prompts_cfg["repetition"]
    rep_yaml = (cfg_dir.parent / rep_rel) if not Path(rep_rel).is_absolute() else Path(rep_rel)
    if not rep_yaml.exists():
        rep_yaml = cfg_dir / Path(rep_rel).name
    rep_template = load_prompt_template(rep_yaml)

    corr_rel = prompts_cfg["correctness"]
    corr_yaml = (cfg_dir.parent / corr_rel) if not Path(corr_rel).is_absolute() else Path(corr_rel)
    if not corr_yaml.exists():
        corr_yaml = cfg_dir / Path(corr_rel).name
    corr_template = load_prompt_template(corr_yaml)

    print(f"[INFO] repetition prompt: {rep_yaml}")
    print(f"[INFO] correctness prompt: {corr_yaml}")

    # ---------- Kimi client ----------
    kimi_cfg = cfg["kimi"]
    kimi_urls = kimi_cfg.get("base_urls") or [kimi_cfg["base_url"]]

    kimi_client = ChatClient(
        base_urls=kimi_urls,
        model=kimi_cfg["model"],
        api_key=kimi_cfg.get("api_key", "dummy"),
        max_concurrency=kimi_cfg.get("max_concurrency"),
    )
    print(f"[INFO] Kimi endpoints: {kimi_urls} (concurrency={kimi_cfg.get('max_concurrency')})")

    # ---------- read trajectory_raw.jsonl ----------
    traj_path = Path(args.trajectory_file)
    trajectory_items = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trajectory_items.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip bad line in trajectory file: {e}", file=sys.stderr, flush=True)

    if args.limit:
        trajectory_items = trajectory_items[: args.limit]

    print(f"[INFO] loaded {len(trajectory_items)} items from {traj_path}")

    # ---------- output files for judged results ----------
    result_path = out_dir / f"result_{args.run_id}.jsonl"
    rep_path = out_dir / f"repetition_{args.run_id}.jsonl"

    # ---------- crash recovery: skip already-judged uuids ----------
    completed_uuids = load_completed_uuids(result_path)
    todo = [it for it in trajectory_items if it["uuid"] not in completed_uuids]
    skipped = len(trajectory_items) - len(todo)
    print(f"[INFO] crash recovery: already judged={len(completed_uuids)}, todo={len(todo)}, skipped={skipped}")

    if not todo:
        print("[INFO] nothing to judge. exiting.")
        return

    # ---------- open output files append ----------
    result_fp = open(result_path, "a", encoding="utf-8")
    rep_fp = open(rep_path, "a", encoding="utf-8")
    result_lock = threading.Lock()
    rep_lock = threading.Lock()

    # ---------- thread-safe counters ----------
    cnt_ok = AtomicCounter()
    cnt_fail = AtomicCounter()
    cnt_rep = AtomicCounter()
    cnt_rep_correct = AtomicCounter()
    start_time = time.time()
    last_print = {"t": start_time}
    print_lock = threading.Lock()

    def print_stats():
        elapsed = time.time() - start_time
        total = cnt_ok.read() + cnt_fail.read()
        rate = total / elapsed * 60.0 if elapsed > 0 else 0.0
        print(
            f"[STATS] total={total}/{len(todo)} repetition={cnt_rep.read()} "
            f"rep&correct={cnt_rep_correct.read()} failures={cnt_fail.read()} "
            f"elapsed={elapsed:.0f}s rate={rate:.1f}/min", flush=True
        )

    def process_one(item):
        uid = item["uuid"]
        prob = item.get("problem", "")
        expected = str(item.get("expected_answer", ""))
        traj = item.get("trajectory", "")

        try:
            # 1. repetition judge
            rep_prompt = render_template(rep_template, trajectory=traj)
            rep_content, rep_reasoning = kimi_client.chat(
                [{"role": "user", "content": rep_prompt}],
                max_tokens=kimi_cfg.get("judge_max_tokens", 4096),
                temperature=kimi_cfg.get("judge_temperature", 0.1),
            )
            is_rep, analysis = parse_repetition_judgement(rep_content, rep_reasoning)

            # 2. extract predicted answer
            predicted = extract_boxed(traj) or ""

            # 3. correctness judge
            corr_prompt = render_template(
                corr_template,
                problem=prob,
                predicted_answer=(predicted if predicted else "(empty)"),
                expected_answer=(expected if expected else "(unknown)"),
            )
            corr_content, corr_reasoning = kimi_client.chat(
                [{"role": "user", "content": corr_prompt}],
                max_tokens=kimi_cfg.get("judge_max_tokens", 4096),
                temperature=kimi_cfg.get("judge_temperature", 0.1),
            )
            is_correct = parse_correctness_judgement(corr_content, corr_reasoning)

        except Exception as e:
            cnt_fail.increment()
            print(f"[ERROR] uuid={uid}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return False

        # ---------- write result ----------
        result = {
            "uuid": uid,
            "problem": prob,
            "expected_answer": expected,
            "trajectory": traj,
            "predicted_answer": predicted,
            "is_repetition": is_rep,
            "is_correct": is_correct,
            "_messages": {
                "rep_prompt": rep_prompt,
                "rep_content": rep_content,
                "rep_reasoning": rep_reasoning,
                "rep_analysis": analysis,
                "corr_prompt": corr_prompt,
                "corr_content": corr_content,
                "corr_reasoning": corr_reasoning,
            },
        }

        with result_lock:
            result_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            result_fp.flush()

        if is_rep:
            with rep_lock:
                rep_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                rep_fp.flush()
            cnt_rep.increment()
            if is_correct is True:
                cnt_rep_correct.increment()

        cnt_ok.increment()
        return True

    # ---------- periodic stats logger ----------
    stats_cancelled = threading.Event()

    def stats_loop():
        while not stats_cancelled.wait(args.stats_interval):
            with print_lock:
                print_stats()

    t_logger = threading.Thread(target=stats_loop, daemon=True)
    t_logger.start()

    # ---------- worker pool ----------
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_one, it): it for it in todo}
        for fut in as_completed(futures):
            fut.result()

    stats_cancelled.set()
    result_fp.close()
    rep_fp.close()

    # ---------- final stats ----------
    total_elapsed = time.time() - start_time
    total_done = cnt_ok.read() + cnt_fail.read()
    rate = total_done / total_elapsed * 60.0 if total_elapsed > 0 else 0.0
    print("\n========== JUDGE FINISHED ==========")
    print(f"  Items processed: {total_done}")
    print(f"  Repetitions:    {cnt_rep.read()}")
    print(f"  Rep&Correct:    {cnt_rep_correct.read()}")
    print(f"  Failures:       {cnt_fail.read()}")
    print(f"  Wall time:      {total_elapsed:.0f}s  rate={rate:.1f}/min")
    print(f"  Output:         {result_path}")
    print(f"  Rep-tracker:    {rep_path}")
    print("======================================\n")


if __name__ == "__main__":
    main()