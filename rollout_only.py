#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rollout_only.py — Qwen 推理, 不依赖 Kimi Judge, 写入中间文件 trajectory_raw.jsonl."""

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
import httpx
from openai import OpenAI

from share_utils import ChatClient


# ==========================================
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


# ==========================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-workers", type=int, default=128)
    p.add_argument("--run-id", default="run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stats-interval", type=int, default=300)
    args = p.parse_args()

    # suppress proxy
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)

    # config
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = cfg["dataset"]
    uuid_key = ds["uuid_field"]
    prob_key = ds["problem_field"]
    ans_key = ds["expected_answer_field"]

    # dataset
    dset_path = Path(args.dataset).resolve()
    rows = []
    with open(dset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if args.limit:
        rows = rows[:args.limit]
    print(f"[INFO] loaded {len(rows)} records from {dset_path}")

    # output
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "trajectory_raw.jsonl"

    # crash recovery
    done = load_completed_uuids(raw_path)
    todo = [r for r in rows if r.get(uuid_key) not in done]
    print(f"[INFO] already done: {len(done)}, todo: {len(todo)}")
    if not todo:
        print("[INFO] nothing to do.")
        return

    # Qwen client (dual pool: apex local 8 H800 + remote GPU)
    apex = cfg["qwen_apex"]
    remote = cfg.get("qwen_remote")  # Might not exist
    base_urls_2 = remote["base_urls"] if remote else None
    max_concurrency_2 = remote.get("max_concurrency") if remote else None

    qwen_client = ChatClient(
        base_urls=apex["base_urls"],
        model=apex["model"],
        api_key=apex.get("api_key", "dummy"),
        max_concurrency=apex.get("max_concurrency"),
        base_urls_2=base_urls_2,
        max_concurrency_2=max_concurrency_2,
    )

    inf_cfg = cfg["inference"]

    raw_fp = open(raw_path, "a", encoding="utf-8")
    raw_lock = threading.Lock()

    stats_ok = 0
    stats_fail = 0
    stats_lock = threading.Lock()

    def rollout_one(it):
        nonlocal stats_ok, stats_fail
        uid = it.get(uuid_key)
        prob = it.get(prob_key, "")
        ans = it.get(ans_key, None)

        try:
            user_prompt = ("Solve the following math problem.\n"
                           "Make sure to put the answer (and only answer) "
                           f"inside \\boxed{{}}.\n\n{prob}")
            msgs = [{"role": "user", "content": user_prompt}]
            content, reasoning = qwen_client.chat(
                messages=msgs,
                max_tokens=inf_cfg["max_tokens"],
                temperature=inf_cfg.get("temperature", 0.7),
                top_p=inf_cfg.get("top_p", 0.95))

            parts = []
            if reasoning:
                parts.append(" thinking\n" + reasoning + "\n response")
            if content:
                parts.append(content)
            traj = "\n".join(parts).strip()

            # SFT-friendly OpenAI messages format (reasoning_content / content split)
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant",
                 "reasoning_content": reasoning,
                 "content": content},
            ]

            record = {"uuid": uid, "problem": prob, "expected_answer": ans,
                      "trajectory": traj, "messages": messages}

            with raw_lock:
                raw_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_fp.flush()
            with stats_lock:
                stats_ok += 1
            return True
        except Exception as e:
            with stats_lock:
                stats_fail += 1
            print(f"[ROLLOUT_FAIL] uuid={uid} {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            return False

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(rollout_one, r): r for r in todo}
        for fut in as_completed(futures):
            fut.result()

    raw_fp.close()

    elapsed = time.time() - t0
    rate = (stats_ok / elapsed * 60.0) if elapsed > 0 else 0.0

    print(f"\n[rollout] ok={stats_ok} failed={stats_fail}"
          f" elapsed={elapsed:.0f}s rate={rate:.1f}/min -> {raw_path}\n")


if __name__ == "__main__":
    main()