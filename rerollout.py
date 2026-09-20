#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rerollout — second-round Qwen reasoning for "stuck in repetition loop" items.

Reads first-round result_*.jsonl, classifies repetition items,
re-runs Qwen with the original trajectory + correct answer as hints,
judges each attempt via Kimi, and emits three output files:

  - passed.jsonl        — non-repetition items (straight through)
  - retry_correct.jsonl  — repetition items that became correct after retry
  - retry_failed.jsonl    — repetition items still incorrect after N attempts

Supports crash recovery: items already in retry_correct.jsonl or retry_failed.jsonl
are auto-skipped on restart (uuid-based dedup).
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


# ============================================================
# ChatClient
# ============================================================
class ChatClient:
    """OpenAI-compatible multi-endpoint round-robin client with semaphore limiting."""

    def __init__(
        self,
        base_urls: List[str],
        model: str,
        api_key: str = "dummy",
        max_concurrency: Optional[int] = None,
    ):
        if not base_urls:
            raise ValueError("base_urls cannot be empty")
        self.base_urls = [u.rstrip("/") for u in base_urls]
        self.model = model
        self.api_key = api_key
        self._rr_idx = 0
        self._rr_lock = threading.Lock()
        self._sema = threading.BoundedSemaphore(max_concurrency) if (max_concurrency and max_concurrency > 0) else None
        self._init_clients()

    def _init_clients(self):
        from openai import OpenAI
        import httpx
        http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=256, max_connections=25000),
            timeout=httpx.Timeout(600.0, connect=10.0),
        )
        self._clients = [
            OpenAI(base_url=u, api_key=self.api_key, http_client=http_client)
            for u in self.base_urls
        ]

    def chat(self, messages, max_tokens=64000, temperature=0.7, top_p=0.95,
             timeout=1800):
        with self._rr_lock:
            idx = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % len(self._clients)
        client = self._clients[idx]
        if self._sema is not None:
            self._sema.acquire()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
            )
        finally:
            if self._sema is not None:
                self._sema.release()
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "") or ""
        return content, reasoning


# ============================================================
# Prompt rendering
# ============================================================
def load_yaml_key(yaml_path: Path, key: str) -> str:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if key not in data:
        raise ValueError(f"{yaml_path} missing '{key}' field")
    return data[key]


def render_template(template: str, **kwargs) -> str:
    """Render {key} placeholders only, leaving {{ and }} intact."""
    S_OPEN = "\x00LB\x00"
    S_CLOSE = "\x00RB\x00"
    text = template.replace("{{", S_OPEN).replace("}}", S_CLOSE)
    for key, val in kwargs.items():
        text = text.replace("{" + key + "}", str(val))
    text = text.replace(S_OPEN, "{").replace(S_CLOSE, "}")
    return text


# ============================================================
# Answer extraction
# ============================================================
_boxed_re = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} in text; handles nested braces."""
    if not text:
        return None
    last = None
    for m in _boxed_re.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "\\" and i + 1 < len(text):
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last = text[start:i]
                    break
            i += 1
    return last


# ============================================================
# Correctness judge
# ============================================================
_judge_re = re.compile(r"Judgement\s*[:：]\s*(Yes|No)", re.IGNORECASE)


def parse_correctness_judgement(content: str, reasoning: str) -> Optional[bool]:
    """Parse Yes/No from judgement text. Returns True/False, or None if unparseable."""
    for text in (content, reasoning):
        if not text:
            continue
        m = _judge_re.search(text)
        if m:
            ans = m.group(1).lower()
            return ans.startswith("y")
        # fallback: last non-empty line
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines:
            tail = lines[-1].lower()
            if tail.startswith(("yes", "true", "是")):
                return True
            if tail.startswith(("no", "false", "否")):
                return False
    return None


# ============================================================
# Judge with retry (exponential backoff)
# ============================================================
def judge_with_retry(kimi_client: ChatClient, messages, max_tokens=100000,
                     temperature=0.1, max_retries=5, timeout=1200):
    for attempt in range(max_retries):
        try:
            return kimi_client.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [WARN] judge retry {attempt+1}/{max_retries}: "
                  f"{type(e).__name__}: {e}, sleeping {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)


# ============================================================
# Load already-done uuids (crash recovery)
# ============================================================
def load_completed_uuids(*paths: str) -> set:
    uuids = set()
    for p_str in paths:
        fp = Path(p_str)
        if not fp.exists():
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    uuids.add(json.loads(line)["uuid"])
                except Exception:
                    pass
    return uuids


# ============================================================
# Retry one item
# ============================================================
def retry_one(
    item: dict,
    rerollout_template: str,
    judge_template: str,
    qwen_client: ChatClient,
    kimi_client: ChatClient,
    inference_cfg: dict,
    kimi_cfg: dict,
    max_attempts: int,
) -> dict:
    uid = item.get("uuid", "?")
    problem = item.get("problem", "")
    expected = str(item.get("expected_answer", ""))
    first_trajectory = item.get("trajectory", "")

    retries_payload = []

    for attempt in range(max_attempts):
        try:
            # 1. Render rerollout prompt with hints
            rendered = render_template(
                rerollout_template,
                trajectory=first_trajectory,
                problem=problem,
                expected_answer=expected,
            )
            msgs = [{"role": "user", "content": rendered}]

            # 2. Qwen re-reasoning
            content, reasoning = qwen_client.chat(
                messages=msgs,
                max_tokens=inference_cfg.get("max_tokens", 64000),
                temperature=inference_cfg.get("temperature", 0.7),
                top_p=inference_cfg.get("top_p", 0.95),
            )

            parts = []
            if reasoning:
                parts.append(f" thinking\n{reasoning}\n response")
            if content:
                parts.append(content)
            new_traj = "\n".join(parts).strip()

            # 3. Extract predicted answer
            pred = extract_boxed(content) or extract_boxed(reasoning) or ""

            # 4. Judge correctness
            judge_prompt = render_template(
                judge_template,
                problem=problem,
                predicted_answer=(pred if pred else "(empty)"),
                expected_answer=expected,
            )
            j_content, j_reasoning = judge_with_retry(
                kimi_client,
                [{"role": "user", "content": judge_prompt}],
                max_tokens=kimi_cfg.get("judge_max_tokens", 100000),
                temperature=kimi_cfg.get("judge_temperature", 0.1),
            )
            is_correct = parse_correctness_judgement(j_content, j_reasoning)

            retries_payload.append({
                "attempt": attempt + 1,
                "predicted_answer": pred,
                "new_trajectory": new_traj,
                "is_correct": is_correct,
            })

            if is_correct:
                result = dict(item)
                result["retry_ok"] = True
                result["retry_attempts"] = retries_payload
                result["final_trajectory"] = new_traj
                result["final_predicted_answer"] = pred
                # SFT-friendly OpenAI messages format (clean user prompt + reasoning/content split)
                user_prompt = (
                    "Solve the following math problem.\n"
                    "Make sure to put the answer (and only answer) inside "
                    "\\boxed{}.\n\n" + problem
                )
                result["final_messages"] = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant",
                     "reasoning_content": reasoning,
                     "content": content},
                ]
                return result

        except Exception as e:
            retries_payload.append({
                "attempt": attempt + 1,
                "error": f"{type(e).__name__}: {e}",
            })

    # All attempts exhausted
    result = dict(item)
    result["retry_ok"] = False
    result["retry_attempts"] = retries_payload
    return result


# ============================================================
# Statistics (thread-safe)
# ============================================================
class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1

    def read(self):
        with self._lock:
            return self._value


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Rerollout — second-round Qwen reasoning for repetition items")
    parser.add_argument("--config", required=True, help="Path to inference_config.yaml")
    parser.add_argument("--input-file", required=True, help="Path to result_*.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for rerollout results")
    parser.add_argument("--max-workers", type=int, default=32,
                       help="Concurrent workers (default: 32)")
    parser.add_argument("--stats-interval", type=int, default=300,
                       help="Stats print interval in seconds (default: 300)")
    args = parser.parse_args()

    # ---- Load config ----
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rerollout_cfg = cfg.get("rerollout", {})
    max_attempts = rerollout_cfg.get("max_rerollout_attempts", 3)
    prompt_rel = rerollout_cfg.get("prompt", "config/turn2_rerollout.yaml")

    # Resolve rerollout prompt template path relative to config directory
    cfg_dir = cfg_path.parent
    rerollout_prompt_path = (cfg_dir / ".." / prompt_rel).resolve()
    if not rerollout_prompt_path.exists():
        # Fallback: try relative to cwd
        rerollout_prompt_path = Path(prompt_rel).resolve()

    rerollout_template = load_yaml_key(rerollout_prompt_path, "rerollout")
    print(f"[INFO] rerollout prompt loaded from: {rerollout_prompt_path}")

    # Resolve correctness judge template
    judge_rel = cfg["prompts"]["correctness"]
    judge_path = (cfg_dir / ".." / judge_rel).resolve()
    if not judge_path.exists():
        judge_path = (cfg_dir / Path(judge_rel).name).resolve()
    correctness_template = load_yaml_key(judge_path, "judge")
    print(f"[INFO] judge prompt loaded from: {judge_path}")

    inference_cfg = cfg.get("inference", {})
    kimi_cfg = dict(cfg["kimi"])

    # ---- Read input JSONL & classify ----
    input_path = Path(args.input_file)
    dataset = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip bad JSON line: {e}", file=sys.stderr, flush=True)

    print(f"[INFO] loaded {len(dataset)} records from {input_path}")

    rep_items = [it for it in dataset if it.get("is_repetition")]
    nonrep_items = [it for it in dataset if not it.get("is_repetition")]
    print(f"[INFO] non-repetitions (-> passed): {len(nonrep_items)}")
    print(f"[INFO] repetitions (-> rerollout): {len(rep_items)}")

    # ---- Create output dir ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    passed_file = out_dir / "passed.jsonl"
    retry_correct_file = out_dir / "retry_correct.jsonl"
    retry_failed_file = out_dir / "retry_failed.jsonl"

    # ---- Write non-repetition items to passed.jsonl ----
    with open(passed_file, "w", encoding="utf-8") as fp:
        for it in nonrep_items:
            fp.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[INFO] {len(nonrep_items)} items -> passed.jsonl")

    # ---- Crash recovery: which repetition uuids already done ----
    completed_uuids = load_completed_uuids(
        str(retry_correct_file), str(retry_failed_file))
    todo = [it for it in rep_items if it.get("uuid") not in completed_uuids]
    skipped = len(rep_items) - len(todo)

    print(f"[INFO] repetition: total={len(rep_items)}, todo={len(todo)}, skipped={skipped}")

    if not todo:
        print("[INFO] no repetition items left to reroll.")
        return

    # ---- Init clients ----
    qwen_urls = cfg["qwen"].get("base_urls") or [cfg["qwen"]["base_url"]]
    kimi_urls = cfg["kimi"].get("base_urls") or [cfg["kimi"]["base_url"]]

    qwen_client = ChatClient(
        base_urls=qwen_urls,
        model=cfg["qwen"]["model"],
        api_key=cfg["qwen"].get("api_key", "dummy"),
        max_concurrency=args.max_workers * 2,
    )
    kimi_client = ChatClient(
        base_urls=kimi_urls,
        model=cfg["kimi"]["model"],
        api_key=cfg["kimi"].get("api_key", "dummy"),
        max_concurrency=args.max_workers * 2,
    )

    # ---- Append-mode output file handles ----
    correct_fp = open(retry_correct_file, "a", encoding="utf-8")
    failed_fp = open(retry_failed_file, "a", encoding="utf-8")
    lock_correct = threading.Lock()
    lock_failed = threading.Lock()

    # ---- Atomic counters for stats ----
    cnt_ok = AtomicCounter()
    cnt_fail = AtomicCounter()

    start_time = time.time()
    last_print_time = {"t": start_time}
    lock_print = threading.Lock()

    def print_stats():
        elapsed = time.time() - start_time
        ok = cnt_ok.read()
        fail = cnt_fail.read()
        done = ok + fail
        rate = (done / elapsed * 60.0) if elapsed > 0 else 0.0
        pct = (ok / done * 100.0) if done > 0 else 0.0
        print(f"[STATS] ok={ok}  fail={fail}  total={done}/{len(todo)}  "
              f"elapsed={elapsed:.0f}s  rate={rate:.1f}/min  success_rate={pct:.1f}%")

    def maybe_print_stats(force=False):
        now = time.time()
        with lock_print:
            if not force and (now - last_print_time["t"]) < args.stats_interval:
                return
            last_print_time["t"] = now
        print_stats()

    # ---- Main worker ----
    def worker(item: dict):
        result = retry_one(
            item=item,
            rerollout_template=rerollout_template,
            judge_template=correctness_template,
            qwen_client=qwen_client,
            kimi_client=kimi_client,
            inference_cfg=inference_cfg,
            kimi_cfg=kimi_cfg,
            max_attempts=max_attempts,
        )
        if result.get("retry_ok"):
            with lock_correct:
                correct_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                correct_fp.flush()
            cnt_ok.increment()
        else:
            with lock_failed:
                failed_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                failed_fp.flush()
            cnt_fail.increment()

        maybe_print_stats()

    # ---- Concurrent execution ----
    max_workers = args.max_workers

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, it): it for it in todo}
        for fut in as_completed(futures):
            fut.result()
        print_stats()

    correct_fp.close()
    failed_fp.close()

    # ---- Final summary ----
    total_elapsed = time.time() - start_time
    ok = cnt_ok.read()
    fail = cnt_fail.read()
    total_rerolled = ok + fail

    print()
    print("========== REROLLOUT FINISHED ==========")
    print(f"  Total input:           {len(dataset)}")
    print(f"  Passed-through:        {len(nonrep_items)}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Rerolled:               {total_rerolled}")
    print(f"  Retry correct:         {ok}")
    print(f"  Retry failed:          {fail}")
    if total_rerolled > 0:
        ok_rate = ok / total_rerolled * 100.0
        speed = total_rerolled / total_elapsed * 60.0
        print(f"  Wall time:              {total_elapsed:.0f}s "
              f"| rate={speed:.1f}/min | success_rate={ok_rate:.1f}%")
    else:
        print(f"  Wall time:              {total_elapsed:.0f}s")
    print(f"  Output dir:            {out_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()