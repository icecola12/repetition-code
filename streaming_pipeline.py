#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""streaming_pipeline.py — fully streaming rollout -> judge -> rerollout.

Three chained thread pools with submit-on-completion dispatch:
  pool R (rollout)  -> pool J (judge)  -> pool RR (rerollout)

Outputs (each is a resumable checkpoint by uuid):
  <out>/trajectory_raw.jsonl     -- rollout done
  <out>/result_<run>.jsonl       -- judge done
  <out>/repetition_<run>.jsonl   -- subset (is_repetition=True)
  <out>/rerollout/retry_correct.jsonl
  <out>/rerollout/retry_failed.jsonl

Restart skips uuids already present in each output.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from share_utils import (
    ChatClient,
    MultiJudgeClient,
    extract_boxed,
    load_prompt_template,
    load_yaml_field,
    parse_correctness_judgement,
    parse_repetition_judgement,
    render_template,
)


# ---------- thread-safe counter ----------
class Counter:
    def __init__(self, v=0):
        self._v = v
        self._l = threading.Lock()

    def inc(self, d=1):
        with self._l:
            self._v += d

    def dec(self, d=1):
        with self._l:
            self._v -= d

    def get(self):
        with self._l:
            return self._v


# ---------- load completed uuids from a jsonl file ----------
def load_uuids(path: Path) -> set:
    uuids = set()
    if not path.exists():
        return uuids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u = json.loads(line).get("uuid")
                if u:
                    uuids.add(u)
            except Exception:
                pass
    return uuids


def load_records(path: Path, limit=None):
    """Load rows from a JSONL file, stopping once limit is reached."""
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
            except Exception:
                pass
    return rows


# ---------- build trajectory string ----------
def build_trajectory(content: str, reasoning: str) -> str:
    parts = []
    if reasoning:
        parts.append(" thinking\n" + reasoning + "\n response")
    if content:
        parts.append(content)
    return "\n".join(parts).strip()


def trajectory_from_messages(messages: list) -> str:
    """从 messages 取最后一条 assistant 消息现算 trajectory 字符串，只用于喂
    judge/rerollout prompt 模板的 {trajectory} 占位符，不落盘。"""
    if not messages:
        return ""
    assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if not assistant:
        return ""
    return build_trajectory(assistant.get("content", ""), assistant.get("reasoning_content", ""))


def main():
    parser = argparse.ArgumentParser(description="Streaming pipeline (rollout -> judge -> rerollout)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollout-api-config",
                        help="Separate rollout API YAML; falls back to --config")
    parser.add_argument("--judge-api-config",
                        help="Separate judge API YAML; falls back to --config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None,
                        help="rollout worker count (overrides config.rollout.max_workers)")
    parser.add_argument("--judge-workers", type=int, default=None,
                        help="judge worker count (overrides config.judge.max_workers)")
    parser.add_argument("--rerollout-workers", type=int, default=None,
                        help="rerollout worker count (overrides config.rerollout.max_workers)")
    parser.add_argument("--stats-interval", type=int, default=60)
    parser.add_argument("--no-rerollout", action="store_true",
                        help="Disable rerollout stage entirely")
    parser.add_argument("--parallel-tail-threshold", type=int, default=None,
                        help="剩余待处理条数 ≤ 该值时, rerollout 的 attempt 改为并行发起 "
                             "(overrides config.rerollout.parallel_tail_threshold; 0=关闭)")
    args = parser.parse_args()

    # ---- proxy hygiene ----
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)

    # ---- config ----
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_dir = cfg_path.parent

    def _load_config(path):
        with open(Path(path).resolve(), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    rollout_api_cfg = _load_config(args.rollout_api_config) if args.rollout_api_config else cfg
    judge_api_cfg = _load_config(args.judge_api_config) if args.judge_api_config else cfg
    print(f"[INFO] rollout API config = {Path(args.rollout_api_config).resolve() if args.rollout_api_config else cfg_path}"
          f"{' (legacy fallback)' if not args.rollout_api_config else ''}")
    print(f"[INFO] judge API config = {Path(args.judge_api_config).resolve() if args.judge_api_config else cfg_path}"
          f"{' (legacy fallback)' if not args.judge_api_config else ''}")

    ds_cfg = cfg["dataset"]
    uuid_key = ds_cfg["uuid_field"]
    prob_key = ds_cfg["problem_field"]
    ans_key = ds_cfg["expected_answer_field"]

    project_root = cfg_dir.parent.parent if cfg_dir.name == "runtime" else cfg_dir.parent

    # ---- prompts ----
    prompts_cfg = cfg.get("prompts", {})
    def _resolve_prompt(rel):
        path = Path(rel).expanduser()
        candidates = [path] if path.is_absolute() else [
            project_root / path,
            cfg_dir / path,
            cfg_dir / path.name,
            Path.cwd() / path,
        ]
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        raise FileNotFoundError(f"prompt template not found: tried {candidates}")

    rollout_template = load_yaml_field(
        _resolve_prompt(prompts_cfg.get("rollout", "config/prompts/rollout_prompt.yaml")),
        "rollout",
    )
    rep_template = load_prompt_template(_resolve_prompt(prompts_cfg["repetition"]))
    corr_template = load_prompt_template(_resolve_prompt(prompts_cfg["correctness"]))

    rerollout_cfg = cfg.get("rerollout", {})
    rerollout_max_attempts = int(rerollout_cfg.get("max_rerollout_attempts", 3))
    rerollout_force_correct = bool(rerollout_cfg.get("force_correct", False))
    # 剩余待处理条数 ≤ 该阈值时，attempt 由串行改为并行发起（0 = 关闭）。
    # 尾部并发池几乎空转，串行 N 轮往返是纯延迟浪费；代价是判对也会跑满 N 次。
    if args.parallel_tail_threshold is not None:
        rerollout_parallel_tail = int(args.parallel_tail_threshold)
    else:
        rerollout_parallel_tail = int(rerollout_cfg.get("parallel_tail_threshold", 0))
    rerollout_prompt_rel = rerollout_cfg.get("prompt", "config/prompts/turn2_rerollout_v4.yaml")
    rerollout_template = load_yaml_field(_resolve_prompt(rerollout_prompt_rel), "rerollout")

    # Stage1 multi-attempt config
    rollout_count = int((cfg.get("rollout") or {}).get("rollout_count", 1))
    repeat_threshold = int((cfg.get("judge") or {}).get("repeat_threshold", 1))

    # ---- resolve worker counts: CLI > config > hard default ----
    def _resolve_workers(cli_val, cfg_section, default):
        if cli_val is not None:
            return int(cli_val)
        return int((cfg.get(cfg_section) or {}).get("max_workers", default))
    rollout_workers   = _resolve_workers(args.rollout_workers,   "rollout",   128)
    judge_workers     = _resolve_workers(args.judge_workers,     "judge",      64)
    rerollout_workers = _resolve_workers(args.rerollout_workers, "rerollout",  32)
    print(f"[INFO] workers: rollout={rollout_workers} judge={judge_workers} "
          f"rerollout={rerollout_workers}")
    print(f"[INFO] rerollout: max_attempts={rerollout_max_attempts} "
          f"parallel_tail_threshold={rerollout_parallel_tail}")

    # ---- clients ----
    if "qwen_apex" not in rollout_api_cfg:
        raise SystemExit("Rollout API config missing qwen_apex")
    apex = rollout_api_cfg["qwen_apex"]
    remote = rollout_api_cfg.get("qwen_remote")
    qwen_client = ChatClient(
        base_urls=apex["base_urls"],
        model=apex["model"],
        api_key=apex.get("api_key", "dummy"),
        max_concurrency=apex.get("max_concurrency"),
        base_urls_2=(remote["base_urls"] if remote else None),
        max_concurrency_2=(remote.get("max_concurrency") if remote else None),
    )

    # ---- judge clients (round-robin across all configured judges) ----
    judge_backends = []
    for section in ["kimi", "glm", "deepseek_397b"]:
        jc = judge_api_cfg.get(section)
        if not jc:
            continue
        urls = jc.get("base_urls") or [jc["base_url"]]
        judge_backends.append((
            ChatClient(
                base_urls=urls,
                model=jc["model"],
                api_key=jc.get("api_key", "dummy"),
                max_concurrency=jc.get("max_concurrency"),
            ),
            jc,
        ))
    if not judge_backends:
        raise SystemExit("No judge backend configured (kimi/glm/deepseek_397b)")
    judges = MultiJudgeClient(judge_backends)

    # Unified judge params (consistent across backends, use minimum for safety)
    judge_max_tokens = min(
        (jc.get("judge_max_tokens", 65536) for _, jc in judge_backends),
        default=65536,
    )
    judge_temperature = judge_backends[0][1].get("judge_temperature", 0.1)
    judge_infinite_retry = judge_backends[0][1].get("judge_infinite_retry", False)
    print(f"[INFO] judge backends: {len(judge_backends)} ("
          f"{', '.join(jc.get('model_name', jc['model'][:20]) for _, jc in judge_backends)})"
          f" max_tokens={judge_max_tokens}")

    inf_cfg = cfg.get("inference", {})

    # Retry knobs (per-stage). Configurable via inference_config.yaml:
    #   retry:
    #     rollout:  { max_retries: 3, backoff_base: 2.0, backoff_max: 30.0 }
    #     judge:    { max_retries: 5, backoff_base: 2.0, backoff_max: 30.0 }
    #     rerollout:{ max_retries: 3, backoff_base: 2.0, backoff_max: 30.0 }
    retry_cfg = cfg.get("retry", {}) or {}
    def _rk(stage, key, default):
        return (retry_cfg.get(stage) or {}).get(key, default)
    rollout_retry = {
        "max_retries":  int(_rk("rollout", "max_retries", 3)),
        "backoff_base": float(_rk("rollout", "backoff_base", 2.0)),
        "backoff_max":  float(_rk("rollout", "backoff_max", 30.0)),
    }
    judge_retry = {
        "max_retries":  int(_rk("judge", "max_retries", 5)),
        "backoff_base": float(_rk("judge", "backoff_base", 2.0)),
        "backoff_max":  float(_rk("judge", "backoff_max", 30.0)),
    }
    rerollout_retry = {
        "max_retries":  int(_rk("rerollout", "max_retries", 3)),
        "backoff_base": float(_rk("rerollout", "backoff_base", 2.0)),
        "backoff_max":  float(_rk("rerollout", "backoff_max", 30.0)),
    }
    print(f"[INFO] retry config: rollout={rollout_retry}  judge={judge_retry}  "
          f"rerollout={rerollout_retry}")
    print(f"[INFO] stage1: rollout_count={rollout_count} repeat_threshold={repeat_threshold}")

    # ---- output dirs (new layout: stage1/ + stage2/) ----
    out_dir = Path(args.output_dir).resolve()
    stage1_dir = out_dir / "stage1"
    stage2_dir = out_dir / "stage2"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # stage1 files: trajectory + judge classification
    traj_path = stage1_dir / "trajectory_raw.jsonl"
    no_rep_path = stage1_dir / "no_repetition.jsonl"
    has_rep_path = stage1_dir / "has_repetition.jsonl"

    # stage2 files: rerollout terminal outcomes (also includes first-round correct repetitions)
    rep_correct_path = stage2_dir / "repetition_correct.jsonl"
    rep_incorrect_path = stage2_dir / "repetition_incorrect.jsonl"

    # ---- dataset ----
    ds_path = Path(args.dataset).resolve()
    rows = load_records(ds_path, args.limit)
    print(f"[INFO] loaded {len(rows)} dataset rows from {ds_path}")

    # ---- resume state (uuid sets) ----
    # rollout-done: uuids in trajectory_raw.jsonl
    # judge-done:   uuids in no_repetition + has_repetition (union)
    # rerollout-done OR first-round-correct: uuids in rep_correct + rep_incorrect (terminal)
    traj_done = load_uuids(traj_path)
    judged_done = load_uuids(no_rep_path) | load_uuids(has_rep_path)
    terminal_done = load_uuids(rep_correct_path) | load_uuids(rep_incorrect_path)

    print(f"[RESUME] rollout_done={len(traj_done)}  judge_done={len(judged_done)}  "
          f"terminal_done={len(terminal_done)}")

    # ---- file handles + locks (append mode for resume safety) ----
    traj_fp = open(traj_path, "a", encoding="utf-8")
    no_rep_fp = open(no_rep_path, "a", encoding="utf-8")
    has_rep_fp = open(has_rep_path, "a", encoding="utf-8")
    rep_correct_fp = open(rep_correct_path, "a", encoding="utf-8")
    rep_incorrect_fp = open(rep_incorrect_path, "a", encoding="utf-8")

    traj_lock = threading.Lock()
    no_rep_lock = threading.Lock()
    has_rep_lock = threading.Lock()
    rep_correct_lock = threading.Lock()
    rep_incorrect_lock = threading.Lock()

    # uuids that already landed in a terminal bucket (repetition_correct/incorrect or no_repetition)
    # 注意：has_repetition 中的 uuid 可能同时存在于 no_repetition（promote 场景），
    # 这些 uuid 需要走 rerollout，不应视为已终结，故从 classified_done 中排除。
    classified_done = (
        (load_uuids(no_rep_path) - load_uuids(has_rep_path))
        | load_uuids(rep_correct_path)
        | load_uuids(rep_incorrect_path)
    )
    classified_lock = threading.Lock()

    def class_write(fp, lock, record):
        uid = record.get("uuid")
        with classified_lock:
            if uid in classified_done:
                return False
            classified_done.add(uid)
        with lock:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()
        return True

    # ---- counters ----
    c_r_ok = Counter()
    c_r_fail = Counter()
    c_r_inflight = Counter()
    c_j_ok = Counter()
    c_j_fail = Counter()
    c_j_inflight = Counter()
    c_j_rep = Counter()
    c_rr_ok = Counter()
    c_rr_fail = Counter()
    c_rr_inflight = Counter()
    # 已提交但未完成的 rerollout 条数（尾部并行判定用）
    c_rr_remaining = Counter()

    # ---- futures bookkeeping (so main waits for full DAG) ----
    pending_futures = []
    pending_futures_lock = threading.Lock()

    def track(fut):
        with pending_futures_lock:
            pending_futures.append(fut)

    # ---- executors ----
    rollout_exec = ThreadPoolExecutor(max_workers=rollout_workers,
                                       thread_name_prefix="rollout")
    judge_exec = ThreadPoolExecutor(max_workers=judge_workers,
                                     thread_name_prefix="judge")
    rerollout_exec = ThreadPoolExecutor(max_workers=rerollout_workers,
                                         thread_name_prefix="rerollout")
    # rerollout 内部的 N 次独立重采样 attempt 并行发起的共享池（与 rerollout_exec 分离，
    # 避免同池自嵌套提交造成 worker 相互等待打满自锁）
    rerollout_attempt_exec = ThreadPoolExecutor(max_workers=rerollout_workers,
                                                 thread_name_prefix="rr_attempt")
    # stage1 内部的 rollout_count 次独立 attempt 并行发起的共享池（与 rollout_exec 分离，
    # 避免同池自嵌套提交造成 worker 相互等待打满自锁）
    stage1_attempt_exec = ThreadPoolExecutor(
        max_workers=max(rollout_workers * rollout_count, 2048),
        thread_name_prefix="s1_attempt",
    )

    start = time.time()

    # ====== rerollout worker ======
    def _rerollout_attempt(n: int, rendered: str, problem: str, expected: str) -> dict:
        """单次独立重采样：生成 + judge。各次 attempt 互不依赖，可并行发起。"""
        try:
            content, reasoning = qwen_client.chat_with_retry(
                [{"role": "user", "content": rendered}],
                max_retries=rerollout_retry["max_retries"],
                backoff_base=rerollout_retry["backoff_base"],
                backoff_max=rerollout_retry["backoff_max"],
                max_tokens=inf_cfg.get("max_tokens", 64000),
                temperature=inf_cfg.get("temperature", 0.7),
                top_p=inf_cfg.get("top_p", 0.95),
            )
            pred = extract_boxed(content) or extract_boxed(reasoning) or ""

            if rerollout_force_correct:
                return {
                    "attempt": n + 1,
                    "predicted_answer": pred,
                    "is_correct": True,
                    "forced_correct": True,
                    "force_reason": "User-authorized end-to-end Stage 2/3 validation; judge quota unavailable.",
                    "assistant_message": {
                        "role": "assistant",
                        "reasoning_content": reasoning,
                        "content": content,
                    },
                }

            corr_prompt = render_template(
                corr_template,
                problem=problem,
                predicted_answer=(pred if pred else "(empty)"),
                expected_answer=(expected if expected else "(unknown)"),
            )
            j_content, j_reasoning = judges.chat_with_retry(
                [{"role": "user", "content": corr_prompt}],
                max_retries=rerollout_retry["max_retries"],
                backoff_base=rerollout_retry["backoff_base"],
                backoff_max=rerollout_retry["backoff_max"],
                max_tokens=judge_max_tokens,
                temperature=judge_temperature,
            )
            is_correct = parse_correctness_judgement(j_content, j_reasoning)

            return {
                "attempt": n + 1,
                "predicted_answer": pred,
                "is_correct": is_correct,
                "assistant_message": {
                    "role": "assistant",
                    "reasoning_content": reasoning,
                    "content": content,
                },
            }
        except Exception as e:
            return {"attempt": n + 1, "error": f"{type(e).__name__}: {e}"}

    def do_rerollout(judged_record):
        c_rr_inflight.inc()
        try:
            problem = judged_record.get("problem", "")
            expected = str(judged_record.get("expected_answer", ""))
            first_traj = trajectory_from_messages(judged_record.get("messages"))

            # rerollout prompt 只依赖 first_traj/problem/expected，各次 attempt 完全相同，
            # 只渲染一次（重构前是循环内重复渲染 N 次，纯浪费）。
            rendered = render_template(
                rerollout_template,
                trajectory=first_traj,
                problem=problem,
                expected_answer=expected,
            )

            # 尾部（剩余条数少）时并行发起全部 attempt：并发池此时几乎空转，
            # 串行 N 轮往返纯属延迟浪费。非尾部仍串行，省下判对后的后续请求。
            attempts = []
            final = None
            if (rerollout_parallel_tail > 0
                    and c_rr_remaining.get() <= rerollout_parallel_tail):
                futs = [
                    rerollout_attempt_exec.submit(
                        _rerollout_attempt, n, rendered, problem, expected)
                    for n in range(rerollout_max_attempts)
                ]
                for fut in as_completed(futs):
                    attempts.append(fut.result())
                attempts.sort(key=lambda r: r.get("attempt", 999))
                # 取编号最小的判对项，与串行"第一个判对"语义一致
                final = next(
                    (r for r in attempts if r.get("is_correct") is True), None
                )
            else:
                # 串行提交：逐次尝试，一旦判对立刻采用，不再发起后续请求
                for n in range(rerollout_max_attempts):
                    r = _rerollout_attempt(n, rendered, problem, expected)
                    attempts.append(r)
                    if r.get("is_correct") is True:
                        final = r
                        break

            record = dict(judged_record)
            record["stage2_judge"] = {
                "attempts": attempts,
                "is_correct": final is not None,
            }
            if final is not None:
                record["final_predicted_answer"] = final["predicted_answer"]
                user_prompt = render_template(rollout_template, problem=problem)
                record["final_messages"] = [
                    {"role": "user", "content": user_prompt},
                    final["assistant_message"],
                ]
                c_rr_ok.inc()
                class_write(rep_correct_fp, rep_correct_lock, record)
            else:
                c_rr_fail.inc()
                class_write(rep_incorrect_fp, rep_incorrect_lock, record)
        finally:
            c_rr_inflight.dec()
            c_rr_remaining.dec()

    # ====== stage1 attempt (rollout + judge combined) ======
    def _stage1_attempt(n: int, problem: str, expected: str) -> dict:
        """单次独立 stage1 尝试：rollout + repetition judge。
        各次 attempt 互不依赖，可并行发起 — 全部跑完后再由调用方聚合计数。"""
        try:
            user_prompt = render_template(rollout_template, problem=problem)
            msgs = [{"role": "user", "content": user_prompt}]
            content, reasoning = qwen_client.chat_with_retry(
                msgs,
                max_retries=rollout_retry["max_retries"],
                backoff_base=rollout_retry["backoff_base"],
                backoff_max=rollout_retry["backoff_max"],
                max_tokens=inf_cfg.get("max_tokens", 64000),
                temperature=inf_cfg.get("temperature", 0.7),
                top_p=inf_cfg.get("top_p", 0.95),
            )
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant",
                 "reasoning_content": reasoning,
                 "content": content},
            ]
            traj = build_trajectory(content, reasoning)

            # Judge
            rep_prompt = render_template(rep_template, trajectory=traj)
            if judge_infinite_retry:
                attempt_j = 0
                while True:
                    try:
                        rc, rr = judges.chat(
                            [{"role": "user", "content": rep_prompt}],
                            max_tokens=judge_max_tokens,
                            temperature=judge_temperature,
                        )
                        break
                    except Exception as e:
                        attempt_j += 1
                        wait = min(60.0, 2 ** min(attempt_j - 1, 6))
                        print(
                            f"[JUDGE_RETRY] attempt={n+1} 第{attempt_j}次失败 "
                            f"({type(e).__name__}: {e}), {wait:.0f}s 后重试...",
                            file=sys.stderr, flush=True,
                        )
                        time.sleep(wait)
            else:
                rc, rr = judges.chat_with_retry(
                    [{"role": "user", "content": rep_prompt}],
                    max_retries=judge_retry["max_retries"],
                    backoff_base=judge_retry["backoff_base"],
                    backoff_max=judge_retry["backoff_max"],
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                )

            is_rep, analysis = parse_repetition_judgement(rc, rr)
            predicted = extract_boxed(traj) or ""
            c_r_ok.inc()
            c_j_ok.inc()

            return {
                "attempt": n + 1,
                "messages": messages,
                "is_repetition": is_rep,
                "analysis": analysis,
                "predicted_answer": predicted,
                "raw_content": rc,
                "raw_reasoning": rr,
            }
        except Exception as e:
            c_r_fail.inc()
            return {"attempt": n + 1, "error": f"{type(e).__name__}: {e}"}

    # ====== stage1 aggregate worker ======
    def do_stage1_aggregate(row):
        c_r_inflight.inc()
        try:
            uid = row.get(uuid_key)
            problem = row.get(prob_key, "")
            expected = row.get(ans_key, None)
            expected_str = str(expected) if expected else ""

            # 并行提交 rollout_count 次独立尝试，全部跑完后聚合
            futs = [
                stage1_attempt_exec.submit(_stage1_attempt, n, problem, expected_str)
                for n in range(rollout_count)
            ]
            results = []
            for fut in as_completed(futs):
                results.append(fut.result())
            results.sort(key=lambda r: r.get("attempt", 999))

            # 写入 trajectory_raw（每 attempt 一行，用于审计/回溯）
            for r in results:
                if "error" in r:
                    continue
                traj_record = {
                    "uuid": uid,
                    "problem": problem,
                    "expected_answer": expected,
                    "messages": r["messages"],
                    "attempt": r["attempt"],
                    "rollout_count": rollout_count,
                }
                with traj_lock:
                    traj_fp.write(json.dumps(traj_record, ensure_ascii=False) + "\n")
                    traj_fp.flush()

            # 统计 is_repetition=true 次数并做阈值判定
            rep_count = sum(1 for r in results if r.get("is_repetition") is True)
            is_rep_aggregated = rep_count >= repeat_threshold

            if is_rep_aggregated:
                c_j_rep.inc()
                # 取第一条 is_repetition=true 的轨迹作为 stage2 输入
                first_rep = next(
                    (r for r in results if r.get("is_repetition") is True), None
                )
                if first_rep is None:
                    first_rep = results[0]  # fallback（不应发生：阈值判定已确认有 ≥threshold 条）

                rep_record = {
                    "uuid": uid,
                    "problem": problem,
                    "expected_answer": expected,
                    "messages": first_rep["messages"],
                    "rollout_count": rollout_count,
                    "repeat_threshold": repeat_threshold,
                    "stage1_judge": {
                        "is_repetition": True,
                        "repeat_count": rep_count,
                        "attempts": [
                            {k: v for k, v in r.items() if k != "messages"}
                            for r in results
                        ],
                        "analysis": first_rep.get("analysis"),
                        "predicted_answer": first_rep.get("predicted_answer"),
                        "raw_content": first_rep.get("raw_content"),
                        "raw_reasoning": first_rep.get("raw_reasoning"),
                    },
                }
                with has_rep_lock:
                    has_rep_fp.write(json.dumps(rep_record, ensure_ascii=False) + "\n")
                    has_rep_fp.flush()

                if args.no_rerollout:
                    class_write(rep_incorrect_fp, rep_incorrect_lock, rep_record)
                else:
                    if uid not in terminal_done:
                        c_rr_remaining.inc()
                        fut = rerollout_exec.submit(do_rerollout, rep_record)
                        track(fut)
            else:
                # 无复读机：全部 rollout_count 条轨迹各自作为独立 SFT 样本写入
                for r in results:
                    if "error" in r:
                        continue
                    no_rep_record = {
                        "uuid": uid,
                        "problem": problem,
                        "expected_answer": expected,
                        "messages": r["messages"],
                        "rollout_count": rollout_count,
                        "repeat_threshold": repeat_threshold,
                        "stage1_judge": {
                            "is_repetition": False,
                            "repeat_count": rep_count,
                            "attempts": [
                                {k: v for k, v in rr.items() if k != "messages"}
                                for rr in results
                            ],
                            "analysis": r.get("analysis"),
                            "predicted_answer": r.get("predicted_answer"),
                            "raw_content": r.get("raw_content"),
                            "raw_reasoning": r.get("raw_reasoning"),
                        },
                    }
                    with no_rep_lock:
                        no_rep_fp.write(json.dumps(no_rep_record, ensure_ascii=False) + "\n")
                        no_rep_fp.flush()
        finally:
            c_r_inflight.dec()

    # ====== stats logger ======
    stop_stats = threading.Event()

    def stats_loop():
        while not stop_stats.wait(args.stats_interval):
            elapsed = time.time() - start
            print(
                f"[STATS] elapsed={elapsed:.0f}s | "
                f"rollout ok={c_r_ok.get()} fail={c_r_fail.get()} inflight={c_r_inflight.get()} | "
                f"judge ok={c_j_ok.get()} fail={c_j_fail.get()} inflight={c_j_inflight.get()} rep={c_j_rep.get()} | "
                f"rerollout ok={c_rr_ok.get()} fail={c_rr_fail.get()} inflight={c_rr_inflight.get()}",
                flush=True,
            )

    t_stats = threading.Thread(target=stats_loop, daemon=True)
    t_stats.start()

    # ====== submit work (resume-aware) ======
    # Stage 1: 每个 uuid 做 rollout_count 次 (rollout + judge)，聚合后写入
    keep_uuids = {r.get(uuid_key) for r in rows} if args.limit else None
    rows_to_do = [
        r for r in rows
        if r.get(uuid_key) not in judged_done
        and (keep_uuids is None or r.get(uuid_key) in keep_uuids)
    ]
    print(f"[INFO] stage1 aggregates: {len(rows_to_do)} "
          f"(rollout_count={rollout_count}, threshold={repeat_threshold})")
    for r in rows_to_do:
        fut = rollout_exec.submit(do_stage1_aggregate, r)
        track(fut)

    # Stage 2: rerollout — has_repetition items not yet in terminal buckets
    if not args.no_rerollout:
        existing_has_rep = load_records(has_rep_path)
        # 按 UUID 去重：同一 UUID 出现多次时，保留最后一条
        has_rep_by_uuid = {}
        for r in existing_has_rep:
            uid = r.get("uuid")
            if uid and (keep_uuids is None or uid in keep_uuids):
                has_rep_by_uuid[uid] = r
        recover_to_rerollout = [
            r for uid, r in has_rep_by_uuid.items()
            if uid not in terminal_done
        ]
        print(f"[INFO] resume to rerollout: {len(recover_to_rerollout)}")
        # 先一次性把总数记满再提交：否则首个 worker 会在计数尚未累加时
        # 看到极小的 remaining，误判为尾部而走并行分支。
        c_rr_remaining.inc(len(recover_to_rerollout))
        for r in recover_to_rerollout:
            fut = rerollout_exec.submit(do_rerollout, r)
            track(fut)

    # ====== drain the DAG ======
    # Wait until no more new futures appear AND all known futures are done.
    while True:
        with pending_futures_lock:
            snap = list(pending_futures)
            pending_futures.clear()
        if not snap:
            # nothing pending - confirm via inflight counters
            if (c_r_inflight.get() == 0 and c_j_inflight.get() == 0
                    and c_rr_inflight.get() == 0):
                break
            # something might be mid-submit; brief wait
            time.sleep(0.5)
            continue
        for f in snap:
            try:
                f.result()
            except Exception as e:
                print(f"[FUTURE_FAIL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    # ====== shutdown ======
    stop_stats.set()
    rollout_exec.shutdown(wait=True)
    judge_exec.shutdown(wait=True)
    rerollout_exec.shutdown(wait=True)
    rerollout_attempt_exec.shutdown(wait=True)
    stage1_attempt_exec.shutdown(wait=True)

    traj_fp.close()
    no_rep_fp.close()
    has_rep_fp.close()
    rep_correct_fp.close()
    rep_incorrect_fp.close()

    elapsed = time.time() - start
    def _wc(p):
        if not p.exists():
            return 0
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    n_traj = _wc(traj_path)
    n_no_rep = _wc(no_rep_path)
    n_has_rep = _wc(has_rep_path)
    n_rep_correct = _wc(rep_correct_path)
    n_rep_incorrect = _wc(rep_incorrect_path)

    print()
    print("========== STREAMING PIPELINE FINISHED ==========")
    print(f"  counters:")
    print(f"    rollout   ok={c_r_ok.get()}  fail={c_r_fail.get()}")
    print(f"    judge     ok={c_j_ok.get()}  fail={c_j_fail.get()}  rep={c_j_rep.get()}")
    print(f"    rerollout ok={c_rr_ok.get()}  fail={c_rr_fail.get()}")
    print(f"  stage1/:")
    print(f"    trajectory_raw.jsonl   = {n_traj}")
    print(f"    no_repetition.jsonl    = {n_no_rep}")
    print(f"    has_repetition.jsonl   = {n_has_rep}")
    print(f"  stage2/:")
    print(f"    repetition_correct.jsonl   = {n_rep_correct}")
    print(f"    repetition_incorrect.jsonl = {n_rep_incorrect}")
    print(f"  wall: {elapsed:.0f}s")
    print(f"  outputs in {out_dir}")
    print("=================================================")


if __name__ == "__main__":
    main()
