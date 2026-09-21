#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn3 Reform: 将 stage2 已通过 rerollout 并判对的数据用 reform 模板重新生成 SFT 训练样例。

实现逻辑：
  1. 读取 stage2 repetition_correct_sample.jsonl
  2. 将 current final_messages 序列化为 chat_history 文本，填充 reform 模板作为 user prompt
  3. 调用 Qwen apex 端点推理 → 获得 reasoning_content + content
  4. final_messages 转为 turn2_messages，LLM 的输出作为新的 final_messages
  5. 用 Kimi 比对 expected_answer vs 新 final_messages 中的 content
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import httpx
import yaml
from openai import OpenAI


# ============================================================
# 统计（线程安全）
# ============================================================
_wall_lock = threading.Lock()
_wall_count = 0
_wall_total = 0.0

_latency_locks: Dict[str, threading.Lock] = {}
_latency_counters: Dict[str, int] = {}
_latency_totals: Dict[str, float] = {}


def _record_latency(phase: str, sec: float):
    global _latency_locks, _latency_counters, _latency_totals
    lock = _latency_locks.setdefault(phase, threading.Lock())
    with lock:
        _latency_counters[phase] = _latency_counters.get(phase, 0) + 1
        _latency_totals[phase] = _latency_totals.get(phase, 0.0) + sec


def _latency_report() -> str:
    lines = []
    for phase in sorted(_latency_counters.keys()):
        n = _latency_counters.get(phase, 0)
        t = _latency_totals.get(phase, 0.0)
        avg = t / n if n > 0 else 0.0
        lines.append(f"{phase}: n={n}, total={t:.1f}s, avg={avg:.1f}s")
    return "  |  ".join(lines)

from share_utils import (
    MultiJudgeClient,
    load_yaml_field,
    parse_correctness_judgement,
    render_template,
)


class ChatClient:
    def __init__(self, base_urls: List[str], model: str, api_key: str = "dummy",
                 max_concurrency: Optional[int] = None):
        if not base_urls:
            raise ValueError("base_urls cannot be empty")
        self.base_urls = [u.rstrip("/") for u in base_urls]
        self.model = model
        self.api_key = api_key
        self._rr_idx = 0
        self._rr_lock = threading.Lock()
        self._sema = threading.Semaphore(max_concurrency) if (max_concurrency and max_concurrency > 0) else None
        h = httpx.Client(limits=httpx.Limits(max_keepalive_connections=256, max_connections=25000),
                          timeout=httpx.Timeout(600.0, connect=10.0))
        self._clients = [OpenAI(base_url=u, api_key=api_key, http_client=h) for u in self.base_urls]

    def chat(self, messages, max_tokens=64000, temperature=0.7, top_p=0.95, timeout=900):
        with self._rr_lock:
            idx = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % len(self._clients)
        client = self._clients[idx]
        if self._sema is not None:
            self._sema.acquire()
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, timeout=timeout)
        finally:
            if self._sema is not None:
                self._sema.release()
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "") or ""
        return content, reasoning


# ============================================================
# 把 final_messages 转化为 chat_history 文本
# ============================================================
def final_messages_to_chat_history(final_messages: list) -> str:
    lines = []
    for msg in final_messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
            if reasoning:
                lines.append(f"Assistant reasoning: {reasoning}")
    return "\n".join(lines)


# ============================================================
# 解析 LLM 输出的 XML → (reasoning_content, content)
# 期望格式:
#   <reasoning_content> ... </reasoning_content>
#   <content> ... </content>
# 返回 (reasoning, content) 或 (None, None) 表示解析失败
# ============================================================
_XML_REASONING_RE = re.compile(r"<reasoning_content>\s*(.*?)\s*</reasoning_content>", re.DOTALL | re.IGNORECASE)
_XML_CONTENT_RE = re.compile(r"<content>\s*(.*?)\s*</content>", re.DOTALL | re.IGNORECASE)


def parse_reform_xml(content_text: str):
    if not content_text:
        return None, None
    m_r = _XML_REASONING_RE.search(content_text)
    m_c = _XML_CONTENT_RE.search(content_text)
    if not m_r or not m_c:
        return None, None
    reasoning = m_r.group(1).strip()
    content = m_c.group(1).strip()
    if not content:
        return None, None
    return reasoning, content


# ============================================================
# 解析 LLM 输出的 JSON → (reasoning_content, content)  [legacy, kept for fallback]
# ============================================================
def parse_reform_json(content_text: str) -> tuple:
    if not content_text:
        return "", ""
    text = content_text.strip()
    # 尝试直接 JSON 解析
    try:
        obj = json.loads(text)
        return obj.get("reasoning_content", ""), obj.get("content", "")
    except Exception:
        pass
    # 尝试从 markdown 包裹中提取
    for pat in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*?\})"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                return obj.get("reasoning_content", ""), obj.get("content", "")
            except Exception:
                continue
    # fallback: 整个 content_text 作为 content
    return "", content_text


# ============================================================
# Kimi Judge
# ============================================================
def judge_correctness(judge_client, judge_template: str, problem: str,
                      expected_answer: str, predicted_answer: str,
                      judge_max_tokens: int, judge_temperature: float) -> Optional[bool]:
    judge_prompt = render_template(
        judge_template,
        problem=problem,
        predicted_answer=(predicted_answer if predicted_answer else "(empty)"),
        expected_answer=(expected_answer if expected_answer else "(unknown)"),
    )
    # 本地 ChatClient 只有 chat(); MultiJudgeClient.chat 每次调用轮转到下一个后端,
    # 因此这里的重试天然分散到不同 judge 后端
    last_err = None
    for attempt in range(5):
        try:
            content, reasoning = judge_client.chat(
                [{"role": "user", "content": judge_prompt}],
                max_tokens=judge_max_tokens,
                temperature=judge_temperature,
            )
            verdict = parse_correctness_judgement(content, reasoning)
            if verdict is not None:
                return verdict
            last_err = ValueError("unparseable judge response")
        except Exception as e:
            last_err = e
        time.sleep(min(30.0, 2.0 ** attempt))
    # 调用失败 ≠ 判错: 返回 None, 上层不计入重试次数并补发 (避免 judge 服务故障
    # 期间把样本静默写成 incorrect 假阴性)
    print(f"[WARN] judge call failed after retries (not counted as attempt): {last_err}",
          file=sys.stderr, flush=True)
    return None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Turn3 Reform + correctness judge")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rollout-api-config",
                        help="Separate rollout API YAML; falls back to --config")
    parser.add_argument("--judge-api-config",
                        help="Separate judge API YAML; falls back to --config")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=0,
                       help="Process only N records; 0 means all")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip records already in output")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Thread pool size (overrides config.turn3_reform.max_workers; default 64)")
    parser.add_argument("--retry-incorrect", action="store_true",
                       help="Re-process samples already in turn3_reform_incorrect.jsonl")
    parser.add_argument("--parallel-tail-threshold", type=int, default=None,
                       help="剩余待处理条数 ≤ 该值时, 单条记录的 attempt 改为并行发起 "
                            "(overrides config.turn3_reform.parallel_tail_threshold; 0=关闭)")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _load_config(path):
        with open(Path(path).resolve(), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    rollout_api_cfg = _load_config(args.rollout_api_config) if args.rollout_api_config else cfg
    judge_api_cfg = _load_config(args.judge_api_config) if args.judge_api_config else cfg
    print(f"[INFO] rollout API config = {Path(args.rollout_api_config).resolve() if args.rollout_api_config else cfg_path}"
          f"{' (legacy fallback)' if not args.rollout_api_config else ''}")
    print(f"[INFO] judge API config = {Path(args.judge_api_config).resolve() if args.judge_api_config else cfg_path}"
          f"{' (legacy fallback)' if not args.judge_api_config else ''}")

    cfg_dir = cfg_path.parent
    project_root = cfg_dir.parent.parent if cfg_dir.name == "runtime" else cfg_dir.parent

    def _resolve_prompt(rel):
        path = Path(rel).expanduser()
        candidates = [path] if path.is_absolute() else [
            project_root / path,
            cfg_dir / path,
            cfg_dir / path.name,
            Path.cwd() / path,
        ]
        prompt_path = next((p for p in candidates if p.exists()), None)
        if prompt_path is None:
            raise FileNotFoundError(f"prompt template not found: tried {candidates}")
        return prompt_path.resolve()

    # Reform prompt 模板（含 {promblem} 和 {chat_history}） — 路径从 config 读取
    turn3_cfg = cfg.get("turn3_reform", {}) or {}
    prompt_rel = turn3_cfg.get("prompt", "config/prompts/turn3_reform_v3.yaml")
    prompt_path = _resolve_prompt(prompt_rel)
    print(f"[INFO] turn3_reform prompt = {prompt_path}")
    reform_prompt = load_yaml_field(prompt_path, "reform")

    prompts_cfg = cfg.get("prompts", {})
    correctness_path = _resolve_prompt(prompts_cfg["correctness"])
    correctness_template = load_yaml_field(correctness_path, "judge")
    print(f"[INFO] correctness prompt = {correctness_path}")

    # 解析失败时的重试次数 (call Qwen 重新生成)
    parse_max_retries = int(turn3_cfg.get("parse_max_retries", 3))
    # 判错重试直到判对的最大尝试次数 (默认 1 = 旧行为)
    max_reform_attempts = int(turn3_cfg.get("max_reform_attempts", 1))
    reform_force_correct = bool(turn3_cfg.get("force_correct", False))
    # 剩余待处理条数 ≤ 该阈值时，attempt 由串行改为并行发起（0 = 关闭）。
    # 尾部并发池几乎空转，串行 N 轮往返是纯延迟浪费；代价是判对也会跑满 N 次。
    if args.parallel_tail_threshold is not None:
        parallel_tail_threshold = int(args.parallel_tail_threshold)
    else:
        parallel_tail_threshold = int(turn3_cfg.get("parallel_tail_threshold", 0))

    # Qwen client
    if "qwen_apex" not in rollout_api_cfg:
        raise SystemExit("Rollout API config missing qwen_apex")
    qwen_cfg = rollout_api_cfg["qwen_apex"]
    qwen_client = ChatClient(
        base_urls=qwen_cfg["base_urls"],
        model=qwen_cfg.get("model", "default"),
        api_key=qwen_cfg.get("api_key", "dummy"),
        max_concurrency=qwen_cfg.get("max_concurrency", 100))

    # Judge clients (round-robin across all configured judges)
    judge_backends = []
    for section in ["kimi", "glm", "deepseek_397b"]:
        jc = judge_api_cfg.get(section)
        if not jc:
            continue
        urls = jc.get("base_urls") or ([jc["base_url"]] if "base_url" in jc else [])
        judge_backends.append((
            ChatClient(
                base_urls=urls,
                model=jc.get("model", "default"),
                api_key=jc.get("api_key", "dummy"),
                max_concurrency=jc.get("max_concurrency", 100),
            ),
            jc,
        ))
    if not judge_backends:
        raise SystemExit("No judge backend configured (kimi/glm/deepseek_397b)")
    judges = MultiJudgeClient(judge_backends)

    # Unified judge params (use minimum for safety)
    judge_max_tokens = min(
        (jc.get("judge_max_tokens", 65536) for _, jc in judge_backends),
        default=65536,
    )
    judge_temperature = judge_backends[0][1].get("judge_temperature", 0.1)
    print(f"[INFO] turn3 judge backends: {len(judge_backends)} max_tokens={judge_max_tokens}")

    inf_cfg = cfg.get("inference", {})
    max_tokens = inf_cfg.get("max_tokens", 64000)
    temperature = inf_cfg.get("temperature", 0.7)
    top_p = inf_cfg.get("top_p", 0.95)

    # 读取输入数据
    input_path = Path(args.input_file).resolve()
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # 必须包含 final_messages 字段
            if "final_messages" not in rec:
                continue
            records.append(rec)

    print(f"Loaded {len(records)} records with final_messages from {input_path}")

    if args.sample_size > 0:
        records = records[: args.sample_size]

    # 输出去重 + 双文件
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_correct_path = output_dir / "turn3_reform_correct.jsonl"
    output_incorrect_path = output_dir / "turn3_reform_incorrect.jsonl"
    old_output_path = output_dir / "turn3_reform.jsonl"

    # 一次性迁移: 旧 turn3_reform.jsonl → _correct.jsonl + _incorrect.jsonl
    if old_output_path.exists() and not output_correct_path.exists() and not output_incorrect_path.exists():
        print("[INFO] 检测到旧版 turn3_reform.jsonl，自动迁移到 _correct.jsonl / _incorrect.jsonl ...")
        migrated_correct = 0
        migrated_incorrect = 0
        with open(old_output_path, "r", encoding="utf-8") as f_old:
            with open(output_correct_path, "w", encoding="utf-8") as f_c, \
                 open(output_incorrect_path, "w", encoding="utf-8") as f_i:
                for line in f_old:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("turn3_is_correct") is True:
                            f_c.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            migrated_correct += 1
                        else:
                            f_i.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            migrated_incorrect += 1
                    except Exception:
                        pass
        print(f"[INFO] 迁移完成: correct={migrated_correct}, incorrect={migrated_incorrect}")

    # 构建 done_uuids
    done_uuids = set()
    if args.skip_existing:
        # 正确样本始终跳过 (已成功, 无需重跑)
        files_to_skip = [output_correct_path]
        # 错误样本仅在未指定 --retry-incorrect 时跳过
        if not args.retry_incorrect:
            files_to_skip.append(output_incorrect_path)
        for fp in files_to_skip:
            if fp.exists():
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            done_uuids.add(json.loads(line)["uuid"])
                        except Exception:
                            pass
        if args.retry_incorrect and output_incorrect_path.exists():
            # 统计有多少 incorrect 样本将被重跑
            inc_count = sum(1 for _ in open(output_incorrect_path, "r", encoding="utf-8"))
            print(f"[INFO] --retry-incorrect: 将重跑 {inc_count} 条已有错误样本")
        print(f"Skipping {len(done_uuids)} already-processed UUIDs")

    output_lock = threading.Lock()
    output_correct_fp = open(output_correct_path, "a", encoding="utf-8")
    output_incorrect_fp = open(output_incorrect_path, "a", encoding="utf-8")

    # 进度
    progress_lock = threading.Lock()
    progress = {"done": 0, "total": len(records), "correct": 0, "incorrect": 0, "dropped": 0}
    start_time = time.time()

    # 待处理队列：先一次性算好总数, 再逐条递减。
    # 必须在提交前把 remaining 记满, 否则首个 worker 会在计数尚未累加时
    # 看到极小的 remaining, 误判为尾部而走并行分支。
    todo_records = [rec for rec in records if rec["uuid"] not in done_uuids]
    remaining_lock = threading.Lock()
    remaining = {"n": len(todo_records)}

    # 记录级线程池大小
    if args.max_workers is not None:
        MAX_WORKERS = int(args.max_workers)
    else:
        MAX_WORKERS = int((cfg.get("turn3_reform") or {}).get("max_workers", 64))
    print(f"[INFO] turn3_reform workers = {MAX_WORKERS}  max_reform_attempts={max_reform_attempts}")
    print(f"[INFO] turn3_reform: todo={len(todo_records)} "
          f"parallel_tail_threshold={parallel_tail_threshold}")

    # 每条记录内部 max_reform_attempts 次独立 reform+judge attempt 并行发起的共享池
    # （与记录级线程池 executor 分离，避免同池自嵌套提交造成 worker 相互等待打满自锁，
    # 与 streaming_pipeline.py 的 rerollout_attempt_exec/stage1_attempt_exec 同理）。
    #
    # 池大小必须受 qwen_apex.max_concurrency 约束: ChatClient 内部有一个全局信号量
    # (_sema, 见 ChatClient.__init__), 所有 attempt 线程共抢这 max_concurrency 个名额。
    # 若 MAX_WORKERS * max_reform_attempts 远超该值, 拿不到名额的线程会全部阻塞在
    # _sema.acquire() 上; 而记录级线程又在 as_completed() 同步等 attempt 结果 ——
    # 两层池经由这个共享信号量互相堵死, 表现为大量线程 futex_wait_queue、CPU 零增长。
    # (2026-07-29 实测: MAX_WORKERS=2000 → 池 16000 vs 信号量 3192, 2097 线程卡死。)
    _apex_conc = int(qwen_cfg.get("max_concurrency", 0) or 0)
    _attempt_pool = max(MAX_WORKERS * max_reform_attempts, MAX_WORKERS)
    if _apex_conc > 0 and _attempt_pool > _apex_conc:
        print(f"[WARN] attempt 池 {_attempt_pool} 超过 qwen_apex.max_concurrency={_apex_conc}, "
              f"下调为 {_apex_conc} 以避免信号量争用死锁")
        _attempt_pool = _apex_conc
    reform_attempt_exec = ThreadPoolExecutor(
        max_workers=_attempt_pool,
        thread_name_prefix="reform_attempt",
    )

    def _reform_attempt(n: int, uid: str, problem: str, expected_answer: str, prompt: str) -> dict:
        """单次独立 reform 尝试：生成 + judge。各次 attempt 互不依赖，可并行发起。"""
        # 调用 Qwen, 最多 parse_max_retries 次, 直到 XML 解析成功（与是否判对无关的内层重试）
        reform_reasoning = None
        reform_content = None
        parse_attempt = 0
        while parse_attempt < parse_max_retries:
            parse_attempt += 1
            try:
                content_text, _reasoning_text = qwen_client.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            except Exception as e:
                print(f"[ERROR][{uid}] attempt={n+1} Qwen call failed "
                      f"(parse retry {parse_attempt}/{parse_max_retries}): {e}",
                      file=sys.stderr, flush=True)
                continue

            r, c = parse_reform_xml(content_text)
            if r is not None and c is not None:
                reform_reasoning, reform_content = r, c
                break
            print(f"[WARN][{uid}] attempt={n+1} XML parse failed "
                  f"(parse retry {parse_attempt}/{parse_max_retries})",
                  file=sys.stderr, flush=True)

        if reform_reasoning is None or reform_content is None:
            return {"attempt": n + 1, "error": "parse_failed"}

        # 用 correctness judge 判对；端到端验证可由本地配置显式强制通过。
        if reform_force_correct:
            verdict = True
            forced_correct = True
        else:
            verdict = judge_correctness(
                judges, correctness_template, problem, expected_answer, reform_content,
                judge_max_tokens, judge_temperature,
            )
            forced_correct = False
        if verdict is None:
            return {"attempt": n + 1, "error": "judge_failed"}

        return {
            "attempt": n + 1,
            "reform_reasoning": reform_reasoning,
            "reform_content": reform_content,
            "is_correct": verdict,
            "forced_correct": forced_correct,
            "force_reason": (
                "User-authorized end-to-end Stage 2/3 validation; judge quota unavailable."
                if forced_correct else None
            ),
        }

    def do_one(item):
        uid = item["uuid"]
        try:
            _do_one_inner(item)
        except Exception as e:
            print(f"[CRITICAL][{uid}] Unexpected failure: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            with remaining_lock:
                remaining["n"] -= 1

    def _do_one_inner(item):
        uid = item["uuid"]
        problem = item.get("problem", "")
        expected_answer = str(item.get("expected_answer", ""))

        t0 = time.time()

        # 将当前 final_messages 转为 chat_history 文本
        chat_history = final_messages_to_chat_history(item["final_messages"])

        # 填充模板（各次 attempt 完全相同，只渲染一次）
        prompt = render_template(reform_prompt, promblem=problem, chat_history=chat_history)

        # 尾部（剩余条数少）时并行发起全部 attempt：并发池此时几乎空转，
        # 串行 N 轮往返纯属延迟浪费。非尾部仍串行，省下判对后的后续请求。
        # judge 调用失败 (error=judge_failed) 的 attempt 不计入 max_reform_attempts 额度，
        # 自动补发；补发上限 max_submit 防止 judge 长时间故障时无限重试。
        attempts = []
        final = None
        valid = 0
        max_submit = max_reform_attempts * 5
        with remaining_lock:
            is_tail = 0 < parallel_tail_threshold and remaining["n"] <= parallel_tail_threshold

        if is_tail and max_reform_attempts > 1:
            # 并行波次提交：一次发起所需的全部 attempt, 若其中有 judge 调用失败
            # (error=judge_failed, 不计入额度) 则再补发一波, 直到凑满 max_reform_attempts
            # 个有效 attempt 或触及 max_submit 上限 —— 与串行分支的补发语义一致。
            submitted = 0
            while valid < max_reform_attempts and submitted < max_submit:
                need = min(max_reform_attempts - valid, max_submit - submitted)
                futs = [
                    reform_attempt_exec.submit(
                        _reform_attempt, submitted + i, uid, problem, expected_answer, prompt)
                    for i in range(need)
                ]
                submitted += need
                for fut in as_completed(futs):
                    r = fut.result()
                    attempts.append(r)
                    if r.get("error") != "judge_failed":
                        valid += 1
                # 已有判对项即可停止, 不再补发后续波次
                if any(r.get("is_correct") is True for r in attempts):
                    break
            attempts.sort(key=lambda r: r.get("attempt", 999))
            # 取编号最小的判对项，与串行"第一个判对"语义一致
            final = next(
                (r for r in attempts if r.get("is_correct") is True), None
            )
        else:
            # 串行提交：逐次尝试，一旦判对立刻采用，不再发起后续请求
            submitted = 0
            while valid < max_reform_attempts and submitted < max_submit:
                r = _reform_attempt(submitted, uid, problem, expected_answer, prompt)
                submitted += 1
                attempts.append(r)
                if r.get("error") == "judge_failed":
                    continue  # 不计入重试次数
                valid += 1
                if r.get("is_correct") is True:
                    final = r
                    break

        # 全部判错时，取最后一个成功解析(无 error)的 attempt 落盘为 incorrect；
        # 若所有 attempt 连 XML 都没解析成功，丢弃该样本（与旧行为一致）
        if final is None:
            parsed_attempts = [a for a in attempts if "error" not in a]
            final = parsed_attempts[-1] if parsed_attempts else None

        if final is None:
            with progress_lock:
                progress["done"] += 1
                progress["dropped"] = progress.get("dropped", 0) + 1
            print(f"[DROP][{uid}] all {len(attempts)} attempts failed (parse/judge), discarding",
                  file=sys.stderr, flush=True)
            return

        is_correct = bool(final.get("is_correct"))

        # 构造输出记录
        result = dict(item)
        # 原来的 final_messages 转为 turn2_messages
        result["turn2_messages"] = item["final_messages"]
        # LLM 输出作为新 final_messages（格式: user+assistant）
        result["final_messages"] = [
            {"role": "user", "content": problem},
            {"role": "assistant",
             "reasoning_content": final["reform_reasoning"],
             "content": final["reform_content"]},
        ]
        result["stage3_judge"] = {
            "is_correct": is_correct,
            "attempts": attempts,
        }

        # 根据正确性写入不同文件
        with output_lock:
            out_fp = output_correct_fp if is_correct else output_incorrect_fp
            out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_fp.flush()

        with progress_lock:
            progress["done"] += 1
            if is_correct:
                progress["correct"] += 1
            else:
                progress["incorrect"] += 1

        total_elapsed = time.time() - t0
        _record_latency("total", total_elapsed)

        with progress_lock:
            done = progress["done"]
            total = progress["total"]
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            correct_rate = progress["correct"] / progress["done"] * 100 if progress["done"] > 0 else 0
            print(f"[{done}/{total}] uuid={uid} correct={is_correct} attempts={valid}/{max_reform_attempts} "
                  f"correct_rate={correct_rate:.0f}% incorrect={progress.get('incorrect',0)} dropped={progress.get('dropped',0)} "
                  f"speed={speed:.2f}/s ETA={eta:.0f}s "
                  f"| {_latency_report()}")

    # 并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(do_one, rec) for rec in todo_records]
        for _ in as_completed(futures):
            pass
    reform_attempt_exec.shutdown(wait=True)

    output_correct_fp.close()
    output_incorrect_fp.close()
    print(f"Done. Correct: {progress['correct']}, Incorrect: {progress['incorrect']}, "
          f"Dropped: {progress.get('dropped', 0)}, Total: {progress['done']}.")
    print(f"Output: {output_correct_path} / {output_incorrect_path}")


if __name__ == "__main__":
    main()