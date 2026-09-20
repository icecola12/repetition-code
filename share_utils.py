#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""share_utils.py — shared utilities for rollout/judge/reaggregation pipelines.

Provides:
  - ChatClient: OpenAI-compatible client with URL list + round-robin + semaphore
  - _boxed_re / extract_boxed: extract \\boxed{...} from text
  - parse_repetition_judgement: parse Kimi JSON rep output → (is_rep, analysis)
  - parse_correctness_judgement: parse Kimi Yes/No output → bool
  - render_template: replace {key} placeholders
  - load_prompt_template: load YAML prompt files
"""

import json
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from openai import OpenAI, BadRequestError


def backoff_with_jitter(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """指数退避 + 随机抖动 (0.5x~1.5x)。

    并发场景下若所有线程使用完全确定性的退避时间表 (2s, 4s, 8s...)，
    会导致大量线程在同一时刻集中重试、加重限流。抖动让重试时间错开。
    attempt 从 1 开始计数。
    """
    wait = min(cap, base ** min(attempt - 1, 6))
    return wait * random.uniform(0.5, 1.5)


# ==========================================
# ChatClient
# ==========================================
class ChatClient:
    """OpenAI-compatible multi-endpoint round-robin client with two-tier semaphore pools.

    Supports two URL groups with independent concurrency limits:
      - base_urls:  general list (or legacy single pool)
      - base_urls_2: optional second pool (e.g. remote endpoint)
    Each group runs its own round-robin and semaphore.

    Backward-compatible: if base_url_1 is None, uses only base_urls.
    """

    def __init__(
        self,
        base_urls: List[str],
        model: str,
        api_key: str = "dummy",
        max_concurrency: Optional[int] = None,
        # -- tier-2 pool (optional) --
        base_urls_2: Optional[List[str]] = None,
        max_concurrency_2: Optional[int] = None,
    ):
        self.model = model
        self.api_key = api_key

        # -- pool 1 --
        if not base_urls:
            raise ValueError("base_urls cannot be empty")
        self._urls1 = [u.rstrip("/") for u in base_urls]
        self._idx1 = 0
        self._sema1 = (
            threading.BoundedSemaphore(max_concurrency)
            if (max_concurrency and max_concurrency > 0)
            else None
        )

        # -- pool 2 (optional) --
        self._urls2: List[str] = []
        self._idx2 = 0
        self._sema2 = None
        if base_urls_2:
            self._urls2 = [u.rstrip("/") for u in base_urls_2]
            self._sema2 = (
                threading.BoundedSemaphore(max_concurrency_2)
                if (max_concurrency_2 and max_concurrency_2 > 0)
                else None
            )

        self._rr_lock = threading.Lock()

        h = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=256, max_connections=25000),
            timeout=httpx.Timeout(600.0, connect=10.0),
        )
        # build OpenAI clients for all urls (pool1 + pool2)
        self._clients1 = [
            OpenAI(base_url=u, api_key=api_key, http_client=h) for u in self._urls1
        ]
        self._clients2 = [
            OpenAI(base_url=u, api_key=api_key, http_client=h) for u in self._urls2
        ]
        # total rounds to alternate between pools
        self._total_pools = 1 + (1 if self._clients2 else 0)
        self._idx_round = 0

    def chat(self, messages, max_tokens=64000, temperature=0.7, top_p=0.95,
             timeout=1800, presence_penalty=None, top_k=None, min_p=None,
             repetition_penalty=None, no_thinking=False):
        """top_k/min_p/repetition_penalty 是 sglang/vLLM 的扩展采样参数，OpenAI SDK
        没有原生入参，通过 extra_body 透传——sglang 的 ChatCompletionRequest 把它们
        定义为请求体顶层字段（非嵌套），extra_body 内容会被合并进请求 JSON 顶层，
        因此无需修改服务端部署即可生效。presence_penalty 是 OpenAI SDK 原生参数。"""
        with self._rr_lock:
            # alternate between pool1 and pool2 (if pool2 exists)
            pool_choice = self._idx_round % self._total_pools
            self._idx_round = (self._idx_round + 1) % (self._total_pools * 8192)
            if pool_choice == 0 or not self._clients2:
                idx = self._idx1
                self._idx1 = (self._idx1 + 1) % len(self._clients1)
                openai_client = self._clients1[idx]
                sema = self._sema1
            else:
                idx = self._idx2
                self._idx2 = (self._idx2 + 1) % len(self._clients2)
                openai_client = self._clients2[idx]
                sema = self._sema2
        if sema:
            sema.acquire()
        acquired = sema is not None
        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if min_p is not None:
            extra_body["min_p"] = min_p
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty
        if no_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        create_kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
        if presence_penalty is not None:
            create_kwargs["presence_penalty"] = presence_penalty
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        try:
            resp = openai_client.chat.completions.create(**create_kwargs)
        except BadRequestError as e:
            if acquired:
                sema.release()
                acquired = False
            msg = str(e)
            # e.g. "maximum context length of 262144 tokens. You requested ... 240483 tokens from the input ... 100000 tokens for the completion"
            m_max = re.search(r'maximum context length of (\d+)', msg)
            m_in = re.search(r'(\d+) tokens from the input', msg)
            if m_max and m_in:
                ctx_max = int(m_max.group(1))
                in_tok = int(m_in.group(1))
                safe_max = ctx_max - in_tok - 1024  # safety margin for tokenizer variance
                if safe_max > 0:
                    return self.chat(messages, max_tokens=safe_max,
                                     temperature=temperature, top_p=top_p,
                                     timeout=timeout, presence_penalty=presence_penalty,
                                     top_k=top_k, min_p=min_p,
                                     repetition_penalty=repetition_penalty,
                                     no_thinking=no_thinking)
            raise
        finally:
            if acquired:
                sema.release()
        msg = resp.choices[0].message
        return (msg.content or "", getattr(msg, "reasoning_content", "") or "")

    def chat_with_retry(self, messages, max_retries=3, backoff_base=2.0,
                        backoff_max=30.0, label="chat", **chat_kwargs):
        """Call chat() with bounded exponential backoff. Each attempt rotates
        round-robin to a new endpoint, so transient single-endpoint failures
        don't hang the request."""
        import time as _time
        last_exc = None
        for attempt in range(max_retries):
            try:
                return self.chat(messages, **chat_kwargs)
            except Exception as e:
                last_exc = e
                if attempt == max_retries - 1:
                    raise
                wait = min(backoff_max, backoff_base ** attempt)
                # Stay quiet by default; caller can wrap if it wants verbose retries
                _time.sleep(wait)
        # unreachable: last attempt re-raises
        raise last_exc  # pragma: no cover


# ==========================================
# MultiJudgeClient — round-robin across multiple judge backends
# ==========================================
class MultiJudgeClient:
    """Round-robin across multiple ChatClient judge backends.

    Each backend has its own api_key / model, but they share the same base_url
    and parameter semantics (judge_max_tokens, judge_temperature, etc.).

    Usage:
        judges = MultiJudgeClient([
            (kimi_client, kimi_cfg),
            (glm_client, glm_cfg),
            (deepseek_client, deepseek_cfg),
        ])
        content, reasoning = judges.chat(messages, max_tokens=..., temperature=...)
    """

    def __init__(self, clients_and_configs: List[Tuple[ChatClient, dict]]):
        if not clients_and_configs:
            raise ValueError("MultiJudgeClient requires at least one (client, cfg) pair")
        self._pairs = clients_and_configs
        self._lock = threading.Lock()
        self._idx = 0

    def _pick(self):
        with self._lock:
            pair = self._pairs[self._idx % len(self._pairs)]
            self._idx = (self._idx + 1) % (len(self._pairs) * 8192)
        return pair

    def chat(self, messages, max_tokens=64000, temperature=0.7, top_p=0.95,
             timeout=1800):
        """Pick next judge backend round-robin and call chat()."""
        client, cfg = self._pick()
        return client.chat(messages, max_tokens=max_tokens, temperature=temperature,
                           top_p=top_p, timeout=timeout)

    def chat_with_retry(self, messages, max_retries=3, backoff_base=2.0,
                        backoff_max=30.0, label="judge", **chat_kwargs):
        """Pick next judge backend round-robin and call chat_with_retry()."""
        client, cfg = self._pick()
        return client.chat_with_retry(messages, max_retries=max_retries,
                                      backoff_base=backoff_base,
                                      backoff_max=backoff_max,
                                      label=label, **chat_kwargs)


# ==========================================
# boxed extraction
# ==========================================
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


# ==========================================
# template rendering
# ==========================================
def render_template(template: str, **kwargs: str) -> str:
    """Replace {key} placeholders; leave {{ and }} intact."""
    sentinels = {"{{": "\x00LB\x00", "}}": "\x00RB\x00"}
    text = template.replace("{{", sentinels["{{"]).replace("}}", sentinels["}}"])
    for key, val in kwargs.items():
        text = text.replace("{" + key + "}", str(val))
    return text.replace(sentinels["{{"], "{").replace(sentinels["}}"], "}")


def load_prompt_template(yaml_path: Path) -> str:
    """Load a prompt template from a YAML file, returning the 'judge' field."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "judge" not in data:
        raise ValueError(f"{yaml_path} missing 'judge' field")
    return data["judge"]


def load_yaml_field(yaml_path: Path, key: str) -> str:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if key not in data:
        raise ValueError(f"{yaml_path} missing '{key}' field")
    return data[key]


# ==========================================
# Kimi judge output parsing
# ==========================================
_judge_line_re = re.compile(r"Judgement\s*[:：]\s*(Yes|No|yes|no|YES|NO|是|否)")


def parse_correctness_judgement(content: str, reasoning: str) -> Optional[bool]:
    """Parse Yes/No from correctness judge output. True=correct, False=incorrect, None=unparseable."""
    for text in (content, reasoning):
        if not text:
            continue
        m = _judge_line_re.search(text)
        if m:
            ans = m.group(1).lower()
            return ans in ("yes", "是")
        # fallback: check last non-empty line
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines:
            tail = lines[-1].lower()
            if tail.startswith(("yes", "true", "是")):
                return True
            if tail.startswith(("no", "false", "否")):
                return False
    return None


def parse_repetition_judgement(content: str, reasoning: str) -> Tuple[Optional[bool], Optional[str]]:
    """Parse Kimi repetition judge JSON output → (is_repetition, analysis)."""
    for text in (content, reasoning):
        if not text:
            continue
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```\s*$", "", s)
        l = s.find("{")
        r = s.rfind("}")
        if l == -1 or r == -1 or r <= l:
            continue
        cand = s[l: r + 1]
        try:
            obj = json.loads(cand)
        except Exception:
            try:
                fixed = re.sub(r",\s*([\]}])", r"\1", cand)
                obj = json.loads(fixed)
            except Exception:
                continue
        if not isinstance(obj, dict) or "repeat" not in obj:
            continue
        rep = obj["repeat"]
        if isinstance(rep, bool):
            return rep, obj.get("analysis", "")
        if isinstance(rep, str):
            v = rep.strip().lower()
            if v in ("true", "yes", "是"):
                return True, obj.get("analysis", "")
            if v in ("false", "no", "否"):
                return False, obj.get("analysis", "")
    return None, None