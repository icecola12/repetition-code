#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_to_data_all.py — 扫描所有 output/<timestamp>/ 子目录中的 judge 和 rerollout 输出，
把数据分为 4 类，重建 data_all/ 文件夹（每次从头开始）。

4 类:
  1. repetition_incorrect.jsonl   — 有复读机，且未成功修复（rerollout 未开始/在途，或已用尽仍未判对）
  2. repetition_correct.jsonl     — 有复读机但最终判对（rerollout 成功）
  3. exceeded_max_attempts.jsonl  — 有复读机，rerollout 用尽次数仍未判对
  4. no_repetition.jsonl          — 无复读机

只支持新版 schema（stage1_judge / stage2_judge 子字段, 参见 streaming_pipeline.py）。
只扫描含 stage1/has_repetition.jsonl 且该文件首条记录带 stage1_judge 键的目录 —— 早于本次
字段重构跑出的旧版目录（顶层 is_repetition/retry_ok 平铺字段）会被自动跳过，不纳入聚合
（历史数据不兼容，需要重跑或忽略，与本目录既有约定一致）。

用法:
    python aggregate_to_data_all.py --output-base-dir ./output --data-all-dir ./output/data_all
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def read_jsonl(path: Path) -> list:
    """读取 jsonl 文件，跳过无效行；文件不存在时返回空列表。"""
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
            except json.JSONDecodeError:
                continue
    return rows


def _is_new_schema(has_rep_path: Path) -> bool:
    """探测 stage1/has_repetition.jsonl 首条有效记录是否带 stage1_judge 键。"""
    if not has_rep_path.exists():
        return False
    with open(has_rep_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            return "stage1_judge" in rec
    return False


def _classify_rerollout(record: dict) -> str:
    """stage2 记录（repetition_correct.jsonl / repetition_incorrect.jsonl）→ 类别名。"""
    is_correct = (record.get("stage2_judge") or {}).get("is_correct")
    return "repetition_correct" if is_correct else "exceeded_max_attempts"


def main():
    parser = argparse.ArgumentParser(description="聚合已判题数据到 data_all/")
    parser.add_argument("--output-base-dir", required=True,
                        help="Directory containing output/<timestamp>/ subfolders")
    parser.add_argument("--data-all-dir", required=True,
                        help="data_all/ output directory")
    args = parser.parse_args()

    base_dir = Path(args.output_base_dir).resolve()
    data_all_dir = Path(args.data_all_dir).resolve()

    if not base_dir.exists():
        print(f"[WARN] {base_dir} doesn't exist. Nothing to aggregate.")
        return

    data_all_dir.mkdir(parents=True, exist_ok=True)

    categories = ("repetition_incorrect", "repetition_correct",
                  "exceeded_max_attempts", "no_repetition")
    buckets: dict[str, dict[str, dict]] = {
        c: {} for c in categories if c != "no_repetition"
    }
    no_rep_entries: list = []  # no_repetition 允许多行同 uuid（stage1 8x rollout 每条轨迹独立保存）

    # 按目录名排序扫描：同一 uuid 出现在多个 run 中时，后出现（目录名靠后）的覆盖前者，
    # 结果可复现。
    run_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir() and p.name != data_all_dir.name)

    for run_dir in run_dirs:
        stage1_dir = run_dir / "stage1"
        stage2_dir = run_dir / "stage2"
        has_rep_path = stage1_dir / "has_repetition.jsonl"

        if not stage1_dir.is_dir():
            continue  # 非本管线目录（如 output/code、output/math、output/RFT_data 等），天然跳过

        if not _is_new_schema(has_rep_path):
            print(f"[WARN] {run_dir} 跳过旧 schema 目录 (缺少 stage1_judge 字段)")
            continue

        print(f"[INFO] Processing {run_dir}", flush=True)

        no_rep_records = read_jsonl(stage1_dir / "no_repetition.jsonl")
        has_rep_records = read_jsonl(has_rep_path)
        rep_correct_records = read_jsonl(stage2_dir / "repetition_correct.jsonl")
        rep_incorrect_records = read_jsonl(stage2_dir / "repetition_incorrect.jsonl")

        for rec in no_rep_records:
            no_rep_entries.append(rec)

        # stage2 两个终态文件先落桶，再用它们的 uuid 集合筛掉 has_repetition 里已终结的部分
        resolved_uuids: set = set()
        for rec in rep_correct_records:
            uid = rec.get("uuid", "")
            if uid:
                buckets[_classify_rerollout(rec)][uid] = rec
                resolved_uuids.add(uid)
        for rec in rep_incorrect_records:
            uid = rec.get("uuid", "")
            if uid:
                buckets[_classify_rerollout(rec)][uid] = rec
                resolved_uuids.add(uid)

        # has_repetition 中未被 stage2 终结的 uuid → 在途/未完成 → repetition_incorrect
        for rec in has_rep_records:
            uid = rec.get("uuid", "")
            if uid and uid not in resolved_uuids:
                buckets["repetition_incorrect"][uid] = rec

    # ---------- 写出 4 类文件 ----------
    total = 0
    for cat in categories:
        out_path = data_all_dir / f"{cat}.jsonl"
        if cat == "no_repetition":
            entries = no_rep_entries
        else:
            entries = list(buckets[cat].values())
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in entries:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OUTPUT] {out_path} {len(entries)} entries")
        total += len(entries)

    stats = {
        "total": total,
        "categories": {
            cat: len(no_rep_entries) if cat == "no_repetition" else len(buckets[cat])
            for cat in categories
        },
    }
    stats_path = data_all_dir / "aggregation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== 聚合完成 ===\n")
    for cat in categories:
        print(f"  {cat:30s} {stats['categories'][cat]:>5d}")
    print(f"\n共计 {total} 条\n")
    print(f"[INFO] aggregation stats written to {stats_path}")


if __name__ == "__main__":
    main()
