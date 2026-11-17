#!/usr/bin/env python3
"""对比两份 LoCoMo 结果 JSON：总体/分类别准确率差异 + 逐题翻转。

用途：换了抽取配置或 prompt 之后，需要一个可复核的对比，而不是只看总分。
默认按 question_id 对齐，只统计两份文件都存在的题。

用法:
  .benchmark-venv/bin/python scripts/eval/compare_results.py \
      eval/results/locomo/locomo_results_A.json eval/results/locomo/locomo_results_B.json

  # 只看翻转的题
  .benchmark-venv/bin/python scripts/eval/compare_results.py A.json B.json --flips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CUTOFF = "top_10"


def load(path: Path) -> tuple[dict[str, dict], dict]:
    data = json.loads(path.read_text())
    rows = {}
    for item in data.get("evaluations", []):
        verdict = (item.get("cutoff_results") or {}).get(CUTOFF, {})
        if not verdict:
            continue
        rows[item["question_id"]] = {
            "category": item.get("category_name", item.get("category")),
            "question": item.get("question", ""),
            "gold": item.get("ground_truth_answer", ""),
            "answer": verdict.get("generated_answer", ""),
            "correct": verdict.get("judgment") == "CORRECT",
        }
    meta = data.get("metadata", {})
    return rows, meta


def accuracy(rows: list[dict]) -> float:
    return 100.0 * sum(1 for r in rows if r["correct"]) / len(rows) if rows else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="对比两份 LoCoMo 结果")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--flips", action="store_true", help="打印逐题翻转明细")
    ap.add_argument("--limit", type=int, default=15, help="每类翻转最多打印多少条")
    args = ap.parse_args()

    a_rows, a_meta = load(Path(args.baseline))
    b_rows, b_meta = load(Path(args.candidate))
    common = sorted(set(a_rows) & set(b_rows))

    if not common:
        print("两份结果没有可对齐的题目（question_id 不重叠）", file=sys.stderr)
        return 1

    a = [a_rows[q] for q in common]
    b = [b_rows[q] for q in common]

    print(f"A: {args.baseline}  run_id={a_meta.get('run_id')}  n={len(a)}")
    print(f"B: {args.candidate}  run_id={b_meta.get('run_id')}  n={len(b)}")
    print("=" * 64)
    print(f"总体:  A {accuracy(a):.1f}%  →  B {accuracy(b):.1f}%   (Δ {accuracy(b) - accuracy(a):+.1f} pt)")
    print("-" * 64)

    categories = sorted({r["category"] for r in a})
    for cat in categories:
        ac = [r for r in a if r["category"] == cat]
        bc = [r for r in b if r["category"] == cat]
        print(f"{cat:<14} n={len(ac):<4} A {accuracy(ac):5.1f}%  →  B {accuracy(bc):5.1f}%   (Δ {accuracy(bc) - accuracy(ac):+.1f} pt)")

    gained = [q for q in common if not a_rows[q]["correct"] and b_rows[q]["correct"]]
    lost = [q for q in common if a_rows[q]["correct"] and not b_rows[q]["correct"]]
    print("-" * 64)
    print(f"翻转:  A错→B对 {len(gained)} 题;  A对→B错 {len(lost)} 题")

    if args.flips:
        for title, ids in (("A对→B错（回归）", lost), ("A错→B对（改善）", gained)):
            if not ids:
                continue
            print(f"\n{title} ({len(ids)}):")
            for q in ids[: args.limit]:
                row = b_rows[q]
                print(f"  {q} [{row['category']}] {row['question'][:70]}")
                print(f"     gold: {row['gold'][:70]}")
                print(f"     A: {a_rows[q]['answer'][:70]}")
                print(f"     B: {row['answer'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
