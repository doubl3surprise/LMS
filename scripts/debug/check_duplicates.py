#!/usr/bin/env python3
"""检查某个 user_id 下是否存在完全重复的记忆文本。

用途：runner 侧的 HTTP 超时（默认 900s）如果短于适配器侧一次抽取的真实耗时，
会出现"runner 判失败并重试、适配器其实已写库"的情况，于是同一条记忆被写两次。
Mem0 2.2.1 的点 ID 是随机 uuid4，重复写入不会被自动覆盖，只会在文本层重复。

用法:
  .benchmark-venv/bin/python scripts/debug/check_duplicates.py --user-id locomo_0_task1_canonical
  .benchmark-venv/bin/python scripts/debug/check_duplicates.py --user-id ... --json
退出码: 0 无重复 / 2 有重复
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter


def fetch(adapter: str, user_id: str) -> list[dict]:
    url = f"{adapter.rstrip('/')}/memories?user_id=" + urllib.parse.quote(user_id)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.load(resp)
    if isinstance(data, dict):
        return data.get("results", data.get("memories", [])) or []
    return data or []


def main() -> int:
    ap = argparse.ArgumentParser(description="检查记忆文本重复")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--adapter", default="http://127.0.0.1:8888")
    ap.add_argument("--show", type=int, default=10, help="最多打印多少条重复样例")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    items = fetch(args.adapter, args.user_id)
    texts = [(item.get("memory") or item.get("data") or "").strip() for item in items]
    counter = Counter(t for t in texts if t)
    dupes = {text: n for text, n in counter.items() if n > 1}

    summary = {
        "user_id": args.user_id,
        "total": len(items),
        "unique": len(counter),
        "duplicate_texts": len(dupes),
        "extra_copies": sum(n - 1 for n in dupes.values()),
    }

    if args.json:
        print(json.dumps({**summary, "duplicates": dupes}, ensure_ascii=False, indent=2))
    else:
        print(f"user_id={args.user_id}  总记忆={summary['total']}  唯一文本={summary['unique']}")
        if dupes:
            print(f"[FAIL] {len(dupes)} 条文本重复，合计多出 {summary['extra_copies']} 条副本：")
            for text, n in sorted(dupes.items(), key=lambda kv: -kv[1])[: args.show]:
                print(f"  x{n}  {text[:100]}")
        else:
            print("[OK] 没有完全重复的记忆文本")

    return 2 if dupes else 0


if __name__ == "__main__":
    sys.exit(main())
