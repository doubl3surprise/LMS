#!/usr/bin/env python3
"""记忆库盘点：按 user_id 列出记忆条数。

为什么要这个脚本：评测脚本过去用 "Qdrant 目录是否存在" 判断记忆库是否就绪，
该判断在记忆库里没有本 run 的 user_id 时依然为真，于是 smoke/full 会静默地
检索到 0 条记忆，最后得到一个看似正常却无意义的准确率。

优先通过适配器 /stats 读取（适配器进程独占 Qdrant 文件锁，另开客户端会报
RuntimeError: Storage folder ... already accessed by another instance）。
适配器未运行时才直接读本地 Qdrant，此时必须确认没有别的进程在用该路径。

用法:
  .benchmark-venv/bin/python scripts/debug/store_inventory.py
  .benchmark-venv/bin/python scripts/debug/store_inventory.py --user-id locomo_0_task1_canonical
  .benchmark-venv/bin/python scripts/debug/store_inventory.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT = REPO_ROOT / "database" / "qdrant"
COLLECTION = "benchmark_memories"


def from_adapter(adapter: str, user_id: str | None) -> dict:
    url = f"{adapter.rstrip('/')}/stats"
    if user_id:
        url += "?user_id=" + urllib.parse.quote(user_id)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def from_qdrant(qdrant_path: str, user_id: str | None) -> dict:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=qdrant_path)
    try:
        names = [c.name for c in client.get_collections().collections]
        total = client.count(COLLECTION).count if COLLECTION in names else 0
        users: dict[str, int] = {}
        if COLLECTION in names:
            offset = None
            while True:
                points, offset = client.scroll(
                    COLLECTION, limit=1000, with_payload=["user_id"], with_vectors=False, offset=offset,
                )
                for point in points:
                    key = str((point.payload or {}).get("user_id"))
                    users[key] = users.get(key, 0) + 1
                if not offset:
                    break
    finally:
        client.close()

    return {
        "status": "ok",
        "source": "qdrant-direct",
        "collection": COLLECTION,
        "collections": len(names),
        "collection_names": names,
        "total_points": total,
        "users": users,
        "target_user_id": user_id,
        "target_user_count": users.get(user_id, 0) if user_id else None,
        "qdrant_path": qdrant_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="记忆库 user_id 盘点")
    ap.add_argument("--adapter", default="http://127.0.0.1:8888")
    ap.add_argument("--user-id", default=None, help="需要重点确认的 user_id（评测用的那个）")
    ap.add_argument("--qdrant", default=str(DEFAULT_QDRANT))
    ap.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = ap.parse_args()

    try:
        stats = from_adapter(args.adapter, args.user_id)
        stats.setdefault("source", "adapter")
    except Exception as exc:  # adapter down -> direct read
        print(f"[warn] 适配器不可用 ({exc}); 直接读取 {args.qdrant}（确保没有其它进程占用）",
              file=sys.stderr)
        stats = from_qdrant(args.qdrant, args.user_id)

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    users: dict[str, int] = stats.get("users", {})
    print(f"来源: {stats.get('source')}   集合: {stats.get('collection')}   "
          f"总点数: {stats.get('total_points')}   collection 数: {stats.get('collections')}")
    print(f"Qdrant 路径: {stats.get('qdrant_path')}")
    if stats.get("llm_model"):
        print(f"抽取模型: {stats['llm_model']}   Embedding: {stats['embedder']}")
    print("-" * 62)
    for uid, count in sorted(users.items(), key=lambda kv: -kv[1]):
        mark = "  ← 本次评测" if uid == args.user_id else ""
        print(f"{count:6d}  {uid}{mark}")
    print("-" * 62)

    target = args.user_id
    if target:
        n = stats.get("target_user_count") or 0
        if n > 0:
            print(f"[OK] {target}: {n} 条记忆，可以跑 smoke/full")
            return 0
        print(f"[FAIL] {target}: 0 条记忆 → 需要先运行 scripts/eval/run_locomo.sh ingest")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
