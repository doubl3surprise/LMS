#!/usr/bin/env python3
"""记忆库导出 / 导入（备份与恢复）。

背景：database/qdrant 被 .gitignore 忽略，一次目录整理就可能让"产出 80.9% 的那批
记忆"彻底消失（2026-11-17 复盘确认：当前库里 359 条全是 11-09 的旧 smoke 数据，
user_id 里没有 locomo_0_task1-dgx-sdk-20261116）。重新 ingest 要 1.5-2 小时，
所以 ingest 完成后应立刻导出一份 JSON 备份。

导出（适配器在跑也可以，走 /export，不需要抢文件锁）:
  .benchmark-venv/bin/python scripts/debug/export_store.py export \
      --user-id locomo_0_task1_canonical --out database/exports/task1_canonical.json

导入（必须先停掉适配器，否则 Qdrant 文件锁冲突）:
  .benchmark-venv/bin/python scripts/debug/export_store.py import \
      --file database/exports/task1_canonical.json

导出文件结构: {"collection", "user_id", "count", "points": [{id, vector, payload}]}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT = REPO_ROOT / "database" / "qdrant"
EXPORT_DIR = REPO_ROOT / "database" / "exports"
COLLECTION = "benchmark_memories"
VECTOR_SIZE = 384


def adapter_alive(adapter: str) -> bool:
    try:
        with urllib.request.urlopen(f"{adapter.rstrip('/')}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def export_via_adapter(adapter: str, user_id: str, out: Path) -> dict:
    url = f"{adapter.rstrip('/')}/export?user_id=" + urllib.parse.quote(user_id)
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = json.load(resp)
    data["exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["source"] = "adapter"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False))
    return data


def export_via_qdrant(qdrant: str, user_id: str, out: Path, collection: str) -> dict:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=qdrant)
    points: list[dict] = []
    try:
        offset = None
        while True:
            batch, offset = client.scroll(
                collection, limit=256, with_payload=True, with_vectors=True, offset=offset,
                scroll_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
            )
            for point in batch:
                vector = point.vector
                if isinstance(vector, dict):
                    vector = next(iter(vector.values()), None)
                points.append({"id": str(point.id), "vector": vector, "payload": point.payload})
            if not offset:
                break
    finally:
        client.close()

    data = {
        "collection": collection,
        "user_id": user_id,
        "count": len(points),
        "points": points,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "qdrant-direct",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False))
    return data


def do_export(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else EXPORT_DIR / f"{args.user_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    if adapter_alive(args.adapter):
        data = export_via_adapter(args.adapter, args.user_id, out)
    else:
        data = export_via_qdrant(args.qdrant, args.user_id, out, args.collection)
    print(f"[OK] 导出 {data['count']} 条记忆 (来源: {data['source']}) -> {out}")
    if data["count"] == 0:
        print("[warn] 该 user_id 没有记忆可导出——确认 user_id 与 ingest 时一致", file=sys.stderr)
        return 2
    return 0


def do_import(args: argparse.Namespace) -> int:
    from qdrant_client import QdrantClient, models

    if adapter_alive(args.adapter):
        print("[error] 适配器正在运行，Qdrant 文件锁被占用；先停它再导入：", file=sys.stderr)
        print("        pkill -f 'local_mem0_server:app'", file=sys.stderr)
        return 1

    payload = json.loads(Path(args.file).read_text())
    points = payload.get("points", [])
    target_collection = args.collection or payload.get("collection", COLLECTION)
    if not points:
        print(f"[error] 备份文件里没有 points: {args.file}", file=sys.stderr)
        return 1

    client = QdrantClient(path=args.qdrant)
    try:
        existing = {c.name for c in client.get_collections().collections}
        if target_collection not in existing:
            client.create_collection(
                target_collection,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            print(f"[info] 创建集合 {target_collection}")

        written = 0
        for i in range(0, len(points), 128):
            batch = points[i:i + 128]
            client.upsert(
                target_collection,
                points=[
                    models.PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p.get("payload") or {},
                    )
                    for p in batch
                ],
            )
            written += len(batch)
    finally:
        client.close()

    print(f"[OK] 导入 {written} 条记忆到 {target_collection} ({args.qdrant})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="记忆库导出/导入")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="导出某个 user_id 的全部记忆")
    e.add_argument("--user-id", required=True)
    e.add_argument("--out", default=None)
    e.add_argument("--adapter", default="http://127.0.0.1:8888")
    e.add_argument("--qdrant", default=str(DEFAULT_QDRANT))
    e.add_argument("--collection", default=COLLECTION)
    e.set_defaults(func=do_export)

    i = sub.add_parser("import", help="从导出文件恢复记忆（需先停适配器）")
    i.add_argument("--file", required=True)
    i.add_argument("--adapter", default="http://127.0.0.1:8888")
    i.add_argument("--qdrant", default=str(DEFAULT_QDRANT))
    i.add_argument("--collection", default=None)
    i.set_defaults(func=do_import)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
