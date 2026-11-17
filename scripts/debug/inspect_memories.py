#!/usr/bin/env python3
"""
查看 Qdrant 记忆库中的记忆条目。

用法:
    python3 inspect_memories.py --user-id conv-26 --limit 10
    python3 inspect_memories.py --user-id conv-26 --search '学习框架'
    python3 inspect_memories.py --list-users
"""
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def get_client():
    sys.path.insert(0, str(REPO_ROOT / 'eval'))
    from benchmarks.common.mem0_client import Mem0Client
    return Mem0Client(mode='oss', host='http://localhost:8888')

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--user-id', default='conv-26', help='用户ID (default: conv-26)')
    p.add_argument('--limit', type=int, default=10, help='显示条数 (default: 10)')
    p.add_argument('--search', help='语义搜索关键词')
    p.add_argument('--list-users', action='store_true', help='列出所有 user_id')
    p.add_argument('--json', action='store_true', help='输出 JSON 格式')
    a = p.parse_args()

    import asyncio

    async def run():
        async with get_client() as client:
            if a.list_users:
                users = await client.list_users()
                print('可用 user_id:')
                for u in users:
                    print(f'  {u}')
                return

            if a.search:
                print(f'搜索: {a.search!r} (user_id={a.user_id})')
                results = await client.search(a.search, a.user_id, top_k=a.limit)
                items = [(r['score'], r['memory']) for r in results]
            else:
                print(f'最近 {a.limit} 条记忆 (user_id={a.user_id})')
                results = await client.get_all(a.user_id, limit=a.limit)
                items = [(None, r['memory']) for r in results]

            if a.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
                return

            print('=' * 60)
            for i, (score, mem) in enumerate(items, 1):
                prefix = f'{score:.3f} | ' if score is not None else ''
                print(f'{i:3}. {prefix}{mem}')
            print(f'总计: {len(items)} 条')

    asyncio.run(run())

if __name__ == '__main__':
    main()
