#!/usr/bin/env bash
# 启动 Mem0 OSS 本地 HTTP 适配器服务（监听 127.0.0.1:8888）
# 评测前必须先运行本脚本

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.benchmark-venv"
SERVER="$REPO_ROOT/eval/local_mem0_server.py"

export MEM0_TELEMETRY=false
export HF_ENDPOINT="https://hf-mirror.com"
export FASTEMBED_CACHE_PATH="$REPO_ROOT/models"

echo "[start_mem0] 启动 Mem0 适配器服务..."
echo "[start_mem0] 数据目录: $REPO_ROOT/database"
echo "[start_mem0] 监听地址: http://127.0.0.1:8888"
echo ""
echo "集合名:  benchmark_memories"
echo "Qdrant:  $REPO_ROOT/database/qdrant"
echo "SQLite:  $REPO_ROOT/database/history.db"
echo ""

source "$VENV/bin/activate"
exec uvicorn --app-dir "$REPO_ROOT/eval" local_mem0_server:app \
    --host 127.0.0.1 --port 8888 --log-level info
