#!/usr/bin/env bash
# 检查评测前置条件

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; FAILED=1; }
warn() { echo -e "${YELLOW}  !${NC} $*"; }

FAILED=0
echo "=== 评测环境检查 ==="
echo ""

# SGLang
if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    MODEL=$(curl -sf http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    ok "SGLang 推理服务: 运行中 (model=$MODEL)"
else
    fail "SGLang 推理服务未运行 (127.0.0.1:8000)"
    echo "       启动: ~/start_sglang_qwen38.sh"
fi

# Mem0 adapter
if curl -sf http://127.0.0.1:8888/health >/dev/null 2>&1; then
    ok "Mem0 适配器服务: 运行中 (127.0.0.1:8888)"
else
    fail "Mem0 适配器服务未运行 (127.0.0.1:8888)"
    echo "       启动: scripts/server/start_mem0.sh"
fi

# venv
if [ -f "$REPO_ROOT/.benchmark-venv/bin/activate" ]; then
    ok "虚拟环境 .benchmark-venv: 存在"
else
    fail "虚拟环境不存在: $REPO_ROOT/.benchmark-venv"
fi

# dataset
if [ -f "$REPO_ROOT/datasets/locomo/locomo10.json" ]; then
    ok "LoCoMo 数据集: 存在"
else
    fail "数据集缺失: datasets/locomo/locomo10.json"
fi

# qdrant
QDRANT="$REPO_ROOT/database/qdrant/collection"
if [ -d "$QDRANT/benchmark_memories" ]; then
    SIZE=$(du -sh "$REPO_ROOT/database/qdrant" 2>/dev/null | cut -f1)
    ok "记忆库 (Qdrant): 已 ingest ($SIZE)"
else
    warn "记忆库为空，需先运行 ingestion: scripts/eval/run_locomo.sh ingest"
fi

echo ""
if [ "$FAILED" = "1" ]; then
    echo -e "${RED}有前置条件未满足，请修复后重试${NC}"
    exit 1
else
    echo -e "${GREEN}所有前置条件就绪，可以运行评测${NC}"
fi
