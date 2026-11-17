#!/usr/bin/env bash
# 检查评测前置条件
#
# 2026-11-17 修复：记忆库检查改为按 user_id 查询适配器 /stats。
# 旧的"目录存在即通过"判断在库里没有本次 run 的记忆时仍会打勾，
# 导致 smoke/full 检索到 0 条记忆却没有任何报错。

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/.benchmark-venv"
PROJECT_NAME="${PROJECT_NAME:-task1_canonical}"
RUN_ID="${RUN_ID:-$PROJECT_NAME}"
USER_ID="locomo_0_${RUN_ID}"
ADAPTER_URL="${MEM0_HOST:-http://127.0.0.1:8888}"

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
    echo "       启动: $REPO_ROOT/model-serving/start.sh"
fi

# Mem0 adapter
if curl -sf "$ADAPTER_URL/health" >/dev/null 2>&1; then
    ok "Mem0 适配器服务: 运行中 ($ADAPTER_URL)"
else
    fail "Mem0 适配器服务未运行 ($ADAPTER_URL)"
    echo "       启动: nohup $REPO_ROOT/scripts/server/start_mem0.sh > $REPO_ROOT/logs/mem0_adapter.log 2>&1 &"
fi

# venv
if [ -f "$VENV/bin/activate" ]; then
    if "$VENV/bin/python" -c "import mem0, fastembed, qdrant_client" 2>/dev/null; then
        ok "虚拟环境 .benchmark-venv: 存在且依赖可导入"
    else
        fail "虚拟环境依赖不完整: $VENV (需要 mem0ai/fastembed/qdrant-client)"
    fi
else
    fail "虚拟环境不存在: $VENV"
    echo "       创建: python3 -m venv .benchmark-venv && .benchmark-venv/bin/pip install -r eval/requirements.txt"
fi

# perf venv（性能实验用，独立环境）
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    ok "虚拟环境 .venv: 存在（perf/ 性能实验使用）"
else
    warn "虚拟环境 .venv 不存在（只影响 perf/ 性能实验）"
fi

# dataset
if [ -f "$REPO_ROOT/datasets/locomo/locomo10.json" ]; then
    ok "LoCoMo 数据集: 存在"
else
    fail "数据集缺失: datasets/locomo/locomo10.json"
fi

# 记忆库：必须按 user_id 确认，而不是看目录
if curl -sf "$ADAPTER_URL/health" >/dev/null 2>&1; then
    STATS=$(curl -sf "$ADAPTER_URL/stats?user_id=$USER_ID" 2>/dev/null)
    COUNT=$(printf '%s' "$STATS" | python3 -c "import json,sys; print(json.load(sys.stdin)['target_user_count'])" 2>/dev/null || echo 0)
    TOTAL=$(printf '%s' "$STATS" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_points'])" 2>/dev/null || echo '?')
    if [ "${COUNT:-0}" -gt 0 ]; then
        ok "记忆库: user_id=$USER_ID 共 $COUNT 条（集合总点数 $TOTAL）"
    else
        warn "记忆库: user_id=$USER_ID 为 0 条（集合总点数 $TOTAL）→ 需先运行 scripts/eval/run_locomo.sh ingest"
    fi
else
    warn "记忆库: 适配器未运行，无法确认 user_id=$USER_ID 的记忆数量"
fi

echo ""
if [ "$FAILED" = "1" ]; then
    echo -e "${RED}有前置条件未满足，请修复后重试${NC}"
    exit 1
else
    echo -e "${GREEN}所有前置条件就绪，可以运行评测${NC}"
fi
