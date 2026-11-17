#!/usr/bin/env bash
# =============================================================================
# run_locomo.sh — Mem0 + SGLang LoCoMo 评测一键脚本
#
# 用法:
#   ./run_locomo.sh                  # 完整评测 (152题)
#   ./run_locomo.sh smoke             # 快速烟雾测试 (10题)
#   ./run_locomo.sh ingest            # 仅执行 ingestion (对话→记忆抽取, ~1.5-2h)
#   ./run_locomo.sh predict           # 仅执行检索+回答+评分 (需已有记忆库)
#   ./run_locomo.sh report [json_path]# 读取并展示评测结果
#   ./run_locomo.sh status            # 检查服务状态 (SGLang/Mem0适配器/记忆库)
#
# 关键设计（2026-11-17 修复）:
#   * PROJECT_NAME / RUN_ID 固定，ingest / smoke / full / predict 共用同一个
#     output 目录和同一个 user_id，因此 checkpoint 与记忆库可以跨模式复用，
#     不会每次运行都重新抽取 419 个 chunk。
#   * ingest 只做抽取（--predict-only --max-questions 0），不写逐题结果，
#     避免"无 cutoff_results 的半成品结果被 resume 跳过"导致 0% 准确率。
#   * 记忆库检查通过适配器 /stats 端点完成，不再用目录存在与否判断。
#
# 前置条件:
#   1. SGLang 服务运行在 127.0.0.1:8000
#   2. Mem0 适配器运行在 127.0.0.1:8888 (scripts/server/start_mem0.sh)
#   3. .benchmark-venv 已安装依赖
# =============================================================================

set -euo pipefail

# ---- 路径常量 ----
REPO_ROOT="$HOME/mem0-reproduction"
BENCH_DIR="$REPO_ROOT/eval"
VENV="$REPO_ROOT/.benchmark-venv"
RESULTS_DIR="$BENCH_DIR/results/locomo"
QDRANT_DIR="$REPO_ROOT/database/qdrant"
DATASET_PATH="$REPO_ROOT/datasets/locomo/locomo10.json"

# ---- 固定的 run 身份（可用环境变量覆盖）----
PROJECT_NAME="${PROJECT_NAME:-task1_canonical}"
RUN_ID="${RUN_ID:-$PROJECT_NAME}"
USER_ID="locomo_0_${RUN_ID}"
ADAPTER_URL="${MEM0_HOST:-http://127.0.0.1:8888}"
MODEL="${MEM0_LLM_MODEL:-qwen3.8-27b-zh}"

# ---- 颜色 ----
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- 前置检查 ----
check_prerequisites() {
    local ok=true

    # 检查 venv
    if [ ! -f "$VENV/bin/activate" ]; then
        error "虚拟环境不存在: $VENV"
        error "创建: cd $REPO_ROOT && python3 -m venv .benchmark-venv && .benchmark-venv/bin/pip install -r eval/requirements.txt"
        ok=false
    fi

    # 检查 SGLang
    if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        error "SGLang 服务未运行 (127.0.0.1:8000)"
        error "启动: cd $REPO_ROOT/model-serving && ./restart.sh"
        ok=false
    else
        info "SGLang 服务正常"
    fi

    # 检查 Mem0 适配器（抽取与检索都必须经过它）
    if ! curl -sf "$ADAPTER_URL/health" >/dev/null 2>&1; then
        error "Mem0 适配器未运行 ($ADAPTER_URL)"
        error "启动: nohup $REPO_ROOT/scripts/server/start_mem0.sh > $REPO_ROOT/logs/mem0_adapter.log 2>&1 &"
        ok=false
    else
        info "Mem0 适配器正常 ($ADAPTER_URL)"
    fi

    # 检查数据集
    if [ ! -f "$DATASET_PATH" ]; then
        error "数据集不存在: $DATASET_PATH"
        error "下载: wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -O $DATASET_PATH"
        ok=false
    fi

    if [ "$ok" = false ]; then
        exit 1
    fi
}

# ---- 检查记忆库中是否已有本 run 的记忆 ----
check_store() {
    local body n
    body=$(curl -sf "$ADAPTER_URL/stats?user_id=$USER_ID" 2>/dev/null) || {
        error "无法读取记忆库状态 ($ADAPTER_URL/stats)"
        return 1
    }
    n=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_user_count"])' 2>/dev/null || echo 0)
    if [ "$n" -le 0 ]; then
        warn "user_id=$USER_ID 在记忆库中有 0 条记忆，需要先执行 ingestion"
        return 1
    fi
    info "记忆库: user_id=$USER_ID 共 $n 条记忆"
    return 0
}

# ---- 激活虚拟环境 ----
activate_venv() {
    source "$VENV/bin/activate"
    cd "$BENCH_DIR"
    export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
    export OPENAI_API_KEY="dummy"
    export MEM0_TELEMETRY=false
    export FASTEMBED_CACHE_PATH="$REPO_ROOT/models"
    export HF_ENDPOINT="https://hf-mirror.com"
}

# ---- 运行评测 ----
run_eval() {
    local mode="${1:-full}"
    local max_questions=152

    case "$mode" in
        smoke)
            max_questions="${SMOKE_QUESTIONS:-10}"
            info "烟雾测试: 最多 $max_questions 题（已完成且带 verdict 的题会被跳过）"
            ;;
        full)
            info "完整评测: 最多 $max_questions 题（已完成且带 verdict 的题会被跳过）"
            ;;
    esac

    info "项目名: $PROJECT_NAME  |  Run ID: $RUN_ID  |  user_id: $USER_ID"
    echo ""

    python3 run_locomo.py \
        --project-name "$PROJECT_NAME" \
        --run-id "$RUN_ID" \
        --conversations "${CONVERSATIONS:-0}" \
        --categories "${CATEGORIES:-1,2,3,4}" \
        --top-k 10 \
        --top-k-cutoffs 10 \
        --max-questions "$max_questions" \
        --answerer-model "$MODEL" \
        --judge-model "$MODEL" \
        --provider openai \
        --dataset-path "$DATASET_PATH" \
        --resume
}

# ---- 仅 ingestion ----
run_ingest() {
    info "开始 ingestion (对话→记忆抽取)，conv=${CONVERSATIONS:-0}，预计 1.5-2 小时"
    info "输出目录: eval/results/locomo/predicted_${PROJECT_NAME} (checkpoint 可断点续跑)"
    echo ""

    python3 run_locomo.py \
        --project-name "$PROJECT_NAME" \
        --run-id "$RUN_ID" \
        --conversations "${CONVERSATIONS:-0}" \
        --answerer-model "$MODEL" \
        --judge-model "$MODEL" \
        --provider openai \
        --predict-only \
        --max-questions 0 \
        --dataset-path "$DATASET_PATH" \
        --resume
}

# ---- 展示结果 ----
show_report() {
    local json_path="${1:-}"

    if [ -z "$json_path" ]; then
        json_path=$(ls -t "$RESULTS_DIR"/locomo_results_*.json 2>/dev/null | head -1)
    fi

    if [ -z "$json_path" ] || [ ! -f "$json_path" ]; then
        error "未找到结果文件"
        error "可用文件: ls $RESULTS_DIR/locomo_results_*.json"
        exit 1
    fi

    info "结果文件: $json_path"
    echo ""

    python3 -c "
import json, sys

d = json.load(open('$json_path'))
meta = d.get('metadata', {})
metrics = d.get('metrics_by_cutoff', {}).get('top_10', {})
overall = metrics.get('overall', {})

print('=' * 50)
print('  LoCoMo 评测结果')
print('=' * 50)
print(f\"  模型:     {meta.get('answerer_model', 'N/A')}\")
print(f\"  Judge:   {meta.get('judge_model', 'N/A')}\")
print(f\"  Top-K:   {meta.get('top_k', 'N/A')}\")
print(f\"  Run ID:  {meta.get('run_id', 'N/A')}\")
print(f\"  时间:     {meta.get('timestamp', 'N/A')}\")
print('-' * 50)
print(f\"  总准确率: {overall.get('accuracy', 0):.1f}%  ({overall.get('correct', 0)}/{overall.get('total', 0)})\")
print('-' * 50)
print('  分题型:')
for cat, v in sorted(metrics.get('by_category', {}).items()):
    bar = '█' * int(v['accuracy'] / 5)
    print(f\"    {cat:<15} {v['accuracy']:>5.1f}%  ({v['correct']}/{v['total']}) {bar}\")
print('=' * 50)

evals = d.get('evaluations', [])
wrong = [e for e in evals if e.get('cutoff_results', {}).get('top_10', {}).get('judgment') == 'WRONG']
if wrong:
    print(f\"\n  失败样例 (共 {len(wrong)} 题):\")
    for e in wrong[:3]:
        t = e['cutoff_results']['top_10']
        print(f\"    Q: {e['question'][:70]}\")
        print(f\"    GT: {e['ground_truth_answer'][:70]}\")
        print(f\"    A:  {t['generated_answer'][:70]}\")
        print()
"
}

# ---- 状态检查 ----
show_status() {
    echo "=== 系统状态 ==="
    echo ""

    # SGLang
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        local model
        model=$(curl -sf http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        info "SGLang: 运行中 (模型: $model)"
    else
        error "SGLang: 未运行 (127.0.0.1:8000)"
    fi

    # Mem0 适配器 + 记忆库
    local stats
    if stats=$(curl -sf "$ADAPTER_URL/stats?user_id=$USER_ID" 2>/dev/null); then
        info "Mem0 适配器: 运行中 ($ADAPTER_URL)"
        printf '%s' "$stats" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"  集合: {d['collection']}  总点数: {d['total_points']}  集合数: {d['collections']}\")
print(f\"  本次 user_id: {d['target_user_id']} → {d['target_user_count']} 条\")
rows = sorted(d['users'].items(), key=lambda kv: -kv[1])[:8]
print('  其他 user_id (前8):')
for k, v in rows:
    print(f'    {v:5d}  {k}')
"
    else
        error "Mem0 适配器: 未运行 ($ADAPTER_URL)"
        echo "       启动: nohup $REPO_ROOT/scripts/server/start_mem0.sh > $REPO_ROOT/logs/mem0_adapter.log 2>&1 &"
    fi

    # 数据集
    if [ -f "$DATASET_PATH" ]; then
        info "数据集: 存在 ($DATASET_PATH)"
    else
        warn "数据集: 不存在 ($DATASET_PATH)"
    fi

    # checkpoint / 逐题结果
    local out_dir="$RESULTS_DIR/predicted_${PROJECT_NAME}"
    if [ -f "$out_dir/_ingestion_0.json" ]; then
        local chunks
        chunks=$(python3 -c "import json;print(json.load(open('$out_dir/_ingestion_0.json')).get('total_chunks_processed'))" 2>/dev/null || echo '?')
        info "ingestion checkpoint: $chunks 个 chunk ($out_dir)"
    else
        warn "ingestion checkpoint: 未发现 ($out_dir)"
    fi
    local n_q
    n_q=$(ls "$out_dir"/conv*_q*.json 2>/dev/null | wc -l)
    info "逐题结果: $n_q 个文件"

    # 汇总结果
    local result_count
    result_count=$(ls "$RESULTS_DIR"/locomo_results_*.json 2>/dev/null | wc -l)
    info "汇总结果文件: $result_count 个"
    local latest
    latest=$(ls -t "$RESULTS_DIR"/locomo_results_*.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        local acc
        acc=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d['metrics_by_cutoff']['top_10']['overall']['accuracy']:.1f}%\")" 2>/dev/null)
        info "最新结果: $acc ($(basename "$latest"))"
    fi
    echo ""
}

# ---- 主逻辑 ----
case "${1:-help}" in
    smoke)
        check_prerequisites
        check_store || { warn "请先运行: $0 ingest"; exit 1; }
        activate_venv
        run_eval smoke
        ;;
    full|"")
        check_prerequisites
        check_store || { warn "请先运行: $0 ingest"; exit 1; }
        activate_venv
        run_eval full
        ;;
    ingest)
        check_prerequisites
        activate_venv
        run_ingest
        ;;
    predict)
        check_prerequisites
        check_store || { error "记忆库中没有 user_id=$USER_ID 的记忆，请先运行: $0 ingest"; exit 1; }
        activate_venv
        run_eval full
        ;;
    report)
        show_report "${2:-}"
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        echo "用法: $0 <command>"
        echo ""
        echo "命令:"
        echo "  smoke       烟雾测试 (默认10题，需已有记忆库)"
        echo "  full        完整评测 (152题，需已有记忆库)"
        echo "  ingest      仅抽取记忆 (conv-0 共419个chunk, ~1.5-2h, 可断点续跑)"
        echo "  predict     同 full（需已有记忆库）"
        echo "  report      展示最新评测结果"
        echo "  status      检查 SGLang / 适配器 / 记忆库 / checkpoint 状态"
        echo ""
        echo "固定 run 身份 (可用环境变量覆盖): PROJECT_NAME=$PROJECT_NAME RUN_ID=$RUN_ID"
        echo "  对应 user_id: $USER_ID"
        echo "  output 目录: $RESULTS_DIR/predicted_$PROJECT_NAME"
        echo ""
        echo "前置条件:"
        echo "  1. SGLang 运行在 127.0.0.1:8000"
        echo "  2. Mem0 适配器运行在 127.0.0.1:8888"
        echo "  3. .benchmark-venv 已安装依赖，datasets/locomo/locomo10.json 已下载"
        ;;
    *)
        error "未知命令: $1"
        echo "运行 $0 help 查看用法"
        exit 1
        ;;
esac
