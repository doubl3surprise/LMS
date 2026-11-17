#!/usr/bin/env bash
# =============================================================================
# run_locomo.sh — Mem0 + SGLang LoCoMo 评测一键脚本
#
# 用法:
#   ./run_locomo.sh                  # 完整评测 (152题)
#   ./run_locomo.sh smoke             # 快速烟雾测试 (10题, ~5分钟)
#   ./run_locomo.sh ingest            # 仅执行 ingestion (对话→记忆抽取, ~2-3h)
#   ./run_locomo.sh predict           # 仅执行检索+回答+评分 (需已有记忆库)
#   ./run_locomo.sh report [json_path]# 读取并展示评测结果
#   ./run_locomo.sh status            # 检查服务状态 (SGLang/Mem0/记忆库)
#
# 需要在 DGX 上运行，前置条件:
#   1. SGLang 服务运行在 127.0.0.1:8000
#   2. .benchmark-venv 虚拟环境已安装依赖
# =============================================================================

set -euo pipefail

# ---- 路径常量 ----
REPO_ROOT="$HOME/mem0-reproduction"
BENCH_DIR="$REPO_ROOT/eval"
VENV="$REPO_ROOT/.benchmark-venv"
RESULTS_DIR="$BENCH_DIR/results/locomo"
QDRANT_DIR="$REPO_ROOT/database/qdrant"
DATASET_PATH="$REPO_ROOT/datasets/locomo/locomo10.json"

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
        error "请先运行: cd $REPO_ROOT && python3 -m venv .benchmark-venv && source .benchmark-venv/bin/activate && pip install -r memory-benchmarks/requirements.txt"
        ok=false
    fi

    # 检查 SGLang
    if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        error "SGLang 服务未运行 (127.0.0.1:8000)"
        error "请先启动: cd $REPO_ROOT/model-serving && ./restart.sh"
        ok=false
    else
        info "SGLang 服务正常"
    fi

    # 检查数据集
    if [ ! -f "$DATASET_PATH" ]; then
        warn "数据集不存在: $DATASET_PATH"
        warn "请下载: mkdir -p $(dirname $DATASET_PATH) && wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -O $DATASET_PATH"
    fi

    if [ "$ok" = false ]; then
        exit 1
    fi
}

# ---- 检查记忆库 ----
check_memories() {
    if [ ! -d "$QDRANT_DIR/collection" ] || [ -z "$(ls -A "$QDRANT_DIR/collection" 2>/dev/null)" ]; then
        warn "记忆库为空，需要先执行 ingestion"
        return 1
    fi
    info "记忆库已存在: $QDRANT_DIR"
    return 0
}

# ---- 激活虚拟环境 ----
activate_venv() {
    source "$VENV/bin/activate"
    cd "$BENCH_DIR"
    export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
    export OPENAI_API_KEY="dummy"
}

# ---- 运行评测 ----
run_eval() {
    local mode="${1:-full}"
    local max_questions=152
    local extra_args=""

    case "$mode" in
        smoke)
            max_questions=10
            info "烟雾测试: $max_questions 题"
            ;;
        full)
            info "完整评测: $max_questions 题"
            ;;
    esac

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local project_name="locomo_eval_${mode}_${timestamp}"
    local run_id="${mode}_${timestamp}"

    info "项目名: $project_name"
    info "Run ID: $run_id"
    echo ""

    python3 run_locomo.py \
        --project-name "$project_name" \
        --run-id "$run_id" \
        --conversations 0 \
        --top-k 10 \
        --top-k-cutoffs 10 \
        --max-questions "$max_questions" \
        --answerer-model qwen3.8-27b-zh \
        --judge-model qwen3.8-27b-zh \
        --provider openai
}

# ---- 仅 ingestion ----
run_ingest() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    info "开始 ingestion (对话→记忆抽取), 预计 2-3 小时"

    python3 run_locomo.py \
        --project-name "locomo_ingest_${timestamp}" \
        --conversations 0 \
        --predict-only
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
print(f\"  时间:     {meta.get('timestamp', 'N/A')}\")
print('-' * 50)
print(f\"  总准确率: {overall.get('accuracy', 0):.1f}%  ({overall.get('correct', 0)}/{overall.get('total', 0)})\")
print('-' * 50)
print('  分题型:')
for cat, v in sorted(metrics.get('by_category', {}).items()):
    bar = '█' * int(v['accuracy'] / 5)
    print(f\"    {cat:<15} {v['accuracy']:>5.1f}%  ({v['correct']}/{v['total']}) {bar}\")
print('=' * 50)

# 失败样例
evals = d.get('evaluations', [])
wrong = [e for e in evals if e.get('cutoff_results', {}).get('top_10', {}).get('judgment') == 'WRONG']
if wrong:
    print(f\"\\n  失败样例 (共 {len(wrong)} 题):\")
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
        local model=$(curl -sf http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        info "SGLang: 运行中 (模型: $model)"
    else
        error "SGLang: 未运行"
    fi

    # 记忆库
    if [ -d "$QDRANT_DIR/collection" ]; then
        local collections=$(ls "$QDRANT_DIR/collection" 2>/dev/null | wc -l)
        local size=$(du -sh "$QDRANT_DIR" 2>/dev/null | cut -f1)
        info "记忆库: $collections 个 collection, $size"
    else
        warn "记忆库: 不存在"
    fi

    # 数据集
    if [ -f "$DATASET_PATH" ]; then
        info "数据集: 存在"
    else
        warn "数据集: 不存在"
    fi

    # 结果文件
    local result_count=$(ls "$RESULTS_DIR"/locomo_results_*.json 2>/dev/null | wc -l)
    info "评测结果: $result_count 个结果文件"

    # 最新结果摘要
    local latest=$(ls -t "$RESULTS_DIR"/locomo_results_*.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        local acc=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d['metrics_by_cutoff']['top_10']['overall']['accuracy']:.1f}%\")" 2>/dev/null)
        info "最新结果: $acc ($(basename $latest))"
    fi
    echo ""
}

# ---- 主逻辑 ----
case "${1:-help}" in
    smoke)
        check_prerequisites
        check_memories || { warn "请先运行: $0 ingest"; exit 1; }
        activate_venv
        run_eval smoke
        ;;
    full|"")
        check_prerequisites
        check_memories || { warn "请先运行: $0 ingest"; exit 1; }
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
        check_memories || { error "记忆库为空，请先运行: $0 ingest"; exit 1; }
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
        echo "  smoke       快速烟雾测试 (10题, ~5分钟)"
        echo "  full        完整评测 (152题, ~72分钟)"
        echo "  ingest      仅执行 ingestion (对话→记忆, ~2-3h)"
        echo "  predict     仅执行检索+回答+评分 (需已有记忆库)"
        echo "  report      展示最新评测结果"
        echo "  status      检查系统状态"
        echo ""
        echo "前置条件:"
        echo "  1. SGLang 运行在 127.0.0.1:8000"
        echo "  2. .benchmark-venv 已安装依赖"
        echo "  3. datasets/locomo/locomo10.json 已下载"
        ;;
    *)
        error "未知命令: $1"
        echo "运行 $0 help 查看用法"
        exit 1
        ;;
esac
