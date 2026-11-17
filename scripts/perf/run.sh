#!/usr/bin/env bash
# 运行性能实验（Task 3-6）
#
# 注意：这些脚本已完成，性能数据已保存在 trace_results/
# 建议只用来参考或在特殊情况下重新测试
#
# 用法:
#   scripts/perf/run.sh [task3|task4|task5|task6|validate]
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PERF_DIR="$REPO_ROOT/perf"
VENV="$REPO_ROOT/.venv"

cd "$PERF_DIR"
source "$VENV/bin/activate"

TASK="${1:-help}"

case "$TASK" in
    task3|task4|task5|task6)
        echo "[perf] 运行 $TASK..."
        python3 project.py run --platform dgx --task "$TASK"
        ;;
    validate)
        echo "[perf] 验证环境..."
        python3 project.py validate
        ;;
    *)
        echo "用法: $(basename "$0") [task3|task4|task5|task6|validate]"
        echo ""
        echo "已完成的性能实验:"
        echo "  task3  - 端到端性能分解 (trace_results/task3_summary/)"
        echo "  task4  - 前缀缓存分析 (trace_results/task4/)"
        echo "  task5  - 多轮会话 + MMR 对比 (trace_results/task5/)"
        echo "  task6  - 并发/依赖分析 (trace_results/task6_final/)"
        echo ""
        echo "结果查看: trace_results/ 目录下的 report.md"
        exit 1
        ;;
esac
