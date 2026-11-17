# scripts/perf/ — 性能实验脚本

## 概述

Task 3-6 的性能分析脚本已集中到此目录。这些实验用来分析系统性能瓶颈。

**状态**: 已完成，结果保存在 `trace_results/`。建议**只做参考**，不需要重新运行。

## 结构

```
scripts/perf/
├── run.sh          ← 一键脚本（封装 project.py）
├── project.py      ← 统一入口（moved from perf/）
└── README.md       ← 本文件
```

## 运行方式

```bash
# 查看帮助
scripts/perf/run.sh

# 运行特定任务（会耗费 GPU 时间）
scripts/perf/run.sh task3    # 端到端性能分解
scripts/perf/run.sh task4    # 前缀缓存分析
scripts/perf/run.sh task5    # 多轮会话 + MMR
scripts/perf/run.sh task6    # 并发测试
```

## 实验说明

| 任务 | 耗时 | 输出目录 | 说明 |
|---|---|---|---|
| task3 | ~2h | trace_results/task3_summary/ | 逐题逐个阶段的执行时间分解 |
| task4 | ~1.5h | trace_results/task4/ | 缓存命中率分析 |
| task5 | ~1.5h | trace_results/task5/ | 多轮 vs 单轮、MMR 有效性 |
| task6 | ~2h | trace_results/task6_final/ | 并发度、资源竞争 |

## 核心性能指标（已测得）

**DGX (TP=1) vs 5880ada (TP=4)**:
- Decode 时延: **13.2x** 差异（GB10 带宽瓶颈）
- 端到端: **10x** 差异
- 原因: 内存带宽（GB10 单卡 vs 5880ada TP=4 聚合）

## 实验环境

- 虚拟环境: `.venv`（注意不是 `.benchmark-venv`）
- 推理服务: SGLang on 127.0.0.1:8000
- 记忆库: database/qdrant
- 数据集: datasets/locomo/locomo10.json

---

**最后更新**: 2026-11-17
