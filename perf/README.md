# perf/ — 性能实验

这个目录存放系统性能分析相关的脚本和数据，对应 task3-6 的性能分解研究。

**状态：已完成**。实验结果在 `../trace_results/`，不需要重新运行。

## 目录结构

```
perf/
├── project.py              ← 统一入口（run/validate/profile/bandwidth/report）
├── tracer/                 ← 观测和追踪框架
│   ├── ...
├── task3/                  ← 端到端性能分解实验
│   ├── task3_matrix.py     ← 矩阵测试（主脚本）
│   ├── task3_pilot.py      ← 先导实验
│   └── ...
├── task4/                  ← 前缀缓存分析
│   ├── analyze.py
│   ├── cache_core.py
│   └── ...
├── task5/                  ← 多轮会话 + MMR 对比
│   ├── benchmark.py
│   └── ...
├── task6/                  ← 并发测试 + 依赖分析
│   ├── concurrency.py
│   └── ...
├── run_trace_v4.py         ← 推理链路追踪主脚本
├── trace_core_v4.py        ← 追踪框架核心
├── host_sampler_v4.py      ← 主机资源采样
├── host_resources_v4.py    ← 资源分析
├── measure_bandwidth.py    ← 带宽测量
├── request_profile.py      ← 请求画像
├── request_resources.py    ← 请求资源分析
├── render_trace_charts_v4.py ← 生成追踪图表
├── calibrate_host.py       ← 主机校准
├── run_contention_v4.py    ← 资源竞争测试
├── run_trace_suite_v4.py   ← 完整追踪套件
├── validate_workloads.py   ← 工作负载验证
└── task5_run.py / task6_probe.py
```

## 实验结果摘要

| 任务 | 研究内容 | 核心发现 | 结果位置 |
|---|---|---|---|
| Task3 | 端到端性能分解 | Decode 占 87.9%（DGX 瓶颈）| `../trace_results/task3_summary/` |
| Task4 | 前缀缓存分析 | 缓存命中率 ~65%，重复 prompt 获益 | `../trace_results/task4/` |
| Task5 | 多轮会话 + MMR | MMR 对精度无显著提升 | `../trace_results/task5/` |
| Task6 | 并发测试 | 单 GPU 并发度受限于 KV cache | `../trace_results/task6_final/` |

### DGX vs 5880ada 性能差异

| 阶段 | DGX (TP=1) | 5880ada (TP=4) | 差距 |
|---|---|---|---|
| Prefill | 270ms | 60ms | 4.5x |
| Decode | 3300ms | 250ms | **13.2x** ← 主要差距 |
| Judge | 800ms | 90ms | 8.9x |
| 总计 | 4432ms | 442ms | **10x** |

> DGX 慢 10 倍的根本原因：TP=1 时内存带宽受限（GB10 单卡带宽 vs 5880ada TP=4 聚合带宽）。

## 如何重新运行

**注意**：这些实验已完成，建议不要重新运行，以免覆盖 `../trace_results/` 中的结果。

如果确实需要，从 `perf/` 目录运行：

```bash
cd ~/mem0-reproduction/perf
python3 project.py run --platform dgx
```

运行脚本需要的 Python 环境：

```bash
source ../.venv/bin/activate    # 注意：用 .venv，不是 .benchmark-venv
```

---

**最后更新**: 2026-11-17
