# Mem0 + SGLang 长期记忆系统研究

长期记忆增强 LLM 系统的设计、实现和评测。

## 快速开始

### 一句话入门

```bash
cd ~/mem0-reproduction/scripts/eval
./run_locomo.sh status         # 检查环境
./run_locomo.sh smoke          # 10题测试 (~5分钟)
./run_locomo.sh full           # 完整152题评测 (~72分钟)
```

### 前置条件

- **SGLang 推理服务**: `http://127.0.0.1:8000` (模型: `qwen3.8-27b-zh`)
  - 检查: `curl http://127.0.0.1:8000/v1/models`
  - 启动: `cd model-serving && ./restart.sh`
  
- **Python 虚拟环境**: `.benchmark-venv` 已安装依赖
  - 激活: `source .benchmark-venv/bin/activate`
  
- **LoCoMo 数据集**: `datasets/locomo/locomo10.json` 已下载
  - 获取: `wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -O datasets/locomo/locomo10.json`

## 目录导览

```
mem0-reproduction/
│
├── 📋 README.md (本文件)
│
├── 📁 eval/                        ★ 评测框架
│   ├── run_locomo.py              ← 标准 LoCoMo-10 评测入口
│   ├── local_mem0_server.py        ← Mem0 HTTP 适配器
│   ├── benchmarks/                 ← 核心评测代码
│   │   ├── common/                 LLM/Mem0 客户端，指标计算
│   │   ├── locomo/                 LoCoMo-10 基准
│   │   ├── longmemeval/            LongMemEval 基准（备用）
│   │   └── beam/                   BEAM 基准（备用）
│   ├── results/locomo/             ← 评测结果 JSON
│   └── README.md                   ← eval 目录详细说明
│
├── 📁 perf/                        ★ 性能实验（已完成）
│   ├── project.py                  ← 统一入口
│   ├── tracer/                     ← 观测和追踪框架
│   ├── task3/ task4/ task5/ task6/ ← 性能分解、缓存、并发分析
│   ├── run_trace_v4.py             ← 主性能测试脚本
│   └── README.md                   ← 详细说明
│
├── 📁 scripts/                     ★ 常用脚本
│   ├── eval/
│   │   ├── run_locomo.sh           ← 一键评测脚本
│   │   └── README.md
│   ├── debug/
│   │   ├── inspect_memories.py     ← 查看记忆库
│   │   └── test_retrieval.py       ← 测试检索
│   └── setup/
│       ├── env.sh                  ← 环境激活
│       └── check_env.sh            ← 环境检查
│
├── 📁 docs/                        ★ 文档
│   ├── ARCHITECTURE.md             ← 系统架构和执行流程 ★必读
│   ├── notes/                      ← 机制分析笔记
│   │   ├── 01-system-map.md
│   │   ├── 02-request-flows.md
│   │   └── ...
│   ├── CURRENT_SYSTEM.md           ← 当前系统状态
│   └── EXPERIMENT_ENVIRONMENT.md   ← 实验环境
│
├── 📁 research/                    ← 进行中的研究
│   ├── candidate_depth_probe/
│   ├── controlled_overlap/
│   └── selection_probe/
│
├── 📁 benchmark-data/              ← 数据和记忆库
│   ├── qdrant/                     ← 记忆向量库 (4.7MB)
│   ├── history.db                  ← 会话历史
│   ├── locomo_task1_dgx_20261116/  ← 评测工作目录
│   └── ...
│
├── 📁 datasets/                    ← 数据集
│   ├── locomo/locomo10.json        ← LoCoMo-10 官方
│   └── ...
│
├── 📁 model-serving/               ← SGLang 模型服务
│   ├── chat.py
│   ├── restart.sh
│   └── cache/                      ← GPU 编译缓存
│
├── 📁 model-serving/               ← 本地 LLM 模型
│   └── qwen3.8-27b-zh/
│
├── 📁 results/                     ← 结果聚合目录
│   ├── locomo/                     ← LoCoMo 汇总结果
│   └── analysis/                   ← 分析报告
│
├── 📁 trace_results/               ← 性能实验结果（只读）
│   ├── task3_summary/
│   ├── task4/
│   ├── task5/
│   └── task6_final/
│
└── .venv, .benchmark-venv          ← Python 虚拟环境
```

## 核心指标

### LoCoMo-10 评测结果（conv_0, 152题, DGX）

| 题型 | 准确率 | 题数 | 主要困难 |
|---|---|---|---|
| **temporal** | 89.2% | 37 | 相对时间推理 ✓ |
| **single-hop** | 82.9% | 70 | 直接检索 ✓ |
| **open-domain** | 76.9% | 13 | 信息不足 |
| **multi-hop** | 68.8% | 32 | **跨记忆推理** ← 瓶颈 |
| **总计** | **80.9%** | **152** | |

### 性能特征（端到端）

| 平台 | Prefill | Decode | Judge | 总时间 | 吞吐量 |
|---|---|---|---|---|---|
| **DGX (TP=1)** | 270ms | 3300ms | 800ms | 4432ms | 22.9 题/分 |
| **5880ada (TP=4)** | 60ms | 250ms | 90ms | 442ms | 136 题/分 |
| **倍数** | 4.5x | 13.2x | 8.9x | **10x** | |

## 常见任务

### 1. 快速验证系统是否可用

```bash
scripts/eval/run_locomo.sh status
```

### 2. 运行完整评测

```bash
cd scripts/eval
./run_locomo.sh full
# 或详细指定
./run_locomo.sh full --project-name my_eval_20261117
```

### 3. 查看最新结果

```bash
scripts/eval/run_locomo.sh report
```

### 4. 查看记忆库中的记忆

```bash
cd eval
python3 -c "
import json
from benchmarks.common.mem0_client import Mem0Client
import asyncio

async def main():
    async with Mem0Client(mode='oss', host='http://localhost:8888') as client:
        results = await client.search('What did Caroline learn?', 'conv-26', top_k=5)
        for r in results:
            print(f\"Score: {r['score']:.3f} | {r['memory'][:60]}\")

asyncio.run(main())
"
```

### 5. 重新 ingest（抽取记忆）

```bash
scripts/eval/run_locomo.sh ingest
# 预计 2-3 小时
```

### 6. 运行性能实验

```bash
cd perf
python3 project.py run --platform dgx --sessions --judge
```

## 学习路径

**新接手者**：
1. 读 `docs/ARCHITECTURE.md` （30分钟）— 系统架构和数据流
2. 读 `docs/notes/02-request-flows.md` （20分钟）— 追踪一条消息
3. 运行 `scripts/eval/run_locomo.sh smoke` （5分钟）— 感受完整流程
4. 查看 `scripts/eval/run_locomo.sh report` （10分钟）— 理解结果格式

**想修改评测逻辑**：
- 编辑 `eval/benchmarks/locomo/prompts.py` （提示词模板）
- 编辑 `eval/benchmarks/locomo/run.py` （评测框架）
- 或修改 `eval/run_locomo.py` （入口点）

**想改进记忆检索**：
- 编辑 `eval/benchmarks/common/mem0_client.py` （Mem0 适配器）
- 调整 `local_mem0_server.py` 中的融合权重 (0.7 * dense + 0.2 * BM25 + 0.1 * entity)

**想加新数据集**：
- 参考 `eval/benchmarks/locomo/` 目录结构
- 在 `eval/benchmarks/mynewbench/` 创建 `run.py` 和 `prompts.py`
- 从 `eval/run_locomo.py` 复制入口逻辑

## 关键概念

### 152题 = LoCoMo-10 的一个会话(conv_0)，筛选掉 adversarial 题

- LoCoMo-10 总共 1986 题 × 10 conversations
- 当前评测用 conv_0 的 199 题，再除去 adversarial (47题) = **152 题**
- 四大题型：single-hop (70) / temporal (37) / open-domain (13) / multi-hop (32)

### Judge 判决 = LLM-as-judge 的二元判定

- 输入：问题 + 生成答案 + 标准答案
- 输出：`CORRECT` 或 `WRONG`
- 不给分数，只判对错

### 记忆库隔离

- 每个会话(conv_0, conv_1, ...) 有独立的 user_id
- Qdrant 查询显式过滤 `filters={"user_id": target_conv}`
- 防止会话间污染

## 环境和配置

### 启用 FastEmbed 缓存

```bash
export FASTEMBED_CACHE_PATH=$HOME/mem0-reproduction/models
```

### 禁用 Mem0 遥测

```bash
export MEM0_TELEMETRY=false
```

### 设置 HuggingFace 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

所有配置已在 `scripts/setup/env.sh` 中，运行：

```bash
source scripts/setup/env.sh
```

## 故障排查

| 问题 | 检查步骤 | 解决方案 |
|---|---|---|
| `Error: SGLang not responding` | `curl http://127.0.0.1:8000/v1/models` | 重启 SGLang: `cd model-serving && ./restart.sh` |
| `ModuleNotFoundError: No module named 'aiolimiter'` | 检查 venv | 激活: `source .benchmark-venv/bin/activate` |
| `0% accuracy (20% or less)` | 检查 runner 脚本 | **必须用 `run_locomo.py`，不要用 `run_locomo_bounded.py`** |
| 记忆库为空 | `ls benchmark-data/qdrant/collection/` | 运行 ingest: `scripts/eval/run_locomo.sh ingest` |
| 内存不足 | 监控 GPU 内存 | SGLang KV 缓存需 48GB（单卡）|

## 贡献指南

1. **添加新基准**：参考 `eval/benchmarks/locomo/` 目录结构
2. **修改提示词**：编辑 `eval/benchmarks/*/prompts.py`
3. **优化检索**：编辑 `eval/benchmarks/common/mem0_client.py`
4. **改进文档**：编辑 `docs/` 下的 Markdown 文件

所有改动请在 `scripts/eval/run_locomo.sh smoke` 上验证（快速反馈）。

---

**最后更新**: 2026-11-17  
**维护**: Kiro Agent  
**文档**: https://github.com/snap-research/locomo
