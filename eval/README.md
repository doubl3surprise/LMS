# eval/ — 评测框架

## 结构

```
eval/
├── run_locomo.py                ← LoCoMo-10 评测入口 ★
├── local_mem0_server.py         ← Mem0 HTTP 适配器
├── requirements.txt             ← 依赖
├── benchmarks/                  ← 核心评测代码
│   ├── common/                  ← 共享组件
│   │   ├── mem0_client.py       ← Mem0 客户端（search/add）
│   │   ├── llm_client.py        ← LLM 推理客户端
│   │   ├── metrics.py           ← 评分计算
│   │   ├── schema.py            ← 数据类型
│   │   └── utils.py             ← 工具函数
│   ├── locomo/                  ← LoCoMo-10 基准
│   │   ├── run.py               ← 评测框架主逻辑
│   │   └── prompts.py           ← 提示词模板 ★ 修改这里调 prompt
│   ├── longmemeval/             ← LongMemEval 基准
│   └── beam/                   ← BEAM 基准
└── results/locomo/              ← 评测结果输出
    ├── locomo_results_*.json    ← 完整结果 JSON
    └── predicted_task1_*/       ← 逐题结果
```

## 运行评测

**推荐方式**（使用封装脚本）:

```bash
../../scripts/eval/run_locomo.sh status   # 先看 SGLang / 适配器 / 记忆库 user_id
../../scripts/eval/run_locomo.sh ingest   # 首次必须：抽记忆（conv-0 419 chunks, ~1.5-2h）
../../scripts/eval/run_locomo.sh smoke    # 10题快速验证（依赖已有记忆库）
../../scripts/eval/run_locomo.sh full     # 152题完整评测
```

`ingest` / `smoke` / `full` 共用固定身份（`PROJECT_NAME=task1_canonical`、
`RUN_ID=task1_canonical`、user_id `locomo_0_task1_canonical`），因此 checkpoint 和记忆库
可以跨模式复用；换 project 名等于从头重新抽取 419 个 chunk。

**直接运行**（灵活控制参数）:

```bash
source ../.benchmark-venv/bin/activate
python3 run_locomo.py \
    --project-name "eval_$(date +%Y%m%d)" \
    --conversations 0 \
    --top-k 10 \
    --top-k-cutoffs 10 \
    --max-questions 152 \
    --answerer-model qwen3.8-27b-zh \
    --judge-model qwen3.8-27b-zh \
    --provider openai
```

## 结果格式

每次评测生成一个 JSON 文件，结构如下：

```json
{
  "metadata": {
    "answerer_model": "qwen3.8-27b-zh",
    "judge_model": "qwen3.8-27b-zh",
    "top_k": 10,
    "timestamp": "2026-11-16T21:26:30"
  },
  "metrics_by_cutoff": {
    "top_10": {
      "overall": {"accuracy": 80.9, "correct": 123, "total": 152},
      "by_category": {
        "temporal":    {"accuracy": 89.2, "correct": 33, "total": 37},
        "single-hop":  {"accuracy": 82.9, "correct": 58, "total": 70},
        "open-domain": {"accuracy": 76.9, "correct": 10, "total": 13},
        "multi-hop":   {"accuracy": 68.8, "correct": 22, "total": 32}
      }
    }
  },
  "evaluations": [
    {
      "question_id": "conv0_q0",
      "question": "...",
      "ground_truth_answer": "...",
      "cutoff_results": {
        "top_10": {
          "generated_answer": "...",
          "judgment": "CORRECT",
          "judge_reason": "..."
        }
      }
    }
  ]
}
```

## 题型编码

| 数字 | 名称 | 说明 |
|---|---|---|
| 1 | multi-hop | 需要跨多条记忆推理 |
| 2 | temporal | 需要时间推理 |
| 3 | open-domain | 高度开放式 |
| 4 | single-hop | 一次检索即可 |
| 5 | adversarial | 陷阱题（**不评测**）|

> conv_0 总共 199 题，去掉 adversarial(47题) 得到 **152 题**。

## 注意

- **必须用 `run_locomo.py`**，旧版已删除（max_tokens=64 截断问题）
- 评测依赖 SGLang 在 127.0.0.1:8000 运行，**且** Mem0 适配器在 127.0.0.1:8888 运行
- 推理虚拟环境是 `.benchmark-venv`，不是 `.venv`（后者给 perf/ 用）
- 数据集用绝对路径传入：`--dataset-path $HOME/mem0-reproduction/datasets/locomo/locomo10.json`
  （默认相对路径基于 CWD，缺文件时会尝试联网下载，而 DGX 无外网直连）
- `--predict-only` 产出的逐题文件只有 retrieval、没有 verdict；`--resume` 不会把它们当成
  已完成（`run.py` 已修），否则最终汇总会变成 0/N

---

**最后更新**: 2026-11-17
