# Mem0 + SGLang 长期记忆系统研究

在 DGX Spark（GB10）上用 **Mem0 OSS 2.2.1 + SGLang（Qwen3.8-27B）** 复现长期记忆增强 LLM
系统的完整链路：历史对话 → 记忆抽取 → 向量/实体存储 → 检索 → 注入上下文 → 回答 → 评判，
并在 LoCoMo-10 上做质量与性能评测。

> **先读这两个文件**
> - `docs/ARCHITECTURE.md` — 系统架构、两条执行路径、与论文的差异、缓存实测、已知坑
> - `docs/flow.svg` — 一页流程图（写入路径 / 查询路径 / 共用组件）

---

## 快速开始

```bash
cd ~/mem0-reproduction

# 0) 环境自检（SGLang / 适配器 / venv / 数据集 / 记忆库 user_id）
scripts/setup/check_env.sh

# 1) 启动 Mem0 适配器（独占 Qdrant 文件锁，需要常驻）
mkdir -p logs
nohup scripts/server/start_mem0.sh > logs/mem0_adapter.log 2>&1 &

# 2) 写入记忆（conv-0：19 sessions → 419 个 chunk，约 1.5-2 小时，可断点续跑）
scripts/eval/run_locomo.sh ingest

# 3) 备份记忆库（database/ 不进 git，务必导出）
.benchmark-venv/bin/python scripts/debug/export_store.py export \
    --user-id locomo_0_task1_canonical --out database/exports/task1_canonical.json

# 4) 质量评测（先 10 题确认链路，再全量 152 题）
scripts/eval/run_locomo.sh smoke
scripts/eval/run_locomo.sh full

# 5) 查看结果 / 状态
scripts/eval/run_locomo.sh report
scripts/eval/run_locomo.sh status
```

固定 run 身份（`ingest` / `smoke` / `full` 共用，保证 checkpoint 与记忆库可复用）：

| 项 | 值 |
|---|---|
| project（output 目录） | `task1_canonical` → `eval/results/locomo/predicted_task1_canonical/` |
| run_id | `task1_canonical` |
| Qdrant user_id | `locomo_0_task1_canonical` |
| 覆盖方式 | `PROJECT_NAME=xxx RUN_ID=yyy scripts/eval/run_locomo.sh ...` |

---

## 目录（实际存在的内容）

```
mem0-reproduction/
├── README.md                       ← 本文件
├── docs/
│   ├── ARCHITECTURE.md             ← 系统架构与执行流程 ★
│   ├── flow.svg                    ← 流程图（可直接插报告）
│   └── diagnostic_analysis/        ← 失败根因分析（Exp A-G，7 份 JSON + 报告）
├── eval/                           ← 评测框架
│   ├── run_locomo.py               ← 入口：简洁直答 prompt + thinking=OFF（monkey-patch）
│   ├── local_mem0_server.py        ← Mem0 HTTP 适配器（:8888，/memories /search /stats /export）
│   ├── requirements.txt
│   ├── benchmarks/
│   │   ├── common/                 ← mem0_client / llm_client / metrics / schema / utils
│   │   ├── locomo/                 ← LoCoMo-10：run.py（ingest+search+answer+judge）+ prompts.py
│   │   ├── longmemeval/            ← 备用基准（仅代码骨架，未跑过）
│   │   └── beam/                   ← 备用基准（仅代码骨架，未跑过）
│   └── results/locomo/             ← 汇总 JSON + predicted_<project>/ 逐题 JSON
├── perf/                           ← 性能实验（已完成，作参考）
│   ├── project.py                  ← 统一入口 run/validate/profile/report
│   ├── tracer/                     ← 观测与追踪框架
│   ├── task3/ task4/ task5/ task6/
│   └── README.md
├── scripts/
│   ├── eval/run_locomo.sh          ← 一键评测（status/smoke/full/ingest/predict/report）
│   ├── server/start_mem0.sh        ← 启动适配器
│   ├── setup/{env.sh,check_env.sh} ← 环境变量 / 前置检查
│   └── debug/
│       ├── store_inventory.py      ← 按 user_id 盘点记忆库
│       ├── export_store.py         ← 记忆库导出/导入（备份恢复）
│       └── inspect_memories.py     ← 直接查记忆内容
├── database/                       ← 运行时状态（可重建，不进 git）
│   ├── qdrant/collection/{benchmark_memories, benchmark_memories_entities}
│   ├── history.db                  ← SQLite：history / messages
│   └── exports/                    ← 记忆库 JSON 备份
├── datasets/locomo/locomo10.json   ← 官方数据集（只读；SHA256 见下）
├── models/                         ← FastEmbed ONNX 缓存（88MB，含 all-MiniLM-L6-v2 与 bm25）
├── model-serving/                  ← SGLang Compose 部署（镜像固定、pull_policy: never）
├── trace_results/                  ← 性能实验输出（16 个 run 段 + task3_summary/report.md）
├── .benchmark-venv/                ← eval/ 使用
└── .venv/                          ← perf/ 使用
```

数据集 SHA256：`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`

---

## 质量基线（LoCoMo conv-0，152 题，cat 1-4）

| 题型 | 准确率 | 题数 |
|---|---|---|
| temporal | 89.2% | 33/37 |
| single-hop | 82.9% | 58/70 |
| open-domain | 76.9% | 10/13 |
| multi-hop | 68.8% | 22/32 |
| **总计** | **80.9%** | **123/152** |

- 结果文件：`eval/results/locomo/locomo_results_20261116_212630.json`
  （run_id `task1-dgx-sdk-20261116`，user_id `locomo_0_task1-dgx-sdk-20261116`，
  `chunk_size=1`、419 chunks、同模型 LLM-as-judge、`max_tokens=128`、thinking=OFF）。
- **注意**：该次运行的 Qdrant 数据在 11-17 的目录整理中丢失（当前库里只剩 11-09 的旧
  smoke 数据），所以它只能作为**历史基线**引用；要现场复现必须重新 ingest（见快速开始）。
  现行流程用固定 user_id + `/stats` 校验 + `export_store.py` 备份来防止再次发生。
- 评分口径是同模型二元判定（CORRECT/WRONG），不是官方 LoCoMo 的 EM/F1。

失败根因（`docs/diagnostic_analysis/`）：冗余不是瓶颈（>0.95 相似仅 1.0%）；
失败中约一半是检索没带回答案、一半是答案在 top-10 里但模型没推理出来；
余弦分数对正确/错误几乎无区分力（0.754 vs 0.743）。多跳是最弱项。

性能（历史测量，条件与口径见 `trace_results/task3_summary/report.md`）：同一 879 条
记忆、K=10 的原生检索路径，DGX 单卡 Response 3982 ms vs 4×RTX 5880 Ada 431 ms（约 9.2×），
DGX 上 decode 占端到端 83%；换成完整历史（约 1.51 万输入 token）后两平台都转为 prefill
主导（DGX prefill 11.5 s）。所以记忆规模/检索深度的最优取值随设备变化，不能按单卡比例外推。

---

## 环境

| 项 | 值 |
|---|---|
| 推理服务 | SGLang Docker `jiangchuan-qwen38-zh`，127.0.0.1:8000，模型 `qwen3.8-27b-zh` |
| 上下文 | `max_model_len=32768`，BF16，TP=1 |
| 记忆层 | mem0ai 2.2.1（OSS），FastEmbed `all-MiniLM-L6-v2`（384 维，CPU） |
| 向量库 | Qdrant 嵌入式（`database/qdrant`），集合 `benchmark_memories` + `_entities` |
| 历史库 | SQLite `database/history.db` |
| 路径隔离 | 一个集合 + `user_id` 过滤（`locomo_0_<run_id>`） |

环境变量（`scripts/setup/env.sh`）：`MEM0_TELEMETRY=false`、
`FASTEMBED_CACHE_PATH=$HOME/mem0-reproduction/models`、
`HF_ENDPOINT=https://hf-mirror.com`、`OPENAI_BASE_URL=http://127.0.0.1:8000/v1`。

启动/重启 SGLang：`cd model-serving && ./restart.sh`（Compose，镜像 `pull_policy: never`）。

---

## 常见任务

| 目标 | 做法 |
|---|---|
| 改回答 prompt | `eval/run_locomo.py` 的 `SIMPLE_SYSTEM` / `_simple_answer_prompt`；`eval/benchmarks/locomo/prompts.py` 是官方版参照 |
| 改评判逻辑 | `eval/benchmarks/locomo/run.py: judge_answer()`（DGX 的 SGLang 不支持 judge 请求带 `extra_body`） |
| 改检索参数 | `eval/run_locomo.py --top-k / --top-k-cutoffs`；检索本身在 Mem0 SDK 内（`mem0/memory/main.py: _search_vector_store`） |
| 改记忆抽取 | `eval/benchmarks/locomo/run.py: CHUNK_SIZE`；改抽取 prompt 需**重新 ingest** |
| 换集合/模型 | `eval/local_mem0_server.py` 顶部 `CONFIG`（`MEM0_COLLECTION` / `MEM0_LLM_MODEL`） |
| 看记忆库有什么 | `scripts/debug/store_inventory.py --user-id locomo_0_task1_canonical` |
| 备份/恢复记忆库 | `scripts/debug/export_store.py export|import`（import 前须停适配器） |
| 加新基准 | 复制 `eval/benchmarks/locomo/` 的 `run.py` + `prompts.py` 结构 |

---

## 排错

| 症状 | 原因 / 处理 |
|---|---|
| 准确率莫名很低、检索为空 | 记忆库里没有本次 `user_id` 的记忆。`run_locomo.sh status` 看 `target_user_count`，先 `ingest` |
| `RuntimeError: Storage folder ... already accessed by another instance` | 嵌入式 Qdrant 文件锁。别另开 `QdrantClient`；盘点/导出走适配器 HTTP，直接读写前先停适配器 |
| `Memory.add(timestamp=...)` 抛 Platform-only | OSS 不支持；适配器已改为在第一条消息里注入 `[Conversation session date: ...]` |
| 抽取返回 `{"results": []}` | 整个会话一次送入会撞 32k 上下文，模型静默返回空。当前按单轮 chunk 送 |
| 答案被截断成半句、judge 说 "incomplete" | 7 步扫描模板配了小 `max_tokens`。用 `eval/run_locomo.py`（简洁模板 + 128 token） |
| 适配器启动即失败 | 另一个适配器进程还活着（文件锁）。`pkill -f local_mem0_server` 后重启 |
| ingest 中途断了 | 直接重跑 `./run_locomo.sh ingest`，checkpoint `_progress_0.json` 会跳过已完成 chunk |
| 同一批记忆出现重复 | runner 的 HTTP 超时（`MEM0_HTTP_TIMEOUT`，默认 900s）小于一次抽取的最坏耗时。用 `scripts/debug/check_duplicates.py --user-id ...` 校验 |
| 停掉适配器后重启报 Qdrant 锁被占用 | 旧进程还活着（uvicorn 卡在处理中的请求）。确认无写入后 `kill -KILL <pid>`，再启动 |

---

## 关键概念

- **152 题** = LoCoMo-10 的 conv-0：199 题去掉 adversarial（47 题）后的 cat 1-4
  （single-hop 70 / temporal 37 / open-domain 13 / multi-hop 32）。
- **judge** = 同模型二元判定（问题 + gold + 生成答案 → CORRECT/WRONG），不给分数。
- **记忆隔离** = 单集合 + `filters={"user_id": ...}`，每个会话一个 user_id，防跨会话污染。

---

## 已知限制

- 质量数字是**同模型自评**的二元判定，不是官方指标，也没做人工复核。
- Embedding 是英文 `all-MiniLM-L6-v2`，**中文记忆检索质量未验证**。
- `payload.created_at` 是入库时间，不是历史事件时间；相对时间的锚点靠消息文本前缀。
- 抽取是 ADD-only，没有论文里的 UPDATE/DELETE/NOOP 决策环。
- 当前只跑 conv-0；`--conversations` 支持多会话，但每个会话都要各自完整 ingest。
- `longmemeval/`、`beam/` 只有代码骨架，未运行过。

---

**最后更新**：2026-11-17
