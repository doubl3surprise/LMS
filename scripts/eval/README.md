# 评测脚本说明

## 前置条件

| 条件 | 检查方式 |
|---|---|
| SGLang 推理服务 | `curl http://127.0.0.1:8000/v1/models` |
| Python 虚拟环境 | `ls ../../.benchmark-venv/bin/activate` |
| LoCoMo 数据集 | `ls ../../datasets/locomo/locomo10.json` |
| Mem0 记忆库 | `../../scripts/setup/check_env.sh` 或 `../../scripts/debug/store_inventory.py --user-id locomo_0_task1_canonical` |

## 命令速查

```bash
# 检查所有前置条件
./run_locomo.sh status

# 快速冒烟（10题，~5分钟）
./run_locomo.sh smoke

# 完整评测（152题，~72分钟）
./run_locomo.sh full

# 仅 ingestion（首次运行时，~2-3小时）
./run_locomo.sh ingest

# 展示最新结果
./run_locomo.sh report
```

## 评测流程说明

```
对话数据 (locomo10.json)
    ↓
Ingestion (Mem0 + Qwen)         # 对话 → 记忆抽取 → Qdrant 向量库
    ↓
对于每道问题 (152题)：
  1. 检索 (Mem0 search)         # 混合检索：密集向量 + BM25 + 实体
  2. 生成答案 (Qwen)            # top-10 记忆 + 问题 → 答案
  3. Judge 判决 (Qwen)         # 生成答案 vs 标准答案 → CORRECT/WRONG
    ↓
汇总 accuracy = CORRECT 数 / 152
```

## 结果文件

- 完整结果：`eval/results/locomo/locomo_results_YYYYMMDD_HHMMSS.json`
- 逐题结果：`eval/results/locomo/predicted_task1_canonical/conv0_qN.json`

## 评测题型

| 类型 | 题数 | 说明 | 当前精度 |
|---|---|---|---|
| single-hop | 70 | 一次检索即可回答 | 82.9% |
| temporal | 37 | 需要时间推理 | 89.2% |
| multi-hop | 32 | 需要跨多条记忆推理 | 68.8% |
| open-domain | 13 | 高度开放式 | 76.9% |
| **总计** | **152** | | **80.9%** |

## 注意事项

1. **必须使用 `run_locomo_fixed.py`**，不要用 `run_locomo_bounded.py`（后者有
   答案截断问题，会得到 ~20% 的虚假低精度）。
2. 脚本会自动激活 `../../.benchmark-venv` 虚拟环境，不需要手动 `source`。
3. 评测支持断点续跑：中断后加 `--resume` 或直接重跑同 `--run-id`。
