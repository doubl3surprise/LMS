"""
schema.py — TraceRecord 数据结构与 JSONL 序列化
================================================
v2 修订记录：
  - ExecutionTiming 增加 response_ms（embed→answer，不含 Judge）和 queue_ms
  - TokenCounts 增加 memory_tokens_actual（从 SGLang 实际 prompt_tokens 反推）
  - MemoryItem 增加 prompt_char_offset（在最终 prompt 中的起始字符位置）
  - TraceRecord 增加 evidence_sessions（题目涉及的 LoCoMo session 编号）
  - HardwareSnapshot 增加 kv_cache_tokens、disk_read_mb、disk_write_mb、queue_depth
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    """单条检索到的记忆的完整元数据。"""
    rank: int               # 1-based，MMR 排序后的位置
    memory_id: str          # Qdrant UUID
    score: float            # 原始向量相似度分（越高越相关）
    text: str               # 记忆文本（前 300 字符）
    text_len: int           # 完整文本字符数
    metadata: dict          # session、时间戳等附加字段
    prompt_char_offset: int | None = None  # 在最终 prompt 中的起始字符位置


@dataclass
class ExecutionTiming:
    """
    每道题各阶段耗时（ms）。

    字段说明：
      response_ms  — 从 embed 开始到 answer 生成完毕（不含 Judge）
                     这是用户视角的"真实响应时间"
      ttft_ms      — Time To First Token（客户端测量）
                     ≈ Prefill + 队列等待。串行无队列时 ≈ Prefill
      queue_ms     — 估算的队列等待时间（从 SGLang metrics 获取；
                     无法获取时为 0，意味着 ttft ≈ prefill）
      prefill_ms   — ttft_ms - queue_ms（真实 Prefill 时间估算）
      decode_ms    — 从首 token 到末 token（纯 Decode 时间）
      end_to_end_ms — embed → judge 结束（含 Judge，用于系统整体吞吐分析）
    """
    embed_ms: float
    search_ms: float
    mmr_ms: float
    assembly_ms: float
    ttft_ms: float
    queue_ms: float          # 队列等待（无法测量时填 0）
    prefill_ms: float        # = ttft_ms - queue_ms
    decode_ms: float
    generate_total_ms: float
    response_ms: float       # = embed→answer 完成，不含 Judge
    judge_ttft_ms: float
    judge_decode_ms: float
    judge_total_ms: float
    end_to_end_ms: float     # embed → judge 结束


@dataclass
class TokenCounts:
    """
    token 计数。

    来源说明：
      prompt_tokens          — SGLang streaming usage 实际值（最准确）
      completion_tokens      — SGLang streaming usage 实际值
      memory_tokens_actual   — prompt_tokens - system_tokens - question_tokens
                               system/question 由字符数/4 估算后修正
      memory_tokens_est      — 记忆文本 chars/4（快速估算，仅作对比）
    """
    memory_tokens_est: int           # 字符/4 估算（备用）
    question_tokens_est: int
    prompt_tokens: int | None        # SGLang 实际值
    completion_tokens: int | None
    memory_tokens_actual: int | None # prompt_tokens - system - question（SGLang来源）
    system_tokens_est: int           # system message chars/4
    decode_tps: float | None         # completion_tokens / decode_ms * 1000
    judge_prompt_tokens: int | None
    judge_completion_tokens: int | None


@dataclass
class HardwareSnapshot:
    """
    单次硬件快照（题目开始 / 结束 各一次）。

    新增字段：
      kv_cache_tokens   — SGLang /metrics 中的 kv cache 已用 token 数
      queue_depth       — SGLang /metrics 中的等待队列深度
      disk_read_mb      — 自进程启动以来累计磁盘读取量（MB）
      disk_write_mb     — 累计磁盘写入量（MB）
    """
    cpu_pct: float | None
    mem_used_gb: float | None
    mem_total_gb: float | None
    gpus: list[dict]           # [{gpu_index, util_pct, mem_used_mb, mem_total_mb}, ...]
    kv_cache_tokens: int | None = None
    queue_depth: int | None    = None
    disk_read_mb: float | None = None
    disk_write_mb: float | None = None


@dataclass
class TraceRecord:
    """一道题的完整 Trace 记录（v2）。"""
    # ── 标识 ──────────────────────────────────────────────────────────
    run_id: str
    platform: str
    q_idx: int
    category: int
    category_name: str
    timestamp: float
    wall_start: float
    wall_end: float

    # ── 内容 ──────────────────────────────────────────────────────────
    question: str
    gold: str
    answer: str
    correct: bool
    judge_raw: str
    evidence_sessions: list[int]   # 题目证据所在的 session 编号（从 LoCoMo evidence 字段解析）

    # ── 性能数据 ──────────────────────────────────────────────────────
    timing: ExecutionTiming
    tokens: TokenCounts
    memories: list[MemoryItem]     # 10 条检索记忆（含 prompt_char_offset）
    prompt_preview: str            # 组装后 prompt 的前 500 字符（调试用）

    # ── 硬件数据 ──────────────────────────────────────────────────────
    hw_before: HardwareSnapshot
    hw_after: HardwareSnapshot


# ── 序列化 / 反序列化 ─────────────────────────────────────────────────

def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


class TraceWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def write(self, record: TraceRecord):
        self._f.write(json.dumps(_to_dict(record), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_trace(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
