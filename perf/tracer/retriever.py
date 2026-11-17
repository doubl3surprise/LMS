"""
retriever.py — 插桩式记忆检索（Embedding + Qdrant + MMR）
==========================================================
直接调用底层组件，绕过 Mem0 SDK，
这样每个子步骤的时间都可以独立计量。

返回带完整元数据的检索结果，供 schema.py 存入 Trace。
"""

import numpy as np
from dataclasses import dataclass
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .timer import PhasedTimer

# ── 单例：Embedding 模型（首次加载耗时，后续复用） ──────────────────
_EMBEDDER: TextEmbedding | None = None

def get_embedder(model: str = "sentence-transformers/all-MiniLM-L6-v2") -> TextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model)
    return _EMBEDDER


@dataclass
class RetrievedMemory:
    rank: int            # 最终排名（1-based）
    memory_id: str       # Qdrant point UUID
    score: float         # MMR 之前的原始相似度分
    text: str            # 记忆文本
    text_len: int        # 字符数
    metadata: dict       # Qdrant payload 中的其他字段


def retrieve(
    query: str,
    qc: QdrantClient,
    collection: str,
    user_id: str,
    cand_k: int = 50,
    final_k: int = 10,
    mmr_lambda: float = 0.5,
    timer: PhasedTimer | None = None,
) -> tuple[list[RetrievedMemory], np.ndarray]:
    """
    执行完整的检索流程，返回 (ranked_memories, query_vector)。

    计时阶段（写入 timer）：
      embed    — 把 query 转成 384-dim 向量
      search   — Qdrant 向量搜索 top-CAND_K
      mmr      — MMR 重排得到 top-FINAL_K
    """
    t = timer or PhasedTimer()
    embedder = get_embedder()
    flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

    # 1. Embedding
    with t.phase("embed"):
        q_vec = np.array(list(embedder.embed([query]))[0])

    # 2. Qdrant 向量搜索
    with t.phase("search"):
        hits = qc.query_points(
            collection_name=collection,
            query=q_vec.tolist(),
            query_filter=flt,
            limit=cand_k,
            with_payload=True,
            with_vectors=True,
        ).points

    if not hits:
        return [], q_vec

    # 3. MMR 重排
    with t.phase("mmr"):
        # 候选向量矩阵
        cand_vecs = np.array([h.vector for h in hits], dtype=np.float32)
        # 与 query 的余弦相似度
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        c_norms = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
        sim_q = c_norms @ q_norm  # shape: (cand_k,)

        selected_idx: list[int] = []
        remaining = list(range(len(hits)))

        while len(selected_idx) < min(final_k, len(hits)):
            if not selected_idx:
                best = int(np.argmax(sim_q[remaining]))
            else:
                sel_vecs = c_norms[selected_idx]
                # 对 remaining 中每个候选，计算与已选集合的最大相似度
                sim_sel = c_norms[remaining] @ sel_vecs.T  # (rem, sel)
                max_sim_sel = sim_sel.max(axis=1) if sim_sel.ndim > 1 else sim_sel
                mmr = mmr_lambda * sim_q[remaining] - (1 - mmr_lambda) * max_sim_sel
                best = int(np.argmax(mmr))
            # best 是 remaining 中的索引，需转回 hits 索引
            best_hit_idx = remaining[best]
            selected_idx.append(best_hit_idx)
            remaining.remove(best_hit_idx)

    # 4. 整理结果
    memories: list[RetrievedMemory] = []
    for rank, hi in enumerate(selected_idx, start=1):
        h = hits[hi]
        payload = h.payload or {}
        text = payload.get("memory", payload.get("data", ""))
        # metadata：除 memory/data/user_id 以外的其他字段
        meta = {k: v for k, v in payload.items() if k not in ("memory", "data", "user_id")}
        memories.append(RetrievedMemory(
            rank=rank,
            memory_id=str(h.id),
            score=float(h.score),
            text=text,
            text_len=len(text),
            metadata=meta,
        ))

    return memories, q_vec


MMR_LAMBDA = 0.5  # module-level default（也被 retrieve() 使用）
