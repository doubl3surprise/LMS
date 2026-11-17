"""Local HTTP adapter for the official memory-benchmarks OSS client."""
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path(__file__).resolve().parents[1] / "models"))

from mem0 import Memory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "database"
DATA.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "llm": {"provider": "openai", "config": {
        "model": os.getenv("MEM0_LLM_MODEL", "qwen3.8-27b-zh"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", "local-not-a-secret"),
        "temperature": 0, "max_tokens": 2048,
    }},
    "embedder": {"provider": "fastembed", "config": {
        "model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384,
    }},
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": os.getenv("MEM0_COLLECTION", "benchmark_memories"),
        "path": str(DATA / "qdrant"), "embedding_model_dims": 384,
    }},
    "history_db_path": str(DATA / "history.db"),
}

memory = Memory.from_config(CONFIG)
app = FastAPI(title="Local Mem0 benchmark adapter")

class AddRequest(BaseModel):
    messages: list[dict[str, Any]]
    user_id: str | None = None
    metadata: dict[str, Any] | None = None
    custom_instructions: str | None = None
    timestamp: int | None = None

class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    limit: int = 100
    rerank: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/memories")
def add_memories(req: AddRequest):
    if not req.user_id:
        raise HTTPException(400, "user_id is required")
    kwargs: dict[str, Any] = {"user_id": req.user_id, "infer": True}
    messages = req.messages
    if req.timestamp is not None:
        session_date = datetime.fromtimestamp(req.timestamp, timezone.utc).strftime("%Y-%m-%d")
        messages = [dict(message) for message in req.messages]
        if messages:
            messages[0]["content"] = f"[Conversation session date: {session_date}] " + str(messages[0].get("content", ""))
    if req.metadata:
        kwargs["metadata"] = req.metadata
    if req.custom_instructions:
        kwargs["prompt"] = req.custom_instructions
    try:
        return memory.add(messages, **kwargs)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@app.post("/search")
def search_memories(req: SearchRequest):
    if not req.user_id:
        raise HTTPException(400, "user_id is required")
    try:
        return memory.search(req.query, top_k=req.limit,
                             filters={"user_id": req.user_id}, rerank=req.rerank)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@app.get("/memories")
def list_memories(user_id: str | None = Query(None)):
    if not user_id:
        raise HTTPException(400, "user_id is required")
    try:
        return memory.get_all(filters={"user_id": user_id})
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@app.delete("/memories")
def delete_memories(user_id: str | None = Query(None)):
    if not user_id:
        raise HTTPException(400, "user_id is required")
    try:
        return memory.delete_all(user_id=user_id)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
