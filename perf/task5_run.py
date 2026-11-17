#!/usr/bin/env python3
"""Task5 native Mem0 retrieval matrix over isolated copies of source memories."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json_hash(value):
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _memory_text(item):
    return item.get("memory", item.get("text", ""))


def _read_source_records(client, collection, user_id):
    from qdrant_client import models
    records = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]),
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()))
            records.append({"id": str(point.id), "vector": vector, "payload": point.payload})
        if offset is None:
            return records


def _instrument_sdk(memory):
    spans = {}
    def instrument(obj, name, key):
        if obj is None or not hasattr(obj, name):
            return
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                spans[key] = spans.get(key, 0.0) + (time.perf_counter() - started) * 1000
        setattr(obj, name, wrapped)
    instrument(getattr(memory, "embedding_model", None), "embed", "embed_ms")
    instrument(getattr(memory, "vector_store", None), "search", "semantic_search_ms")
    instrument(getattr(memory, "vector_store", None), "keyword_search", "keyword_search_ms")
    instrument(memory, "_compute_entity_boosts", "entity_boost_ms")
    return spans


def _search(memory, query, top_k, user_id):
    started = time.perf_counter()
    response = memory.search(query, top_k=top_k, filters={"user_id": user_id}, rerank=False)
    elapsed = (time.perf_counter() - started) * 1000
    results = response.get("results", []) if isinstance(response, dict) else response
    memories = []
    for rank, item in enumerate(results, 1):
        memories.append({
            "id": str(item.get("id", item.get("memory_id", ""))),
            "text": _memory_text(item),
            "score": item.get("score"),
            "native_rank": rank,
        })
    return memories, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    task_dir = project_root / "task5"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(task_dir))
    from task5_core import assemble_block, conditions, fit_token_budget, limit_record_text, merge_supplement_results, supplement_query

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "trace.jsonl"
    if trace_path.exists() or (out / "DONE").exists():
        raise SystemExit("Refusing to overwrite existing Task5 run output")
    workload_path = Path(args.workload)
    workload_bytes = workload_path.read_bytes()
    workload = json.loads(workload_bytes)
    questions = workload["questions"]
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    os.environ["FASTEMBED_CACHE_PATH"] = str(project_root / "models")
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from tracer.config import get_config
    from tracer.observability_v4 import MetricsSampler
    from run_trace_v4 import tokenize
    from qdrant_client import QdrantClient, models
    from mem0 import Memory
    from tokenizers import Tokenizer
    import requests

    cfg = get_config(args.platform)
    if cfg.qdrant_mode == "server":
        source = QdrantClient(url=cfg.qdrant_url)
    else:
        source = QdrantClient(path=cfg.qdrant_path)
    try:
        source_records = _read_source_records(source, cfg.collection, cfg.user_id)
    finally:
        source.close()
    if len(source_records) != 879:
        raise RuntimeError("Expected 879 source records, found %d" % len(source_records))
    source_records_hash = _json_hash(source_records)
    dataset = Path(cfg.dataset_path)
    dataset_hash = _sha256(dataset.read_bytes())
    if workload.get('dataset_sha256') != dataset_hash:
        raise RuntimeError('Workload dataset hash mismatch')
    tokenizer = Tokenizer.from_file(str(project_root / "model-serving/experiment/tokenizer/tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

    # Every experiment point is a private, exact copy; the source store stays read-only.
    isolated_name = "t5_%s_879" % args.nonce.replace("-", "_")
    if cfg.qdrant_mode == "server":
        isolated_client = QdrantClient(url=cfg.qdrant_url)
    else:
        isolated_client = QdrantClient(path=str(out / "qdrant_embedded"))
    if isolated_client.collection_exists(isolated_name):
        isolated_client.close()
        raise RuntimeError("Refusing to reuse existing collection " + isolated_name)
    isolated_client.create_collection(
        isolated_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    for start in range(0, len(source_records), 100):
        isolated_client.upsert(isolated_name, [models.PointStruct(**r) for r in source_records[start:start + 100]])

    memory_config = {
        "history_db_path": str(out / "history.db"),
        "llm": {"provider": "openai", "config": {"model": cfg.model, "openai_base_url": cfg.sglang_url + "/v1", "api_key": "unused-local"}},
        "embedder": {"provider": "fastembed", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "qdrant", "config": {"client": isolated_client, "collection_name": isolated_name, "embedding_model_dims": 384, "path": str(out / "unused")}},
    }
    memory = Memory.from_config(memory_config)
    sdk_spans = _instrument_sdk(memory)
    collection_payload = [{"id": r["id"], "vector": r["vector"], "payload": r["payload"]} for r in source_records]
    manifest = {
        "schema_version": "task5-native-1.0",
        "platform": args.platform,
        "nonce": args.nonce,
        "workload_sha256": _sha256(workload_bytes),
        "dataset_sha256": dataset_hash,
        "source_records_sha256": source_records_hash,
        "source_record_count": len(source_records),
        "isolated_collection": isolated_name,
        "isolated_records_sha256": _json_hash(collection_payload),
        "source_collection": cfg.collection,
        "model": cfg.model,
        "source_versions": {name: (importlib.metadata.version(name) if _has_dist(name) else None) for name in ("mem0ai", "qdrant-client", "tokenizers", "fastembed")},
        "expected_questions": len(questions),
        "conditions": conditions(),
        "retrieval_policy": "Mem0 SDK native search, rerank=False; question-only supplement keyword expansion; exact-text stable merge",
        "supplement_rule": "Extract question tokens, remove leading interrogatives and common stop words, preserve entity/name tokens and order; no answer, evidence or category input.",
        "output_policy": {"temperature": 0, "top_p": 1, "top_k": -1, "max_new_tokens": 128, "min_new_tokens": 0, "ignore_eos": False},
        "context_truncation": "rank-ordered labeled memory block, exact tokenizer token prefix, re-encode after decode",
        "cache_policy": "unique leading nonce+job index per request, matched across hosts; no cache flush, cached_tokens measured",
        "source_store": "879 records copied read-only to an isolated collection for this run",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    server_info = requests.get(cfg.sglang_url + "/get_server_info", timeout=20).json()
    manifest["server_info"] = server_info
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    jobs = [(q, condition) for q in questions for condition in conditions()]
    random.Random(42).shuffle(jobs)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    manifest["expected_requests"] = len(jobs)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    # Initialization and one search warmup are outside request measurements.
    _search(memory, questions[0]['question'], 10, cfg.user_id)
    sdk_spans.clear()
    manifest['warmup_policy'] = 'one native K10 query outside measured requests; no LLM warmup or cache flush'
    (out / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sampler = MetricsSampler(cfg.sglang_url, out / "metrics_samples.jsonl", interval=0.2)
    sampler.start()
    try:
        with trace_path.open("w", encoding="utf-8") as trace:
            for index, (question_row, condition) in enumerate(jobs):
                question = question_row["question"]
                request_id = "%s:%d" % (args.nonce, index)
                sdk_spans.clear()
                request_started = time.perf_counter()
                wall_start = time.time()
                retrieval_calls = []
                retrieval_started = time.perf_counter()
                first, first_ms = _search(memory, question, 10 if condition["id"] == "k10_b512" else 20 if condition["id"] in ("b128", "b512", "len24") else condition["top_k"], cfg.user_id)
                retrieval_calls.append({"query": question, "top_k": 10 if condition["id"] == "k10_b512" else 20 if condition["id"] in ("b128", "b512", "len24") else condition["top_k"], "elapsed_ms": first_ms, "results": first})
                merge_ms = 0.0
                if condition["id"] == "supplement":
                    expanded_query = supplement_query(question)
                    second, second_ms = _search(memory, expanded_query, 10, cfg.user_id)
                    retrieval_calls.append({"query": expanded_query, "top_k": 10, "elapsed_ms": second_ms, "results": second})
                    merge_start = time.perf_counter()
                    selected = merge_supplement_results(first, second, limit=10)
                    merge_ms = (time.perf_counter() - merge_start) * 1000
                else:
                    selected = first[:condition["top_k"]]
                retrieval_total_ms = (time.perf_counter() - retrieval_started) * 1000
                assembly_started = time.perf_counter()
                if condition["per_record_tokens"] is not None:
                    selected = limit_record_text(selected, condition["per_record_tokens"], encode, decode)
                block_before = assemble_block(selected)
                block_before_ids = encode(block_before)
                block, block_ids = fit_token_budget(block_before, condition["block_budget"], encode, decode)
                messages = [
                    {"role": "system", "content": "%s request %s. Answer the question using supplied context when available. Be concise." % (args.nonce, index)},
                    {"role": "user", "content": "Context:\n%s\n\nQuestion: %s\nAnswer:" % (block, question)},
                ]
                injected = []
                cursor = 0
                for rank, memory_row in enumerate(selected, 1):
                    label = "[Memory %d] " % rank
                    start = block.find(label, cursor)
                    if start < 0:
                        continue
                    text_start = start + len(label)
                    end = block.find("\n[Memory ", text_start)
                    if end < 0:
                        end = len(block)
                    injected.append({"id": memory_row["id"], "rank": rank, "text": block[text_start:end], "char_start": text_start, "char_end": end, "token_offset_in_context": len(encode(block[:text_start]))})
                    cursor = end
                assembly_ms = (time.perf_counter()-assembly_started)*1000
                tokenize_started = time.perf_counter()
                input_ids = tokenize(cfg.sglang_url, cfg.model, messages=messages)
                prompt_tokenize_ms = (time.perf_counter() - tokenize_started) * 1000
                if len(input_ids) + 128 > server_info["context_length"]:
                    raise RuntimeError("Input plus output cap exceeds server context length")
                payload = {"rid": request_id, "input_ids": input_ids, "sampling_params": manifest["output_policy"], "stream": True}
                response_started = time.perf_counter()
                first_token = None
                native_meta = {}
                answer = ""
                with requests.post(cfg.sglang_url + "/generate", json=payload, stream=True, timeout=300) as response:
                    response.raise_for_status()
                    for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
                        if not raw or not raw.startswith("data: "):
                            continue
                        if raw[6:].strip() == "[DONE]":
                            break
                        chunk = json.loads(raw[6:])
                        now = time.perf_counter()
                        if chunk.get("text") and first_token is None:
                            first_token = now
                        answer = chunk.get("text", "")
                        native_meta = chunk.get("meta_info") or native_meta
                answer_end = time.perf_counter()
                if native_meta.get("completion_tokens", 0) > 128:
                    raise RuntimeError("Server returned more than 128 completion tokens")
                timings = {
                    "assembly_ms": assembly_ms,
                    "retrieval_total_ms": retrieval_total_ms,
                    "retrieval_calls_ms": [call["elapsed_ms"] for call in retrieval_calls],
                    "supplement_merge_ms": merge_ms,
                    "sdk_embed_ms": sdk_spans.get("embed_ms", 0.0),
                    "sdk_semantic_search_ms": sdk_spans.get("semantic_search_ms", 0.0),
                    "sdk_keyword_search_ms": sdk_spans.get("keyword_search_ms", 0.0),
                    "sdk_entity_boost_ms": sdk_spans.get("entity_boost_ms", 0.0),
                    "sdk_other_ms": None,
                    "prompt_tokenize_ms": prompt_tokenize_ms,
                    "server_prefill_elapsed_ms": ((native_meta["prefill_finished_time"] - native_meta["forward_entry_time"]) * 1000 if "prefill_finished_time" in native_meta and "forward_entry_time" in native_meta else None),
                    "server_queue_ms": native_meta.get("queue_time") * 1000 if native_meta.get("queue_time") is not None else None,
                    "ttft_ms": (first_token - response_started) * 1000 if first_token else None,
                    "decode_ms": (answer_end - first_token) * 1000 if first_token else None,
                    "generate_total_ms": (answer_end - response_started) * 1000,
                    "response_ms": (answer_end - request_started) * 1000,
                }
                row = {
                    "schema_version": "task5-native-1.0", "request_id": request_id, "platform": args.platform,
                    "condition": condition, "condition_id": condition["id"], "source_q_idx": question_row["source_q_idx"], "repetition": 0,
                    "question": question, "gold": question_row["answer"], "answer": answer, "category": question_row["category"],
                    "messages": messages, "retrieval_calls": retrieval_calls, "query_count": len(retrieval_calls),
                    "memories": selected, "injected_memories": injected,
                    "tokens": {"prompt": len(input_ids), "context_before_budget": len(block_before_ids), "context_actual": len(block_ids), "output": native_meta.get("completion_tokens"), "cached": native_meta.get("cached_tokens"), "server_decode_tps": native_meta.get("decode_throughput")},
                    "input_sha256": _json_hash(input_ids), "timing": timings, "native_meta": native_meta,
                    "wall_start": wall_start, "answer_end": time.time(),
                }
                trace.write(json.dumps(row, ensure_ascii=False) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
                print(json.dumps({"request": index + 1, "total": len(jobs), "condition": condition["id"], "retrieval_ms": retrieval_total_ms, "context_tokens": len(block_ids)}, ensure_ascii=False), flush=True)
        (out / "DONE").write_text("%d completed\n" % len(jobs))
    finally:
        sampler.stop()
        isolated_client.close()


def _has_dist(name):
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
