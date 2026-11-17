#!/usr/bin/env python3
"""Bounded native Mem0/LLM overlap probe; run separately on each target host.

Example (target environment, not a remote launcher)::
    python3 task6_probe.py --platform dgx --out-dir trace_results/T6/dgx \
        --nonce matched-across-hosts --workload task5/workload.json

Default: 3 repeats x 4 modes = 12 trials. Each retrieval worker executes ten
closed-loop native SDK K10 searches, cycling the first four workload questions.
Each LLM worker submits one frozen input_ids prompt with exactly 32 output
tokens. Input IDs are identical across the three generation modes within a
repeat; their order-dependent actual cache hits are retained. These are
independent work makespans, not latency savings for a dependent QA request.

Artifacts: manifest.json (settings/provenance/prompts), trace.jsonl (one row
per trial), events.jsonl (all observed client spans), results.json (paired
comparisons), metrics_samples.jsonl (existing server sampler),
process_samples.jsonl (own process CPU/RSS/I/O), DONE (only after verification).
Unsupported bandwidth/kernel attribution stays explicitly unavailable.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from task5_run import _json_hash, _read_source_records, _search
from task6.probe_core import MODES, interval_union_ms, overlap_ms, run_mode, theoretical_makespan_ms, trial_schedule

SCHEMA = "task6-overlap-1.0"
SEMANTICS = "independent_work_makespan_not_same_question_latency"
OUTPUT_POLICY = {"temperature": 0, "top_p": 1, "top_k": -1,
                 "max_new_tokens": 32, "min_new_tokens": 32, "ignore_eos": True}


def write_json(path, value):
    """Atomically publish a completed JSON document within the run directory."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class EventRecorder:
    """One perf-counter clock plus a wall anchor; no engine clock subtraction."""
    def __init__(self):
        self.origin_ns = time.perf_counter_ns()
        self.wall_anchor = time.time()
        self.events = []
        self.lock = threading.Lock()
        self.trial_id = None
        self.call_index = None

    @contextmanager
    def span(self, name, **labels):
        event = {"trial_id": self.trial_id, "call_index": self.call_index,
                 "name": name, "thread_id": threading.get_ident(), **labels}
        event["start_ns"] = time.perf_counter_ns()
        try:
            yield event
        except BaseException as exc:
            event["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            event["end_ns"] = time.perf_counter_ns()
            event["start_ms"] = (event["start_ns"] - self.origin_ns) / 1e6
            event["end_ms"] = (event["end_ns"] - self.origin_ns) / 1e6
            event["duration_ms"] = (event["end_ns"] - event["start_ns"]) / 1e6
            with self.lock:
                self.events.append(event)


def instrument_sdk(memory, recorder):
    """Wrap real SDK boundaries without changing their results or search policy.

    Returns coverage plus a restoration callback. Absent stages are absent,
    never zero-valued synthetic spans. Module-level score fusion is scoped to
    this single-SDK-worker process and restored on exit.
    """
    restored = []
    coverage = {}
    targets = [(getattr(memory, "embedding_model", None), "embed", "sdk_embed"),
               (getattr(memory, "vector_store", None), "search", "sdk_semantic_search"),
               (getattr(memory, "vector_store", None), "keyword_search", "sdk_keyword_search"),
               (memory, "_compute_entity_boosts", "sdk_entity_boost")]
    module = sys.modules.get(type(memory).__module__)
    targets.extend((module, attr, name) for attr, name in
                   [("lemmatize_for_bm25", "sdk_lemmatize"), ("extract_entities", "sdk_entity_extract"),
                    ("score_and_rank", "sdk_score_and_rank")])
    for obj, attr, name in targets:
        original = getattr(obj, attr, None)
        coverage[name] = callable(original)
        if not callable(original):
            continue
        def wrap(fn, label):
            def wrapped(*args, **kwargs):
                with recorder.span(label):
                    return fn(*args, **kwargs)
            return wrapped
        restored.append((obj, attr, original))
        setattr(obj, attr, wrap(original, name))
    def restore():
        for obj, attr, original in reversed(restored):
            setattr(obj, attr, original)
    return coverage, restore


def run_trials(out, nonce, platform, questions, prompts, repeats, search, generate, recorder):
    """Dependency-injected local-testable measured loop; no fake benchmark mode.

    search(question) -> (native_memories, SDK_elapsed_ms)
    generate(input_ids, request_id) -> streaming-result dictionary
    Prompts have one entry per repeat and are prepared outside measured work.
    """
    out = Path(out)
    rows = []
    # Exclusive create also protects library callers from accidental reuse.
    with (out / "trace.jsonl").open("x", encoding="utf-8") as trace, (out / "events.jsonl").open("x", encoding="utf-8") as event_file:
        for ordinal, (repeat, mode) in enumerate(trial_schedule(repeats), 1):
            trial_id = "%s:r%d:%s" % (nonce, repeat, mode)
            recorder.trial_id = trial_id
            recorder.call_index = None
            first_event = len(recorder.events)
            prompt = prompts[repeat]
            wall_start = time.time()
            trial_start = time.perf_counter_ns()
            def retrieval():
                calls = []
                with recorder.span("retrieval_worker"):
                    for index in range(10):
                        question = questions[index % 4]
                        recorder.call_index = index
                        with recorder.span("sdk_search", query=question["question"], top_k=10):
                            memories, sdk_elapsed = search(question["question"])
                        calls.append({"call_index": index, "source_q_idx": question["source_q_idx"],
                                      "query": question["question"], "top_k": 10,
                                      "sdk_elapsed_ms": sdk_elapsed, "results": memories})
                    recorder.call_index = None
                return calls
            def llm():
                # SDK's call index is not meaningful on the independent LLM span.
                with recorder.span("llm_generate", call_index=None):
                    return generate(prompt["input_ids"], trial_id)
            try:
                output = run_mode(mode, retrieval, llm)
            except BaseException:
                for event in sorted(recorder.events[first_event:], key=lambda e: e["start_ns"]):
                    event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_file.flush()
                raise
            trial_end = time.perf_counter_ns()
            events = sorted(recorder.events[first_event:], key=lambda e: e["start_ns"])
            def intervals(name):
                return [(e["start_ms"], e["end_ms"]) for e in events if e["name"] == name]
            r, g, sdk = intervals("retrieval_worker"), intervals("llm_generate"), intervals("sdk_search")
            timing = {"total_makespan_ms": (trial_end - trial_start) / 1e6,
                      "retrieval_ms": interval_union_ms(r) if r else None,
                      "llm_ms": interval_union_ms(g) if g else None,
                      "sdk_search_union_ms": interval_union_ms(sdk) if sdk else None,
                      "retrieval_llm_overlap_ms": overlap_ms(sdk, g) if sdk and g else None,
                      "worker_overlap_ms": overlap_ms(r, g) if r and g else None}
            stage_names = sorted({e["name"] for e in events if e["name"].startswith("sdk_")})
            row = {"schema_version": SCHEMA, "trial_id": trial_id, "platform": platform,
                   "ordinal": ordinal, "repetition": repeat, "mode": mode,
                   "latency_semantics": SEMANTICS, "wall_start": wall_start,
                   "start_ns": trial_start, "end_ns": trial_end,
                   "timing": timing,
                   "sdk_stage_union_ms": {name: interval_union_ms(intervals(name)) for name in stage_names},
                   "query_count": len(output["retrieval"]) if output["retrieval"] is not None else 0,
                   "retrieval_calls": output["retrieval"], "llm": output["llm"],
                   "input_sha256": prompt["input_sha256"] if g else None,
                   "prompt_tokens": len(prompt["input_ids"]) if g else None}
            for event in events:
                event["trial_start_ms"] = (event["start_ns"] - trial_start) / 1e6
                event["trial_end_ms"] = (event["end_ns"] - trial_start) / 1e6
                event_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            event_file.flush()
            trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()
            os.fsync(trace.fileno())
            rows.append(row)
            print(json.dumps({"trial": ordinal, "total": repeats * len(MODES), "mode": mode, **timing}), flush=True)
    return rows


def summarize(rows):
    """Compare each repeat to its own isolated controls, with signed savings."""
    comparisons = []
    for repeat in sorted({r["repetition"] for r in rows}):
        selected = [r for r in rows if r["repetition"] == repeat]
        modes = {r["mode"]: r for r in selected}
        if len(selected) != len(MODES) or set(modes) != set(MODES):
            raise ValueError("Incomplete or duplicate trial modes for repeat %d" % repeat)
        r = modes["retrieval_only"]["timing"]["retrieval_ms"]
        g = modes["llm_only"]["timing"]["llm_ms"]
        serial = modes["serial"]["timing"]["total_makespan_ms"]
        concurrent = modes["concurrent"]["timing"]
        generation_hashes = {modes[m]["input_sha256"] for m in ("llm_only", "serial", "concurrent")}
        if len(generation_hashes) != 1:
            raise ValueError("Paired input hash mismatch")
        cached = {mode: modes[mode]["llm"]["tokens"].get("cached") for mode in ("llm_only", "serial", "concurrent")}
        cache_counts_match = len(set(cached.values())) == 1 if all(value is not None for value in cached.values()) else None
        comparisons.append({"repetition": repeat, "latency_semantics": SEMANTICS,
                            "input_sha256": generation_hashes.pop(),
                            "cache_tokens_by_mode": cached, "cache_counts_match": cache_counts_match,
                            "cache_caveat": "Matching token counts do not prove identical engine state; mismatched/missing counts confound LLM slowdown attribution",
                            "serial_makespan_ms": serial,
                            "concurrent_makespan_ms": concurrent["total_makespan_ms"],
                            "saved_ms": serial - concurrent["total_makespan_ms"],
                            "makespan_speedup": serial / concurrent["total_makespan_ms"],
                            "retrieval_slowdown_ratio": concurrent["retrieval_ms"] / r if r > 0 else None,
                            "llm_slowdown_ratio": concurrent["llm_ms"] / g if g > 0 else None,
                            "serial_retrieval_slowdown_ratio": modes["serial"]["timing"]["retrieval_ms"] / r if r > 0 else None,
                            "serial_llm_slowdown_ratio": modes["serial"]["timing"]["llm_ms"] / g if g > 0 else None,
                            "observed_sdk_llm_overlap_ms": concurrent["retrieval_llm_overlap_ms"],
                            "ideal_no_contention": theoretical_makespan_ms(r, g)})
    return comparisons


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def records_digest(records):
    """Stable source identity independent of Qdrant scroll ordering."""
    return _json_hash(sorted(records, key=lambda record: record["id"]))


def verify_clone(original, cloned):
    """Require exact IDs/payloads and dense vectors within float32 roundoff.

    COSINE stores may renormalize already normalized vectors on insert. Keep
    exact digests for both stores; a tolerance pass never means hash identity.
    The original store's before/after check remains strictly hash-equal.
    """
    import math
    a = {r["id"]: r for r in original}
    b = {r["id"]: r for r in cloned}
    if len(a) != len(original) or len(b) != len(cloned) or a.keys() != b.keys():
        raise RuntimeError("Clone point IDs/count mismatch")
    maximum = 0.0
    for identity in a:
        left, right = a[identity], b[identity]
        if left["payload"] != right["payload"]:
            raise RuntimeError("Clone payload differs for " + identity)
        if len(left["vector"]) != len(right["vector"]):
            raise RuntimeError("Clone vector dimension mismatch")
        for x, y in zip(left["vector"], right["vector"]):
            if not math.isfinite(x) or not math.isfinite(y) or not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError("Clone vector differs beyond float32 roundoff")
            maximum = max(maximum, abs(x - y))
    return {"sha256": records_digest(cloned), "equivalent": True, "record_count": len(cloned),
            "max_vector_abs_error": maximum, "vector_rel_tol": 1e-6, "vector_abs_tol": 1e-7,
            "exact_records": records_digest(original) == records_digest(cloned)}


def generate_stream(http, url, input_ids, request_id):
    """Native streaming transport; preserve observed cache and engine metadata."""
    payload = {"rid": request_id, "input_ids": input_ids,
               "sampling_params": dict(OUTPUT_POLICY), "stream": True}
    started = time.perf_counter_ns()
    first_token = None
    meta = {}
    answer = ""
    chunks = 0
    done = False
    with http.post(url + "/generate", json=payload, stream=True, timeout=(20, 300)) as response:
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError("Engine error: " + str(chunk["error"]))
            chunks += 1
            current = time.perf_counter_ns()
            if chunk.get("text") and first_token is None:
                first_token = current
            answer = chunk.get("text", answer)
            meta.update(chunk.get("meta_info") or {})
    ended = time.perf_counter_ns()
    if not done or not chunks or meta.get("completion_tokens") != 32:
        raise RuntimeError("Incomplete generation or output-token mismatch; expected exactly 32 tokens")
    if meta.get("prompt_tokens") is not None and meta["prompt_tokens"] != len(input_ids):
        raise RuntimeError("Engine prompt-token count differs from submitted input IDs")
    prefill = None
    if meta.get("forward_entry_time") is not None and meta.get("prefill_finished_time") is not None:
        prefill = (meta["prefill_finished_time"] - meta["forward_entry_time"]) * 1000
        if prefill < 0:
            raise RuntimeError("Engine returned reversed native prefill timestamps")
    return {"answer": answer, "native_meta": meta,
            "tokens": {"prompt": len(input_ids), "output": meta.get("completion_tokens"),
                       "cached": meta.get("cached_tokens"), "server_decode_tps": meta.get("decode_throughput")},
            "timing": {"client_generate_ms": (ended - started) / 1e6,
                       "ttft_ms": (first_token - started) / 1e6 if first_token is not None else None,
                       "client_stream_tail_ms": (ended - first_token) / 1e6 if first_token is not None else None,
                       "server_prefill_elapsed_ms": prefill,
                       "server_queue_ms": meta["queue_time"] * 1000 if meta.get("queue_time") is not None else None},
            "native_clock": "engine host wall timestamps; separate from client perf_counter_ns",
            "stream_chunks": chunks}


class ProcessSampler:
    """Continuous benchmark process CPU/RSS/I/O counters; no server attribution."""
    def __init__(self, url, path, interval=0.2):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.path = Path(path)
        self.interval = interval
        self.finished = threading.Event()
        self.thread = None
        self.ready = threading.Event()
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._sample, name="task6-process-sampler", daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=10):
            self.finished.set()
            raise RuntimeError("Process sampler did not produce its first sample")
        if self.error is not None:
            raise RuntimeError("Process sampler failed: " + str(self.error)) from self.error

    def _sample(self):
        try:
            self._sample_loop()
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()

    def _sample_loop(self):
        with self.path.open("x", encoding="utf-8") as file:
            while True:
                row = {"wall_time": time.time(), "perf_counter_ns": time.perf_counter_ns(),
                       "pid": os.getpid(), "source": "psutil benchmark process", "errors": {}}
                for key, getter in [("cpu_times_s", self.process.cpu_times), ("memory_bytes", self.process.memory_info),
                                    ("io_counters", self.process.io_counters if hasattr(self.process, "io_counters") else None)]:
                    try:
                        row[key] = getter()._asdict() if getter else None
                        if getter is None:
                            row["errors"][key] = "unsupported"
                    except (self.psutil.Error, OSError) as exc:
                        row[key] = None
                        row["errors"][key] = str(exc)
                file.write(json.dumps(row) + "\n")
                file.flush()
                self.ready.set()
                if self.finished.wait(self.interval):
                    break

    def stop(self):
        self.finished.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Process sampler did not stop")
        if self.error is not None:
            raise RuntimeError("Process sampler failed: " + str(self.error)) from self.error


def execute_probe(args, cfg, workload_bytes, read_source, clone_store, make_memory,
                  tokenize_prompt, generate, server_info, metrics_sampler, process_sampler):
    """Lifecycle shared by real target adapters and local contract tests.

    read_source() must open/scroll/close the original without writes.
    clone_store(records, unique_name) returns (client, verification dictionary).
    All mutation occurs in the unique clone and output directory.
    """
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing nonempty Task6 output directory: " + str(out))
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    workload = json.loads(workload_bytes)
    questions = [{"question": q["question"], "source_q_idx": q["source_q_idx"]}
                 for q in workload["questions"][:4]]
    if len(questions) != 4 or any(not isinstance(q["question"], str) or not q["question"].strip() for q in questions):
        raise ValueError("Expected at least four nonempty workload questions")
    dataset_hash = sha256(Path(cfg.dataset_path).read_bytes())
    if workload.get("dataset_sha256") != dataset_hash:
        raise RuntimeError("Workload dataset hash mismatch")
    records = read_source()
    if len(records) != 879 or len({r["id"] for r in records}) != 879:
        raise RuntimeError("Expected exactly 879 unique source records")
    before_hash = records_digest(records)
    out.mkdir(parents=True, exist_ok=True)
    # Atomic run ownership; a failed run retains this marker and cannot be reused.
    with (out / "STARTED").open("x", encoding="utf-8") as file:
        file.write(str(time.time()) + "\n")
    clone = None
    samplers = []
    restore = lambda: None
    recorder = EventRecorder()
    manifest = {"schema_version": SCHEMA, "platform": args.platform, "nonce": args.nonce,
                "latency_semantics": SEMANTICS, "repeats": args.repeats,
                "expected_trials": args.repeats * len(MODES), "seed": 42,
                "schedule": [{"repetition": rep, "mode": mode} for rep, mode in trial_schedule(args.repeats)],
                "questions": questions, "question_selection": "first four workload question rows; gold/evidence excluded",
                "workload_sha256": sha256(workload_bytes), "dataset_sha256": dataset_hash,
                "source_collection": cfg.collection, "source_user_id": cfg.user_id,
                "source_record_count": len(records), "source_records_sha256": before_hash,
                "source_digest_policy": "sort records by string point id; JSON includes vectors and payloads",
                "isolated_collection": "t6_" + args.nonce.replace("-", "_"),
                "retrieval_policy": "Mem0 SDK Memory.search(top_k=10, rerank=False), dense-only cloned store",
                "retrieval_load": "10 closed-loop calls cycling four questions; no pacing or parallel SDK calls",
                "output_policy": dict(OUTPUT_POLICY), "thinking": False, "model": cfg.model,
                "server_info": server_info, "qdrant_mode": cfg.qdrant_mode,
                "versions": {}, "prompts": [],
                "clock": {"client": "perf_counter_ns", "origin_ns": recorder.origin_ns, "wall_anchor": recorder.wall_anchor,
                          "wall_anchor_uncertainty": "non-atomic adjacent clock reads", "unit": "ms unless ns suffix"},
                "cache_policy": "same input IDs per repeat across llm_only/serial/concurrent; unique leading repeat nonce; order-dependent warm cache; no flush; actual cached_tokens saved",
                "warmup_policy": "one native K10 search; no LLM warmup; export/clone/init/tokenize outside measured trials",
                "limits": ["Independent retrieval and generation work; not same-question response latency or dependent multi-round QA",
                           "Client interval overlap does not establish concurrent GPU kernels or causal resource contention",
                           "No CPU/GPU/shared-controller memory-bandwidth counters; no fabricated payload IO stage",
                           "Process sampler covers benchmark only; external host sampler required for SGLang/Qdrant/device attribution",
                           "Three repetitions validate mechanism, not statistical significance; three blocks only partially order-balanced"]}
    for dist in ("mem0ai", "qdrant-client", "fastembed", "requests", "psutil", "sglang"):
        try:
            manifest["versions"][dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][dist] = None
    manifest["python"] = sys.version
    manifest["runner_sha256"] = sha256(Path(__file__).read_bytes())
    write_json(out / "manifest.json", manifest)
    try:
        clone, clone_verification = clone_store(records, manifest["isolated_collection"])
        manifest["isolated_records_sha256"] = clone_verification["sha256"]
        manifest["clone_verification"] = clone_verification
        if not clone_verification["equivalent"]:
            raise RuntimeError("Clone read-back verification failed")
        memory = make_memory(clone, out)
        manifest["sdk_instrumentation"], restore = instrument_sdk(memory, recorder)
        # K10 warmup also supplies the single frozen memory context for the LLM.
        warmup, warmup_ms = _search(memory, questions[0]["question"], 10, cfg.user_id)
        if not warmup:
            raise RuntimeError("Native SDK warmup returned no memories")
        manifest["warmup_retrieval"] = {"elapsed_ms": warmup_ms, "results": warmup}
        recorder.events.clear()
        block = "\n".join("[Memory %d] %s" % (i, item["text"]) for i, item in enumerate(warmup, 1))
        body = "Context:\n%s\n\nQuestion: %s\nAnswer:" % (block, questions[0]["question"])
        manifest["common_body_sha256"] = sha256(body.encode("utf-8"))
        manifest["frozen_context_text_sha256"] = _json_hash([item["text"] for item in warmup])
        for rep in range(args.repeats):
            messages = [{"role": "system", "content": "%s repeat %d. Answer using supplied memories. Be concise." % (args.nonce, rep)},
                        {"role": "user", "content": body}]
            input_ids = tokenize_prompt(messages)
            if not input_ids or not all(isinstance(token, int) and not isinstance(token, bool) for token in input_ids):
                raise RuntimeError("Tokenization did not return nonempty integer input IDs")
            if len(input_ids) + 32 > server_info["context_length"]:
                raise RuntimeError("Frozen input plus 32 output tokens exceeds context length")
            manifest["prompts"].append({"repetition": rep, "messages": messages, "input_ids": input_ids,
                                        "input_sha256": _json_hash(input_ids), "prompt_tokens": len(input_ids)})
        write_json(out / "manifest.json", manifest)
        for factory, filename in [(metrics_sampler, "metrics_samples.jsonl"), (process_sampler, "process_samples.jsonl")]:
            sampler = factory(cfg.sglang_url, out / filename, interval=0.2)
            samplers.append(sampler)
            sampler.start()
        rows = run_trials(out, args.nonce, args.platform, questions, manifest["prompts"], args.repeats,
                          lambda query: _search(memory, query, 10, cfg.user_id), generate, recorder)
        comparisons = summarize(rows)
        # Always re-read the original after measured work and before publishing DONE.
        after_hash = records_digest(read_source())
        manifest["source_records_after_sha256"] = after_hash
        manifest["source_unchanged"] = after_hash == before_hash
        write_json(out / "manifest.json", manifest)
        if not manifest["source_unchanged"]:
            raise RuntimeError("Source-store read-back hash changed during probe")
        stored = [json.loads(line) for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(stored) != args.repeats * len(MODES) or len({r["trial_id"] for r in stored}) != len(stored):
            raise RuntimeError("Trace trial count/identity verification failed")
        if summarize(stored) != comparisons:
            raise RuntimeError("Trace read-back comparisons changed")
        result = {"schema_version": SCHEMA, "platform": args.platform, "source_unchanged": True,
                  "trial_count": len(stored), "trials": stored, "comparisons": comparisons,
                  "latency_semantics": SEMANTICS, "limits": manifest["limits"]}
        write_json(out / "results.json", result)
        if json.loads((out / "results.json").read_text(encoding="utf-8"))["trial_count"] != len(stored):
            raise RuntimeError("Results read-back failed")
    except BaseException as exc:
        # An SDK/transport failure must not skip the read-only source audit.
        if "source_records_after_sha256" not in manifest:
            try:
                manifest["source_records_after_sha256"] = records_digest(read_source())
                manifest["source_unchanged"] = manifest["source_records_after_sha256"] == before_hash
            except BaseException as audit_error:
                manifest["source_unchanged"] = None
                manifest["source_audit_error"] = str(audit_error)
            write_json(out / "manifest.json", manifest)
        write_json(out / "FAILED.json", {"type": type(exc).__name__, "error": str(exc), "wall_time": time.time()})
        raise
    finally:
        # Attempt every cleanup even if a sampler fails. Preserve a primary
        # SDK/transport error while recording all cleanup failures separately.
        primary_failure = sys.exc_info()[0] is not None
        cleanup_errors = []
        for sampler in reversed(samplers):
            try:
                sampler.stop()
            except BaseException as exc:
                cleanup_errors.append(type(exc).__name__ + ": " + str(exc))
        try:
            restore()
        except BaseException as exc:
            cleanup_errors.append(type(exc).__name__ + ": " + str(exc))
        if clone is not None:
            try:
                clone.close()
            except BaseException as exc:
                cleanup_errors.append(type(exc).__name__ + ": " + str(exc))
        if cleanup_errors:
            write_json(out / "CLEANUP_FAILED.json", {"errors": cleanup_errors})
            if not primary_failure:
                write_json(out / "FAILED.json", {"type": "CleanupError", "errors": cleanup_errors})
                raise RuntimeError("Task6 cleanup failed: " + "; ".join(cleanup_errors))
    # Sampler threads are stopped now, so line reads cannot observe partial
    # writes. Refuse completion when a supposedly enabled collector is empty.
    sample_counts = {}
    try:
        for filename in ("metrics_samples.jsonl", "process_samples.jsonl"):
            path = out / filename
            if not path.exists():
                raise RuntimeError("Missing sampler artifact " + filename)
            count = 0
            with path.open(encoding="utf-8") as file:
                for line in file:
                    json.loads(line)
                    count += 1
            if not count:
                raise RuntimeError("Empty sampler artifact " + filename)
            sample_counts[filename] = count
        result["sample_counts"] = sample_counts
        write_json(out / "results.json", result)
    except BaseException as exc:
        write_json(out / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    with (out / "DONE").open("x", encoding="utf-8") as file:
        file.write("%d verified trials; source unchanged\n" % len(result["trials"]))
    return result


def build_target_adapters(cfg, out, isolated_name, QdrantClient, models, Memory):
    """Build target SDK adapters without opening stores until explicitly called."""
    def read_source():
        source = QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode == "server" else QdrantClient(path=cfg.qdrant_path)
        try:
            return _read_source_records(source, cfg.collection, cfg.user_id)
        finally:
            source.close()

    def clone_store(records, name):
        if name != isolated_name or name == cfg.collection or not name.startswith("t6_"):
            raise ValueError("Clone target must be this run's isolated collection")
        if any(not isinstance(r["vector"], list) or len(r["vector"]) != 384 for r in records):
            raise ValueError("Expected 384-dimensional dense-only source vectors")
        # Only one SDK retrieval call runs at a time, but modes move that
        # worker between main/background threads. Permit SQLite's thread move.
        client = QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_mode == "server" else QdrantClient(
            path=str(out / "qdrant_embedded"), force_disable_check_same_thread=True)
        try:
            if client.collection_exists(name):
                raise FileExistsError("Refusing existing isolated collection " + name)
            client.create_collection(name, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            for start in range(0, len(records), 100):
                client.upsert(name, [models.PointStruct(**r) for r in records[start:start + 100]])
            verified = verify_clone(records, _read_source_records(client, name, cfg.user_id))
            return client, verified
        except BaseException:
            client.close()
            raise

    def make_memory(client, run_dir):
        config = {"history_db_path": str(run_dir / "history.db"),
                  "llm": {"provider": "openai", "config": {"model": cfg.model, "openai_base_url": cfg.sglang_url + "/v1", "api_key": "unused-local"}},
                  "embedder": {"provider": "fastembed", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
                  "vector_store": {"provider": "qdrant", "config": {"client": client, "collection_name": isolated_name,
                                                                          "embedding_model_dims": 384, "path": str(run_dir / "unused")}}}
        return Memory.from_config(config)
    return {"read_source": read_source, "clone_store": clone_store, "make_memory": make_memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", required=True, choices=("dgx", "5880ada"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nonce", required=True, help="same safe token on both hosts; use a fresh token for each paired run")
    parser.add_argument("--workload", required=True, help="Task5 JSON workload; first four question rows are used")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.nonce):
        parser.error("--nonce must contain 1-80 letters, digits, underscores or hyphens")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error("Refusing nonempty Task6 output directory")
    root = Path(__file__).resolve().parent
    os.environ["FASTEMBED_CACHE_PATH"] = str(root / "models")
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    # Reuse the deployed project's adapters; importing this module for tests
    # requires none of these target-only dependencies.
    from tracer.config import get_config
    from tracer.observability_v4 import MetricsSampler
    from run_trace_v4 import tokenize
    from qdrant_client import QdrantClient, models
    from mem0 import Memory
    import requests

    cfg = get_config(args.platform)
    response = requests.get(cfg.sglang_url + "/get_server_info", timeout=20)
    response.raise_for_status()
    server_info = response.json()
    # The existing tokenize helper uses the server chat template defaults.
    # Fail closed when the live serving configuration does not confirm OFF.
    if (server_info.get("default_chat_template_kwargs") or {}).get("enable_thinking") is not False:
        raise RuntimeError("Live server must explicitly have default_chat_template_kwargs.enable_thinking=false")
    adapters = build_target_adapters(cfg, out, "t6_" + args.nonce.replace("-", "_"), QdrantClient, models, Memory)
    execute_probe(args, cfg, Path(args.workload).read_bytes(), **adapters,
                  tokenize_prompt=lambda messages: tokenize(cfg.sglang_url, cfg.model, messages=messages),
                  generate=lambda ids, rid: generate_stream(requests, cfg.sglang_url, ids, rid),
                  server_info=server_info, metrics_sampler=MetricsSampler, process_sampler=ProcessSampler)


if __name__ == "__main__":
    main()
