#!/usr/bin/env python3
"""Analyze schema v4 trace JSONL files into honest Markdown reports."""
import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

TIMINGS = [
    ("Embedding", "embed_ms"), ("Qdrant 搜索", "search_ms"),
    ("MMR 重排", "mmr_ms"), ("Prompt 组装", "assembly_ms"),
    ("客户端 TTFT", "client_ttft_ms"), ("TTFT", "ttft_ms"),
    ("服务端排队", "server_queue_ms"),
    ("服务端 Prefill", "server_prefill_elapsed_ms"),
    ("Decode", "decode_ms"), ("Generate 总计", "generate_total_ms"),
    ("Response", "response_ms"), ("Judge 总计", "judge_total_ms"),
    ("端到端", "end_to_end_ms"),
]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def _values(rows, group, key):
    result = []
    for row in rows:
        item = row.get(group) or {}
        value = item.get(key) if isinstance(item, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result.append(float(value))
    return result


def stats(values):
    values = sorted(values)
    if not values:
        return {"avg": None, "p50": None, "p95": None, "min": None, "max": None}
    def percentile(p):
        at = (len(values) - 1) * p
        lo, hi = int(at), min(int(at) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (at - lo)
    return {"avg": statistics.fmean(values), "p50": percentile(.5),
            "p95": percentile(.95), "min": values[0], "max": values[-1]}


def fmt(value, suffix=""):
    return "N/A" if value is None else f"{value:.1f}{suffix}"


def _engine(row):
    meta = row.get("native_meta") or {}
    if isinstance(meta, dict):
        return meta.get("engine") or meta.get("backend") or meta.get("server_version")
    return None


def _validate_v4(rows):
    if rows and not all(row.get("schema_version") == "4.0" for row in rows):
        raise ValueError("输入包含旧格式或非 v4 trace；请使用 legacy 分析器 analyze_trace.py")


def _session_summary(rows):
    sessions = defaultdict(set)
    sources = Counter()
    for row in rows:
        sid = row.get("request_session_id")
        source = row.get("session_source") or "unknown"
        sources[source] += 1
        if sid is not None:
            sessions[str(sid)].add(str(row.get("request_id", row.get("q_idx", "?"))))
    reused = {sid: len(reqs) for sid, reqs in sessions.items() if len(reqs) > 1}
    return sessions, sources, reused


def _resource_report(samples):
    lines = ["## 主机与 GPU 资源", "", "时间序列累计计数器取首末差；RSS 报峰值；缺失显示 N/A。", "",
             "| 组件 | CPU 累计差 user (s) | CPU 累计差 system (s) | RSS 峰值 (bytes) | read 差 (bytes) | write 差 (bytes) | rchar 差 | wchar 差 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    components = defaultdict(list)
    gpus = defaultdict(lambda: {"util": [], "mem": []})
    for sample in samples:
        for name, comp in (sample.get("components") or {}).items():
            if isinstance(comp, dict):
                components[name].append(comp)
        for gpu in sample.get("gpus") or []:
            idx = gpu.get("gpu_index", "?")
            for key, target in (("util_pct", "util"), ("mem_used_mb", "mem")):
                val = gpu.get(key)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    gpus[idx][target].append(float(val))
    counters = ("cpu_user_s", "cpu_system_s", "read_bytes", "write_bytes", "rchar", "wchar")
    for name, values in sorted(components.items()):
        values.sort(key=lambda x: x.get("timestamp", 0))
        deltas = []
        for key in counters:
            valid = [x[key] for x in values if isinstance(x.get(key), (int, float))]
            deltas.append(fmt(valid[-1] - valid[0]) if len(valid) >= 2 else "N/A")
        rss = [x["rss_bytes"] for x in values if isinstance(x.get("rss_bytes"), (int, float))]
        lines.append(f"| {name} | " + " | ".join(deltas[:2] + [fmt(max(rss)) if rss else "N/A"] + deltas[2:]) + " |")
    if not components:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines += ["", "| GPU | utilization avg | utilization peak | memory avg (MB) | memory peak (MB) |", "|---|---:|---:|---:|---:|"]
    if gpus:
        for idx, values in sorted(gpus.items(), key=lambda pair: str(pair[0])):
            util, mem = values["util"], values["mem"]
            lines.append(f"| GPU {idx} | {fmt(statistics.fmean(util) if util else None, '%')} | {fmt(max(util) if util else None, '%')} | {fmt(statistics.fmean(mem) if mem else None)} | {fmt(max(mem) if mem else None)} |")
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A |")
    lines += ["", "带宽数据缺失：未提供有效的带宽采样；I/O 字节计数差不等同于带宽。"]
    return lines


def _contention_report(data):
    lines = ["## 并发争用", ""]
    if not data:
        return lines + ["未提供 contention.json，未计算 slowdown。", ""]
    conditions = data.get("conditions", [])
    solos = [c for c in conditions if str(c.get("mode", "")).lower() in {"solo", "baseline", "control"}]
    loads = [c for c in conditions if c not in solos]
    lines += ["配对 slowdown 只在显式 pair_id 或同名同 repetitions 的 solo condition 上计算。", "",
              "| condition | mode | repetitions | LLM avg (ms) | paired slowdown | auxiliary ops | wall (s) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for cond in conditions:
        vals = [v for v in cond.get("llm_ms", []) if isinstance(v, (int, float))]
        avg = statistics.fmean(vals) if vals else None
        paired = None
        # Only pair explicit pair_id values, or same-name conditions with matched repetitions.
        for baseline in solos:
            same_pair = (cond.get("pair_id") is not None and cond.get("pair_id") == baseline.get("pair_id"))
            same_name = (cond.get("name") == baseline.get("name") and
                         cond.get("repetitions") == baseline.get("repetitions"))
            basevals = [v for v in baseline.get("llm_ms", []) if isinstance(v, (int, float))]
            if avg is not None and basevals and len(vals) == len(basevals) and (same_pair or same_name):
                paired = avg / statistics.fmean(basevals) if statistics.fmean(basevals) else None
                break
        lines.append(f"| {cond.get('name', '?')} | {cond.get('mode', '?')} | {cond.get('repetitions', 'N/A')} | {fmt(avg)} | {f'{paired:.3f}x' if paired is not None else '不可计算'} | {cond.get('auxiliary_ops', 'N/A')} | {fmt(cond.get('wall_s'))} |")
    for limitation in data.get("limitations", []):
        lines.append(f"\n限制：{limitation}")
    return lines + [""]


def render_report(rows, platform, resources=None, metrics=None, profile=None, contention=None):
    _validate_v4(rows)
    # Restrict coarse resource counters to the observed request window; model warmup
    # and later profiler/co-run experiments have separate files and boundaries.
    if rows and resources and all('wall_start' in r and 'wall_end' in r for r in rows):
        start=min(r.get('wall_start',float('inf')) for r in rows)
        end=max(r.get('wall_end',0) for r in rows)
        resources=[s for s in resources if start<=s.get('timestamp',-1)<=end]
    judged=[r for r in rows if isinstance(r.get('correct'),bool)]
    quality=f"{sum(r['correct'] for r in judged)}/{len(judged)} (LLM Judge, small selected subset)" if judged else 'not evaluated'
    lines = ["# Trace v4 分析报告", "", f"- Platform: `{platform}`",
             f"- Run ID: `{rows[0].get('run_id', 'N/A') if rows else 'N/A'}`",
             f"- Requests: {len(rows)}", "- Schema: `4.0`", f"- Quality: {quality}", "",
             "> Scope: instrumented FastEmbed + Qdrant/MMR + native SGLang, not full Mem0 SDK. Small characterization after template alignment; engine/runtime versions still differ. Server Prefill elapsed is scheduler wall time, not GPU kernel time. Profiling and co-run measurements are separate experiments; sampler overhead is not calibrated.", "",
             "## 延迟分解", "",
             "阶段统计基于该字段实际存在的请求；缺失不补零，分位数采用线性插值。客户端 TTFT、服务端队列与服务端 prefill elapsed 是不同测量边界，不可互相替代。", "",
             "| 阶段 | avg (ms) | p50 (ms) | p95 (ms) | 覆盖率 | ASCII avg bar |", "|---|---:|---:|---:|---:|---|"]
    timing_stats = {}
    e2e_avg = statistics.fmean(_values(rows, "timing", "end_to_end_ms")) if _values(rows, "timing", "end_to_end_ms") else None
    for label, key in TIMINGS:
        vals = _values(rows, "timing", key)
        s = timing_stats[key] = stats(vals)
        bar = "#" * min(20, round(20 * s["avg"] / max((e2e_avg or s["avg"]), 1))) if s["avg"] is not None else "N/A"
        lines.append(f"| {label} | {fmt(s['avg'])} | {fmt(s['p50'])} | {fmt(s['p95'])} | {len(vals)}/{len(rows)} | `{bar}` |")
    lines += ["", "## Tokens 与 session", "", "Token 值直取记录，不从字符数反推。", "",
              "| 字段 | avg | 覆盖率 | 方法/备注 |", "|---|---:|---:|---|"]
    for key in ("prompt_tokens", "completion_tokens", "memory_tokens_actual", "decode_tps"):
        vals = _values(rows, "tokens", key)
        method = next((r.get("tokens", {}).get("memory_token_method") for r in rows if r.get("tokens", {}).get("memory_token_method")), "N/A") if key == "memory_tokens_actual" else "trace value"
        lines.append(f"| {key} | {fmt(statistics.fmean(vals) if vals else None)} | {len(vals)}/{len(rows)} | {method} |")
    _, sources, reused = _session_summary(rows)
    memory_sessions = defaultdict(set)
    for row in rows:
        if row.get('request_session_id') is not None:
            for memory in row.get('memories', []):
                memory_sessions[memory['memory_id']].add((row.get('user_id'), row['request_session_id']))
    cross_count = sum(len(sessions)>1 for sessions in memory_sessions.values())
    lines += ["", f"跨会话访问的记忆：{cross_count} / {len(memory_sessions)}（同一 Memory ID 出现在多个 request_session_id）。",
              f"单会话内包含多个请求的 session 数：{len(reused)}。", "",
              "session_source 覆盖：" + (", ".join(f"{k}={v}" for k, v in sorted(sources.items())) or "N/A"),
              "synthetic session 不代表原始 LoCoMo 跨会话复用。", "", "## 记忆热点与 Prompt 位置", "",
              "| rank | memory_id | 访问次数 | avg score | min~max rank | avg prompt offset | text preview |", "|---:|---|---:|---:|---:|---:|---|"]
    memories = defaultdict(list)
    for row in rows:
        for mem in row.get("memories") or []:
            memories[str(mem.get("memory_id", "?"))].append(mem)
    for mid, items in sorted(memories.items(), key=lambda pair: (-len(pair[1]), pair[0]))[:20]:
        ranks = [m["rank"] for m in items if isinstance(m.get("rank"), (int, float))]
        scores = [m["score"] for m in items if isinstance(m.get("score"), (int, float))]
        offsets = [m["prompt_char_offset"] for m in items if isinstance(m.get("prompt_char_offset"), (int, float))]
        preview = str(items[0].get("text", "")).replace("|", "\\|").replace("\n", " ")[:70]
        rank_range = f"{min(ranks)}~{max(ranks)}" if ranks else "N/A"
        lines.append(f"| {min(ranks) if ranks else 'N/A'} | `{mid}` | {len(items)} | {fmt(statistics.fmean(scores) if scores else None)} | {rank_range} | {fmt(statistics.fmean(offsets) if offsets else None)} | {preview} |")
    if not memories:
        lines.append("| N/A | N/A | 0 | N/A | N/A | N/A | N/A |")
    lines += ["", "KV 按提供的 rank 独立展示；Mamba 等 backend 仍保留。logical slots 是逻辑容量，逻辑 slots 不跨 rank 相加。", ""]
    if profile:
        kv = profile.get("kv_cache", {}) if isinstance(profile, dict) else {}
        ranks = kv.get("ranks", []) if isinstance(kv, dict) else []
        lines += ["| rank | backend | logical slots | 其他字段 |", "|---:|---|---:|---|"]
        if ranks:
            for item in ranks:
                extras = ", ".join(f"{k}={v}" for k, v in item.items() if k not in {"rank", "backend", "logical_slots"})
                lines.append(f"| {item.get('rank', 'N/A')} | {item.get('backend', 'N/A')} | {item.get('logical_slots', 'N/A')} | {extras or 'N/A'} |")
            lines += ["", "KV min/max/avg by rank", "",
                      "| rank | backend | logical slots | metric | min | max | avg |",
                      "|---:|---|---:|---|---:|---:|---:|"]
            emitted = False
            for item in ranks:
                for name, value in item.items():
                    if not isinstance(value, dict) or not any(k in value for k in ("min", "max", "avg")):
                        continue
                    lines.append(f"| {item.get('rank', 'N/A')} | {item.get('backend', 'N/A')} | {item.get('logical_slots', 'N/A')} | {name} | {fmt(value.get('min'))} | {fmt(value.get('max'))} | {fmt(value.get('avg'))} |")
                    emitted = True
            if not emitted:
                lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append("| N/A | N/A | N/A | profiler KV 结构缺失 |")
    else:
        lines.append("Profiler KV/rank 数据缺失。")
    gpu_profile = profile.get("gpus", []) if isinstance(profile, dict) else []
    if isinstance(profile, dict):
        lines += ['', f"Profiler status: {profile.get('status','unknown')}; independent representative request, not per-request benchmark attribution."]
        for message in profile.get('errors', []) + profile.get('limitations', []):
            lines.append('> '+str(message))
        bandwidth=profile.get('bandwidth', {})
        if bandwidth:
            lines += ['', f"DRAM bandwidth: {bandwidth.get('status','not_measured')}. Counter probe is capability detection only."]
            probe=bandwidth.get('probe', {})
            if 'ERR_NVGPUCTRPERM' in probe.get('output',''):
                lines.append('> ERR_NVGPUCTRPERM: NVIDIA performance-counter permission is restricted; no bandwidth value was measured.')
    lines += ["", "### Profiler GPU 活跃区间", "",
              "| Device | Rank | Process | Kernels | GPU busy union (ms) | kernel duration sum (ms) |",
              "|---:|---:|---:|---:|---:|---:|"]
    if gpu_profile:
        for gpu in gpu_profile:
            lines.append(f"| {gpu.get('device', 'N/A')} | {gpu.get('rank', 'N/A')} | {gpu.get('process', 'N/A')} | {gpu.get('kernel_count', 'N/A')} | {fmt(gpu.get('gpu_busy_union_ms'))} | {fmt(gpu.get('kernel_sum_ms'))} |")
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
    lines += ["", "## Prometheus metrics", ""]
    if metrics:
        samples = [metric for sample in metrics for metric in sample.get("metrics", []) if any(k in metric.get('name','') for k in ['kv_cache_memory_usage_gb','kv_used_tokens','kv_available_tokens','kv_evictable_tokens','mamba_used_tokens','mamba_available_tokens','mamba_evictable_tokens','num_queue_reqs','num_running_reqs','weight_memory_usage_gb'])]
        lines += ["| metric | labels | avg | min | max | samples | scrape errors |", "|---|---|---:|---:|---:|---:|---:|"]
        by_metric = defaultdict(list)
        for item in samples:
            by_metric[(item.get("name", "?"), json.dumps(item.get("labels", {}), sort_keys=True, ensure_ascii=False))].append(item.get("value"))
        errors = sum(bool(sample.get("error")) for sample in metrics)
        for (name, labels), vals in sorted(by_metric.items()):
            valid = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
            lines.append(f"| {name} | `{labels}` | {fmt(statistics.fmean(valid) if valid else None)} | {fmt(min(valid) if valid else None)} | {fmt(max(valid) if valid else None)} | {len(valid)} | {errors} |")
    else:
        lines.append("未提供 metrics 样本。")
    lines += [""] + _resource_report(resources or []) + _contention_report(contention)
    lines += ["## 逐请求明细", "", "| request_id | q_idx | session | source | correct | TTFT ms | generate ms | response ms | prompt/completion/memory tokens | memory count | answer |", "|---|---:|---|---|---|---:|---:|---:|---|---:|---|"]
    for row in rows:
        t, tok = row.get("timing") or {}, row.get("tokens") or {}
        answer = str(row.get("answer") or "").replace("|", "\\|").replace("\n", " ")[:80]
        lines.append(f"| {row.get('request_id', 'N/A')} | {row.get('q_idx', 'N/A')} | {row.get('request_session_id', 'N/A')} | {row.get('session_source', 'N/A')} | {row.get('correct', 'N/A')} | {fmt(t.get('ttft_ms'))} | {fmt(t.get('generate_total_ms'))} | {fmt(t.get('response_ms'))} | {tok.get('prompt_tokens', 'N/A')}/{tok.get('completion_tokens', 'N/A')}/{tok.get('memory_tokens_actual', 'N/A')} | {len(row.get('memories') or [])} | {answer} |")
    lines += ["", "## 语义与覆盖率", "", "所有阶段覆盖率以非空有效记录数/请求数表示；N/A 代表未采集或不适用，不表示零。资源 counter 只对同组件首末有效样本求差，RSS 为采样峰值。GPU utilization 与显存使用量分别报告均值和峰值；profiler 的 GPU busy union 与 kernel duration sum 不合并。", ""]
    return "\n".join(lines)


def render_comparison(rows_a, platform_a, rows_b, platform_b):
    _validate_v4(rows_a)
    _validate_v4(rows_b)
    engines_a = sorted({e for e in map(_engine, rows_a) if e})
    engines_b = sorted({e for e in map(_engine, rows_b) if e})
    mismatch = bool(engines_a and engines_b and engines_a != engines_b)
    lines = ["# Trace v4 比较报告", "", "> 模板/权重已对齐；SGLang、CUDA、TP 和 Qdrant 部署仍不同。此表为不同平台软件栈的观测值，不是纯硬件受控对比。", ""]
    if mismatch:
        lines += [f"注意：engine 不同，{platform_a}={', '.join(engines_a)}，{platform_b}={', '.join(engines_b)}；延迟比较不是同 engine 控制实验。", ""]
    lines += ["| 阶段 | " + platform_a + " avg/p50/p95 ms | 覆盖率 | " + platform_b + " avg/p50/p95 ms | 覆盖率 |", "|---|---:|---:|---:|---:|"]
    for label, key in TIMINGS:
        cells = []
        for rows in (rows_a, rows_b):
            vals = _values(rows, "timing", key)
            s = stats(vals)
            cells += [f"{fmt(s['avg'])}/{fmt(s['p50'])}/{fmt(s['p95'])}", f"{len(vals)}/{len(rows)}"]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines += ["", "缺失数据保持 N/A；各平台测量覆盖率可能不同。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--platform", default="unknown")
    parser.add_argument("--out", default="report.md")
    parser.add_argument("--trace2")
    parser.add_argument("--platform2", default="unknown2")
    parser.add_argument("--resources")
    parser.add_argument("--metrics")
    parser.add_argument("--profile")
    parser.add_argument("--contention")
    args = parser.parse_args(argv)
    try:
        rows = load_jsonl(args.trace)
        _validate_v4(rows)
        if args.trace2:
            report = render_comparison(rows, args.platform, load_jsonl(args.trace2), args.platform2)
        else:
            read = lambda path: load_jsonl(path) if path else None
            profile = json.loads(Path(args.profile).read_text(encoding="utf-8")) if args.profile else None
            contention = json.loads(Path(args.contention).read_text(encoding="utf-8")) if args.contention else None
            report = render_report(rows, args.platform, read(args.resources), read(args.metrics), profile, contention)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
