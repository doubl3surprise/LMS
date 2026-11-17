#!/usr/bin/env python3
"""Bounded SGLang GPU trace collection and honest hardware-counter discovery."""

import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

PLATFORMS = ("dgx", "5880ada")
PROMPT = "Reply with one short greeting."
GPU_WORKDIR = "/root/.cache/trace_v4_profiles"


def parse_traces(paths):
    """Read Kineto Chrome traces. Timestamps/durations are microseconds.

    displayTimeUnit controls the viewer, not the Chrome trace timestamp unit.
    TP workers may all call their device 0, so retain rank and process identity.
    Identical artifact copies are deduplicated without removing repeats in one file.
    """
    result = {"status": "success", "errors": [], "files": [], "gpus": [],
              "limitations": [], "duplicate_events": 0}
    groups = defaultdict(list)
    seen = Counter()
    for path in sorted(set(map(Path, paths))):
        record = {"path": str(path), "status": "success", "kernel_events": 0}
        result["files"].append(record)
        try:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as source:
                trace = json.load(source)
            events = trace if isinstance(trace, list) else trace.get("traceEvents")
            if not isinstance(events, list):
                raise ValueError("missing traceEvents list")
            info = trace if isinstance(trace, dict) else {}
            rank = info.get("distributedInfo", {}).get("rank", info.get("rank"))
            match = re.search(r"(?:^|[_.-])(?:tp_?rank|rank)[_-]?(\d+)(?:[_.-]|$)", path.name)
            if rank is None and match:
                rank = int(match.group(1))
            local = Counter()
            for event in events:
                if not isinstance(event, dict):
                    continue
                categories = {c.strip().lower() for c in str(event.get("cat", "")).split(",")}
                if not categories.intersection({"kernel", "cuda_kernel", "gpu_kernel", "cuda_concurrent_kernel"}):
                    continue
                if event.get("ph") != "X":
                    result["limitations"].append("Non-complete GPU kernel events are unsupported (expected ph=X).")
                    continue
                args = event.get("args") or {}
                start, duration = float(event["ts"]), float(event["dur"])
                if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                    raise ValueError("invalid kernel timestamp/duration")
                if duration == 0:
                    continue
                device = args.get("device", args.get("device id", args.get("Device Id")))
                event_rank = args.get("rank", args.get("tp_rank", rank))
                pid = event.get("pid")
                scope = str(path) if device is None or pid is None else None
                group = (device, event_rank, pid, scope)
                fingerprint = (scope, json.dumps([device, event_rank, pid, event.get("tid"),
                               event.get("name"), start, duration, args], sort_keys=True))
                local[fingerprint] += 1
                record["kernel_events"] += 1
                if local[fingerprint] <= seen[fingerprint]:
                    result["duplicate_events"] += 1
                    continue
                groups[group].append((start, start + duration))
                if device is None:
                    result["limitations"].append("GPU kernel device identity is absent.")
                if event_rank is None:
                    result["limitations"].append("Rank is absent; process identity is retained without inferring TP rank.")
            seen |= local
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            record["status"] = "error"
            result["errors"].append(str(path) + ": " + str(exc))
    for (device, rank, pid, _scope), intervals in sorted(groups.items(), key=lambda item: str(item[0])):
        result["gpus"].append({"device": device, "rank": rank, "process": pid,
                               "kernel_count": len(intervals),
                               "kernel_sum_ms": sum(b - a for a, b in intervals) / 1000.0,
                               "gpu_busy_union_ms": interval_union_ms(intervals)})
    if not result["gpus"]:
        result["errors"].append("No GPU kernel events collected; empty/CPU-only traces are not a successful profile.")
        result["limitations"].append("CUPTI availability and CUDA Graph kernel visibility are unverified; graph replay may omit kernel events.")
    result["limitations"].append("GPU busy union is per device/rank/process; it is not request latency or measured DRAM bandwidth.")
    result["limitations"] = sorted(set(result["limitations"]))
    if result["errors"]:
        result["status"] = "error"
    return result


def check_idle_metrics(text):
    """Parse Prometheus samples; malformed/incomplete data fails closed."""
    samples = []
    errors = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$", line)
        if not match:
            errors.append("invalid Prometheus sample at line %d" % number)
            continue
        name, label_text, raw_value = match.groups()
        try:
            value = float(raw_value)
            if not math.isfinite(value):
                if name in {'sglang:num_queue_reqs','sglang:num_running_reqs'}:
                    raise ValueError('non-finite queue/running gauge')
                continue
        except ValueError:
            errors.append("invalid metric value at line %d" % number)
            continue
        labels = {}
        if label_text:
            parts, current, quoted, escaped = [], [], False, False
            for char in label_text:
                if escaped:
                    current.extend(("\\", char))
                    escaped = False
                elif char == "\\" and quoted:
                    escaped = True
                elif char == '"':
                    quoted = not quoted
                    current.append(char)
                elif char == "," and not quoted:
                    parts.append("".join(current))
                    current = []
                else:
                    current.append(char)
            parts.append("".join(current))
            if quoted or escaped:
                errors.append("invalid escaped label at line %d" % number)
                continue
            valid = True
            for part in parts:
                lm = re.fullmatch(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"\s*', part)
                if not lm:
                    valid = False
                    break
                try:
                    labels[lm.group(1)] = json.loads('"' + lm.group(2) + '"')
                except ValueError:
                    valid = False
                    break
            if not valid:
                errors.append("invalid metric label at line %d" % number)
                continue
        samples.append({"name": name, "labels": labels, "value": value})
    queues = [s for s in samples if s['name'] in {'sglang:num_queue_reqs','sglang_num_queue_reqs'}]
    running = [s for s in samples if s['name'] in {'sglang:num_running_reqs','sglang_num_running_reqs'}]
    relevant = queues + running
    idle = bool(queues) and bool(running) and not errors and all(s["value"] == 0 for s in relevant)
    return {"idle": idle, "samples": relevant, "errors": errors,
            "reason": None if idle else "queue/running metrics missing, non-zero, malformed, or incomplete"}


def interval_union_ms(intervals):
    """Union intervals in Chrome trace microseconds; report milliseconds."""
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        total += stop - max(start, end if end is not None else start) if end is None or stop > end else 0
        end = max(stop, end) if end is not None else stop
    return total / 1000.0


def _http_json(url, payload=None, timeout=10):
    from urllib.request import Request, urlopen
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"} if body else {})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise RuntimeError("HTTP response exceeded 2 MB")
    try:
        return json.loads(raw.decode("utf-8")) if body and raw else raw.decode("utf-8", errors="replace")
    except json.JSONDecodeError:
        return raw.decode("utf-8", errors="replace")


def _run(command, timeout=30, output_limit=8000):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    output = (completed.stdout + completed.stderr)[-output_limit:]
    if completed.returncode:
        raise RuntimeError("command failed (%d): %s\n%s" % (completed.returncode, " ".join(command), output))
    return output


def _find_container(requested):
    if requested:
        return requested
    output = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"], timeout=15)
    candidates = [line.split("\t", 1)[0] for line in output.splitlines()
                  if "sglang" in line.lower() or "sglang" in line.split("\t")[-1].lower()]
    if len(candidates) != 1:
        raise RuntimeError("expected one running SGLang container, found %d; pass --container" % len(candidates))
    return candidates[0]


def _probe_counters(container, timeout=20):
    result = {"status": "not_measured", "tool": "ncu --query-metrics", "candidate_metrics": [], "output": ""}
    try:
        output = _run(["docker", "exec", container, "ncu", "--query-metrics"], timeout=timeout)
        candidates = sorted(set(re.findall(r"\b(?:dram__bytes_read|dram__bytes_write|dram__throughput|dram__bytes)\w*", output)))
        status='permission_denied' if 'ERR_NVGPUCTRPERM' in output else ('query_succeeded_candidates_only' if candidates else 'no_candidate_metrics_found')
        result.update({"status": status,
                       "candidate_metrics": candidates, "output": output[-4000:]})
    except Exception as exc:
        result.update({"status": "query_failed", "output": str(exc)[-4000:]})
    return result


def _discover_traces(container, remote_dir, timeout=30):
    command = ["docker", "exec", container, "sh", "-lc",
               "find %s -type f \\( -name '*.json' -o -name '*.json.gz' \\) -print" % remote_dir]
    return [line.strip() for line in _run(command, timeout=timeout, output_limit=20000).splitlines()
            if line.strip().endswith((".json", ".json.gz"))]


def collect(args):
    run_id = "trace-v4-%s-%s" % (args.platform, uuid.uuid4().hex[:12])
    container = None
    remote_dir = GPU_WORKDIR + "/" + run_id
    out_dir = Path(args.out_dir).expanduser().resolve()
    run_out = out_dir / "gpu_profile" / run_id
    run_out.mkdir(parents=True, exist_ok=False)
    result = {"status": "error", "errors": [], "platform": args.platform, "run_id": run_id,
              "container": None, "files": [], "gpus": [], "limitations": [],
              "bandwidth": {"status": "not_measured", "note": "ncu query is capability discovery only; no DRAM bandwidth sample was collected."}}
    profile_started = False
    try:
        container = _find_container(args.container)
        result["container"] = container
        if args.probe_counters:
            result["bandwidth"]["probe"] = _probe_counters(container)
        metrics_url = args.base_url.rstrip("/") + "/metrics"
        metrics = _http_json(metrics_url)
        idle = check_idle_metrics(metrics)
        result["queue_check"] = idle
        if not idle["idle"]:
            raise RuntimeError("refusing to profile a busy or unverifiable server: " + idle["reason"])
        _run(["docker", "exec", container, "mkdir", "-p", remote_dir])
        profile_id = uuid.uuid4().hex
        start_payload = {"output_dir": remote_dir, "activities": ["CPU", "GPU"],
                         "profile_by_stage": False, "profile_id": profile_id,
                         "merge_profiles": True, "profile_prefix": run_id}
        profile_started = True
        _http_json(args.base_url.rstrip("/") + "/start_profile", start_payload, timeout=15)
        request_payload = {"model": args.model, "messages": [{"role": "user", "content": PROMPT}],
                           "max_tokens": args.max_tokens,
                           "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
        response = _http_json(args.base_url.rstrip("/") + "/v1/chat/completions", request_payload,
                              timeout=args.request_timeout)
        result["request"] = {"status": "success", "usage": response.get("usage") if isinstance(response, dict) else None}
    except Exception as exc:
        result["errors"].append(str(exc))
    finally:
        if profile_started:
            try:
                _http_json(args.base_url.rstrip("/") + "/stop_profile", {"profile_id": profile_id}, timeout=20)
                profile_started = False
                result["profile_stopped"] = True
            except Exception as exc:
                profile_started = False
                result["profile_stopped"] = False
                result["errors"].append("stop_profile failed: " + str(exc))
        else:
            result["profile_stopped"] = False
        if container:
            try:
                remote_files = _discover_traces(container, remote_dir)
                for index, remote_path in enumerate(remote_files):
                    name = Path(remote_path).name
                    local = run_out / ("%03d_%s" % (index, name))
                    _run(["docker", "cp", container + ":" + remote_path, str(local)], timeout=args.copy_timeout)
                    result["files"].append({"path": str(local), "remote_path": remote_path,
                                            "status": "copied", "size_bytes": local.stat().st_size})
            except Exception as exc:
                result["errors"].append("artifact collection failed: " + str(exc))
    parsed = parse_traces([item["path"] for item in result["files"] if item.get("status") == "copied"])
    result["gpus"] = parsed["gpus"]
    result["limitations"] = parsed["limitations"]
    result["duplicate_events"] = parsed["duplicate_events"]
    result["files"] = parsed["files"] + result["files"]
    result["errors"].extend(parsed["errors"])
    result["bandwidth"]["status"] = "not_measured"
    result["status"] = "success" if not result["errors"] and bool(result["gpus"]) and result.get("profile_stopped") else "error"
    summary_path = run_out / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=PLATFORMS)
    parser.add_argument("--out-dir", default="./gpu_profile")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--model", default="qwen3.8-27b-zh")
    parser.add_argument("--container", help="Docker container; otherwise discover the unique running SGLang container")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--copy-timeout", type=int, default=90)
    parser.add_argument("--probe-counters", action="store_true", help="query ncu metric names only; does not measure bandwidth")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.max_tokens > 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    result, summary_path = collect(args)
    print(json.dumps({"status": result["status"], "summary": str(summary_path),
                      "errors": result["errors"], "gpus": result["gpus"],
                      "bandwidth": result["bandwidth"]}, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
