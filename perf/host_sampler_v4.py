"""Host-side process CPU/IO accounting for SGLang, Qdrant and benchmark.

CPU/IO fields are cumulative observed process-lifetime counters: include each
process's first observed lifetime total, then only increments; exited processes
remain in totals. RSS is a current sum. PID + create_time prevents PID reuse.
Sampling cannot recover activity after a process's last observation or processes
that start and exit entirely between polls. RSS can double-count shared pages.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import psutil

COUNTERS = ("cpu_user_s", "cpu_system_s", "read_bytes", "write_bytes", "rchar", "wchar")
COMPONENTS = ("sglang", "qdrant", "benchmark")


class ProcessAccumulator:
    """Deduplicate identities globally and accumulate deltas per component."""

    def __init__(self):
        self.previous = {}
        self.totals = {}

    def update(self, records, errors=None):
        errors = errors or {}
        names = set(COMPONENTS) | set(records) | set(self.totals)
        result = {}
        seen = set()
        for name in sorted(names):
            totals = self.totals.setdefault(name, {key: None for key in COUNTERS})
            rows = []
            for row in records.get(name, []):
                identity = (row["pid"], row["create_time"])
                if identity in seen:
                    continue
                seen.add(identity)
                previous = self.previous.setdefault(identity, {})
                for key in COUNTERS:
                    value = row.get(key)
                    if value is not None:
                        delta = max(0, value - previous.get(key, 0))
                        totals[key] = (totals[key] or 0) + delta
                        previous[key] = max(value, previous.get(key, 0))
                rows.append(row)
            issues = list(errors.get(name, []))
            issues.extend(error for row in rows for error in row.get("errors", []))
            if not rows:
                issues.append("No matching live processes observed")
            reported = {key: (None if (errors.get(name) and not rows) or
                any(row.get(key) is None for row in rows) else totals[key])
                for key in COUNTERS}
            result[name] = dict(reported, rss_bytes=(sum(row["rss_bytes"] for row in rows)
                if rows and all(row.get("rss_bytes") is not None for row in rows) else None),
                pids=sorted(row["pid"] for row in rows), errors=issues)
        return result


def read_process(pid):
    """Read own CPU (exclude children counters), RSS and Linux IO char counts."""
    row = dict.fromkeys((*COUNTERS, "rss_bytes", "create_time"))
    row.update(pid=pid, errors=[])
    try:
        proc = psutil.Process(pid)
        row["create_time"] = proc.create_time()
    except (psutil.Error, OSError) as exc:
        row["errors"].append(f"pid {pid} identity: {type(exc).__name__}: {exc}")
        return row
    for group in ("cpu", "memory", "io"):
        try:
            if group == "cpu":
                cpu = proc.cpu_times()
                row.update(cpu_user_s=cpu.user, cpu_system_s=cpu.system)
            elif group == "memory":
                row["rss_bytes"] = proc.memory_info().rss
            else:
                io = proc.io_counters()
                for key, attr in (("read_bytes", "read_bytes"), ("write_bytes", "write_bytes"),
                                  ("rchar", "read_chars"), ("wchar", "write_chars")):
                    value = getattr(io, attr, None)
                    row[key] = value if value is not None and value >= 0 else None
                    if row[key] is None:
                        row["errors"].append(f"pid {pid} io: {attr} unavailable")
        except (psutil.Error, OSError, AttributeError, NotImplementedError) as exc:
            row["errors"].append(f"pid {pid} {group}: {type(exc).__name__}: {exc}")
    return row


def sample_system():
    """Host-wide absolute counters and memory, with field-level null failures."""
    result = dict.fromkeys(("cpu_user_s", "cpu_system_s", "cpu_idle_s", "cpu_iowait_s",
                           "mem_total_bytes", "mem_available_bytes", "mem_used_bytes",
                           "disk_read_bytes", "disk_write_bytes"))
    result["errors"] = []
    for group in ("cpu", "memory", "disk"):
        try:
            if group == "cpu":
                cpu = psutil.cpu_times()
                result.update(cpu_user_s=cpu.user, cpu_system_s=cpu.system,
                              cpu_idle_s=cpu.idle, cpu_iowait_s=getattr(cpu, "iowait", None))
            elif group == "memory":
                memory = psutil.virtual_memory()
                result.update(mem_total_bytes=memory.total, mem_available_bytes=memory.available,
                              mem_used_bytes=memory.used)
            else:
                disk = psutil.disk_io_counters()
                if disk is None:
                    raise RuntimeError("disk counters unavailable")
                result.update(disk_read_bytes=disk.read_bytes, disk_write_bytes=disk.write_bytes)
        except (psutil.Error, OSError, AttributeError, RuntimeError) as exc:
            result["errors"].append(f"system {group}: {type(exc).__name__}: {exc}")
    return result


class HostSampler:
    """One persistent accumulator; rediscover processes for every sample."""

    def __init__(self, container="jiangchuan-qwen38-zh"):
        self.container = container
        self.accumulator = ProcessAccumulator()

    def sample(self):
        started = time.monotonic_ns()
        timestamp = time.time()
        discovery = discover_components(self.container)
        records = {name: [read_process(pid) for pid in sorted(pids)]
                   for name, pids in discovery["pids"].items()}
        components = self.accumulator.update(records, discovery["errors"])
        gpus, gpu_errors = sample_gpus()
        system = sample_system()
        system["gpu_errors"] = gpu_errors
        return {"timestamp": timestamp, "monotonic_ns": started,
                "sample_ms": (time.monotonic_ns() - started) / 1e6,
                "components": components, "gpus": gpus, "system": system}


def sample_gpus():
    """nvidia-smi gives per-device telemetry, never inferred execution time."""
    fields = ("gpu_index", "util_pct", "mem_used_mb", "mem_total_mb", "power_w", "sm_clock_mhz")
    command = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm",
               "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return [], [f"nvidia-smi: {type(exc).__name__}: {exc}"]
    gpus, errors = [], []
    for values in csv.reader(output.splitlines()):
        if len(values) != len(fields):
            errors.append(f"nvidia-smi: expected {len(fields)} columns, got {len(values)}")
            continue
        gpu = {"errors": []}
        for key, value in zip(fields, values):
            try:
                number = float(value.strip())
                if not math.isfinite(number) or number < 0:
                    raise ValueError("non-finite or negative")
                gpu[key] = int(number) if key == "gpu_index" else number
            except ValueError:
                gpu[key] = None
                gpu["errors"].append(f"{key} unavailable: {value.strip()}")
        if gpu["mem_total_mb"] == 0:
            gpu["mem_used_mb"] = gpu["mem_total_mb"] = None
            gpu["errors"].append("GPU memory unavailable (zero total / unified memory)")
        gpus.append(gpu)
    return gpus, errors


def discover_components(container="jiangchuan-qwen38-zh"):
    """Map Docker top's *host* PIDs and native process trees on each poll.

    Must run in the host PID namespace. Container-local PID numbers from docker
    exec are never used. Docker/psutil failures remain explicit in errors.
    """
    pids = {name: set() for name in COMPONENTS}
    errors = {name: [] for name in COMPONENTS}
    containers = {"sglang": container, "qdrant": "memory-benchmarks-qdrant-1",
                  "benchmark": "mem0-benchmark-python"}
    for name, target in containers.items():
        try:
            output = subprocess.run(["docker", "top", target, "-eo", "pid"],
                capture_output=True, text=True, check=True, timeout=2).stdout
            for line in output.splitlines():
                word = line.strip()
                if word.isdigit():
                    pids[name].add(int(word))
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            errors[name].append(f"docker top {target}: {type(exc).__name__}: {detail.strip()}")
    parents = {}
    try:
        for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
            try:
                info = proc.info
                pid = info["pid"]
                parents[pid] = info["ppid"]
                args = info.get("cmdline") or []
                if pid == os.getpid():
                    continue
                if any(Path(arg).name == "run_trace.py" for arg in args):
                    pids["benchmark"].add(pid)
                elif args and (Path(args[0]).name.startswith(("sglang", "SGLang")) or
                    (Path(args[0]).name.startswith("python") and any(
                        arg == "sglang" or arg.startswith("sglang.") or
                        Path(arg).name.startswith("sglang") for arg in args[1:]))):
                    pids["sglang"].add(pid)
                elif args and "qdrant" in Path(args[0]).name.lower():
                    pids["qdrant"].add(pid)
            except (psutil.Error, OSError) as exc:
                for issues in errors.values():
                    issues.append(f"process discovery: {type(exc).__name__}: {exc}")
    except (psutil.Error, OSError) as exc:
        for issues in errors.values():
            issues.append(f"process discovery: {type(exc).__name__}: {exc}")
    for members in pids.values():
        while True:
            children = {pid for pid, parent in parents.items() if parent in members}
            if children <= members:
                break
            members.update(children)
    # A PID must never be counted in two components (specific services win).
    pids["qdrant"] -= pids["sglang"]
    pids["benchmark"] -= pids["sglang"] | pids["qdrant"]
    return {"pids": pids, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Append flushed JSONL samples")
    parser.add_argument("--interval", type=float, default=.1, help="Target polling period in seconds")
    parser.add_argument("--duration", type=float, help="Stop after seconds (otherwise wait for signal)")
    parser.add_argument("--container", default="jiangchuan-qwen38-zh", help="SGLang Docker container")
    args = parser.parse_args(argv)
    for key in ("interval", "duration"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key} must be finite and positive")
    stop = threading.Event()
    previous = {}

    def handle_signal(signum, frame):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, handle_signal)
    try:
        sampler = HostSampler(args.container)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        deadline = None if args.duration is None else time.monotonic() + args.duration
        with args.out.open("a", encoding="utf-8") as output:
            while not stop.is_set():
                started = time.monotonic()
                if deadline is not None and started >= deadline:
                    break
                row = sampler.sample()
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                delay = max(0, args.interval - (time.monotonic() - started))
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.monotonic()))
                stop.wait(delay)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
