"""Standalone SGLang Prometheus sampling; no imports from tracer.__init__.

Requires prometheus-client. Values retain every label (including TP/DP rank).
Non-finite Prometheus numbers become null plus raw_value for strict JSON output.
"""
import json
import math
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

from prometheus_client.parser import text_string_to_metric_families


def parse_metrics(text):
    """Parse the actual Prometheus exposition grammar, retaining sample names."""
    result = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            value = float(sample.value)
            metric = {"name": sample.name, "labels": dict(sample.labels),
                      "value": value if math.isfinite(value) else None}
            if not math.isfinite(value):
                metric["raw_value"] = "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
            result.append(metric)
    return result


def summary_metrics(samples):
    """Return name -> list of per-label min/max/avg/count, never rank sums.

    A name with no labels still maps to a one-element list (labels={}). Failed
    scrapes and non-finite observations contribute no measurement.
    """
    groups = {}
    for sample in samples:
        if sample.get("error"):
            continue
        for metric in sample.get("metrics", []):
            value = metric.get("value")
            if value is None or not math.isfinite(value):
                continue
            key = (metric["name"], tuple(sorted(metric.get("labels", {}).items())))
            if key not in groups:
                groups[key] = {"labels": dict(key[1]), "min": value, "max": value,
                               "total": 0, "count": 0}
            stats = groups[key]
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)
            stats["total"] += value
            stats["count"] += 1
    result = {}
    for (name, _), stats in sorted(groups.items()):
        stats["avg"] = stats.pop("total") / stats["count"]
        result.setdefault(name, []).append(stats)
    return result


class MetricsSampler:
    """Continuously scrape /metrics to flushed JSONL and ``samples``.

    ``start`` and ``stop`` are idempotent. Restart appends to the same file and
    sample list. Scrapes are sequential; missed intervals are not replayed.
    HTTP requests have a bounded timeout (2s); stop joins an in-flight scrape.
    Background disk failures are raised by stop rather than silently dropped.
    """

    def __init__(self, base_url, out_path, interval=.1):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        self.url = base_url.rstrip("/") + "/metrics"
        self.out_path = Path(out_path)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._file = None
        self._failure = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.out_path.open("a", encoding="utf-8")
        self._stop.clear()
        self._failure = None
        self._thread = threading.Thread(target=self._run, name="sglang-metrics", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._failure is not None:
            raise RuntimeError("Metrics JSONL writer failed") from self._failure

    def _run(self):
        try:
            while not self._stop.is_set():
                started = time.monotonic_ns()
                sample = {"timestamp": time.time(), "monotonic_ns": started,
                          "scrape_ms": None, "metrics": [], "error": None}
                try:
                    request = Request(self.url, headers={"Accept": "text/plain"})
                    with urlopen(request, timeout=2) as response:
                        sample["metrics"] = parse_metrics(response.read().decode("utf-8"))
                except Exception as exc:
                    sample["error"] = f"{type(exc).__name__}: {exc}"
                sample["scrape_ms"] = (time.monotonic_ns() - started) / 1e6
                self._file.write(json.dumps(sample, allow_nan=False) + "\n")
                self._file.flush()
                self.samples.append(sample)
                self._stop.wait(max(0, self.interval - sample["scrape_ms"] / 1000))
        except Exception as exc:
            self._failure = exc
        finally:
            self._file.close()
