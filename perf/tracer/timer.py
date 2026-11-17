"""
timer.py — PhasedTimer
======================
用法：
    t = PhasedTimer()
    with t.phase("embed"):
        vec = embedder.embed(query)
    with t.phase("search"):
        hits = qdrant.search(...)

    t.ms("embed")  # → 3.2
    t.all()        # → {"embed": 3.2, "search": 4.1, ...}
"""

import time
from contextlib import contextmanager


class PhasedTimer:
    def __init__(self):
        self._phases: dict[str, float] = {}   # phase → elapsed ms
        self._start: dict[str, float] = {}    # phase → perf_counter start

    @contextmanager
    def phase(self, name: str):
        """Context manager that measures wall-clock time for one phase."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            # 累加（允许同名 phase 多次调用）
            self._phases[name] = self._phases.get(name, 0.0) + elapsed

    def ms(self, name: str) -> float | None:
        return self._phases.get(name)

    def all(self) -> dict[str, float]:
        return dict(self._phases)
