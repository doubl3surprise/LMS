"""Pure interval arithmetic and bounded scheduling for the Task6 probe.

All intervals use milliseconds on a single client perf_counter clock. Native
engine timestamps must never be mixed with these intervals.
"""
import math


def _merged(intervals):
    ordered = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Expected finite, nondecreasing interval endpoints")
        if end > start:
            ordered.append((start, end))
    merged = []
    for start, end in sorted(ordered):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_union_ms(intervals):
    """Occupied wall time, counting nested/overlapping spans only once."""
    return sum(end - start for start, end in _merged(intervals))


def overlap_ms(left, right):
    """Intersection of two interval unions, not their Cartesian duration sum."""
    a, b = _merged(left), _merged(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


MODES = ("retrieval_only", "llm_only", "serial", "concurrent")


def theoretical_makespan_ms(retrieval_ms, llm_ms):
    """No-contention ideal bound for independent work; not QA latency."""
    for value in (retrieval_ms, llm_ms):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Durations must be finite and nonnegative")
    return {"serial_ms": retrieval_ms + llm_ms,
            "overlap_ms": max(retrieval_ms, llm_ms),
            "max_saved_ms": min(retrieval_ms, llm_ms)}


def trial_schedule(repeats=3, seed=42):
    """Randomized Latin rotation: each block contains each mode exactly once.

    A complete four-block cycle balances every mode's ordinal position.
    The default three blocks are a bounded, partially balanced subset.
    """
    import random
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    rng = random.Random(seed)
    modes = list(MODES)
    rng.shuffle(modes)
    offsets = list(range(len(modes)))
    rng.shuffle(offsets)
    return [(rep, modes[(position + offsets[rep % len(modes)]) % len(modes)])
            for rep in range(repeats) for position in range(len(modes))]


def run_mode(mode, retrieve, generate):
    """Run at most one sequential SDK worker and one LLM worker.

    The barrier synchronizes worker launch; measured intervals establish
    whether the underlying calls actually overlap. It proves no GPU overlap.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    if mode not in MODES:
        raise ValueError("Unknown mode: " + str(mode))
    result = {"retrieval": None, "llm": None}
    if mode == "concurrent":
        barrier = Barrier(2)
        def invoke(fn):
            barrier.wait(timeout=30)
            return fn()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="task6") as pool:
            r = pool.submit(invoke, retrieve)
            g = pool.submit(invoke, generate)
            result["retrieval"], result["llm"] = r.result(), g.result()
    else:
        if mode != "llm_only":
            result["retrieval"] = retrieve()
        if mode != "retrieval_only":
            result["llm"] = generate()
    return result
