from typing import Hashable, Iterable, List, Optional


def reuse_distances(keys: Iterable[Hashable]) -> List[Optional[int]]:
    """Return distinct-key reuse distances; cold accesses are represented by None."""
    sequence = list(keys)
    last_seen = {}
    distances = []
    for index, key in enumerate(sequence):
        previous = last_seen.get(key)
        if previous is None:
            distances.append(None)
        else:
            distances.append(len(set(sequence[previous + 1:index])))
        last_seen[key] = index
    return distances


def simulate(events, capacity, policy="LRU", unit="objects"):
    """Simulate sequential demand accesses using LRU or resident-frequency LFU."""
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    policy = policy.upper()
    unit = unit.lower()
    if policy not in ("LRU", "LFU"):
        raise ValueError("policy must be 'LRU' or 'LFU'")
    if unit not in ("objects", "bytes"):
        raise ValueError("unit must be 'objects' or 'bytes'")

    cache = {}
    clock = 0
    used = 0
    hits = misses = hit_bytes = miss_bytes = evictions = 0
    for event in events:
        key = event["key"]
        size = event["size"]
        if size <= 0:
            raise ValueError("event size must be positive")
        clock += 1
        if key in cache:
            hits += 1
            hit_bytes += size
            entry = cache[key]
            entry["frequency"] += 1
            entry["last"] = clock
            continue

        misses += 1
        miss_bytes += size
        weight = 1 if unit == "objects" else size
        if capacity == 0 or weight > capacity:
            continue

        while used + weight > capacity:
            victim = min(cache, key=lambda item: (
                cache[item]["frequency"] if policy == "LFU" else cache[item]["last"],
                cache[item]["last"],
            ))
            used -= cache[victim]["weight"]
            del cache[victim]
            evictions += 1

        cache[key] = {"weight": weight, "frequency": 1, "last": clock}
        used += weight

    total = hits + misses
    total_bytes = hit_bytes + miss_bytes
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "byte_hit_rate": hit_bytes / total_bytes if total_bytes else 0.0,
        "hit_bytes": hit_bytes,
        "miss_bytes": miss_bytes,
        "evictions": evictions,
    }
