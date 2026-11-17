import unittest

from cache_core import reuse_distances, simulate


class ReuseDistanceTests(unittest.TestCase):
    def test_counts_distinct_intervening_keys_and_cold_accesses(self):
        self.assertEqual(
            reuse_distances(["a", "b", "b", "c", "a", "b", "a"]),
            [None, None, 0, None, 2, 2, 1],
        )

    def test_lru_hits_match_reuse_distances_below_object_capacity(self):
        keys = ["a", "b", "c", "a", "b", "a", "d", "b"]
        events = [{"key": key, "size": 1} for key in keys]
        result = simulate(events, capacity=3, policy="LRU")
        expected_hits = sum(distance is not None and distance < 3 for distance in reuse_distances(keys))
        self.assertEqual((result["hits"], result["misses"]), (expected_hits, len(keys) - expected_hits))
        self.assertEqual(result["hit_bytes"] + result["miss_bytes"], len(keys))

    def test_lfu_uses_resident_frequency_and_lru_tie_break(self):
        events = [{"key": key, "size": 1} for key in "abacbda"]
        result = simulate(events, capacity=3, policy="LFU")
        self.assertEqual((result["hits"], result["misses"], result["evictions"]), (3, 4, 1))
        lru_result = simulate(events, capacity=3, policy="LRU")
        self.assertEqual((lru_result["hits"], lru_result["misses"]), (2, 5))

    def test_lfu_frequency_tie_evicts_least_recently_used(self):
        events = [{"key": key, "size": 1} for key in "abca"]
        result = simulate(events, capacity=2, policy="LFU")
        self.assertEqual((result["hits"], result["misses"], result["evictions"]), (0, 4, 2))

    def test_byte_capacity_and_oversized_bypass_do_not_evict(self):
        events = [
            {"key": "a", "size": 3},
            {"key": "large", "size": 8},
            {"key": "a", "size": 3},
            {"key": "b", "size": 2},
            {"key": "large", "size": 8},
        ]
        result = simulate(events, capacity=5, policy="LRU", unit="bytes")
        self.assertEqual((result["hits"], result["misses"], result["evictions"]), (1, 4, 0))
        self.assertEqual((result["hit_bytes"], result["miss_bytes"]), (3, 21))

    def test_zero_capacity_and_hit_rates(self):
        result = simulate([{"key": "a", "size": 2}, {"key": "a", "size": 2}], 0)
        self.assertEqual((result["hits"], result["misses"], result["evictions"]), (0, 2, 0))
        self.assertEqual((result["hit_rate"], result["byte_hit_rate"]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
