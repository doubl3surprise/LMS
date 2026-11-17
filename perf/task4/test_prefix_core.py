import unittest

from prefix_core import analyze_prefixes


class AnalyzePrefixesTests(unittest.TestCase):
    def test_matches_exact_token_id_prefixes_only(self):
        result = analyze_prefixes([[10, 20, 30], [10, 20, 40], [10, 20, 30]])

        self.assertEqual([row["prior_max_prefix_tokens"] for row in result], [0, 2, 3])
        self.assertEqual([row["full_prompt_duplicate"] for row in result], [False, False, True])
        self.assertEqual([row["repeated_tokens_this_request"] for row in result], [0, 2, 3])

    def test_cumulative_repeats_and_totals_are_explicit(self):
        result = analyze_prefixes([[1, 2, 3], [1, 2, 4, 5], [9]])

        self.assertEqual(result[-1]["cumulative_repeated_tokens"], 2)
        self.assertEqual(result[-1]["cumulative_total_tokens"], 8)
        self.assertEqual(result[-1]["cumulative_repeated_fraction"], 0.25)

    def test_block_alignment_is_a_separate_logical_opportunity(self):
        result = analyze_prefixes([[1, 2, 3, 4, 5], [1, 2, 3, 8]], block_size=4)

        self.assertEqual(result[1]["prior_max_prefix_tokens"], 3)
        self.assertEqual(result[1]["block_aligned_prefix_tokens"], 0)

    def test_rejects_invalid_block_size_and_non_integer_ids(self):
        with self.assertRaises(ValueError):
            analyze_prefixes([[1]], block_size=0)
        with self.assertRaises(TypeError):
            analyze_prefixes([[1, "2"]])

    def test_empty_inputs_and_zero_denominator(self):
        self.assertEqual(analyze_prefixes([]), [])
        result = analyze_prefixes([[]])
        self.assertEqual(result[0]["cumulative_repeated_fraction"], 0.0)
        self.assertTrue(result[0]["full_prompt_duplicate"] is False)


if __name__ == "__main__":
    unittest.main()
