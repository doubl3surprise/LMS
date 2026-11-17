import unittest
from pathlib import Path
import tempfile
import analyze
class AnalysisTest(unittest.TestCase):
 def test_cohort_counts_and_no_strategy_mixing(self):
  rows=[{'condition':{'id':'native_n879_k10'},'memories':[]} for _ in range(32)]+[{'condition':{'id':'full_history'}} for _ in range(32)]
  self.assertEqual(len(analyze.select_rows('native32',rows)),32)
 def test_memory_overlap_is_intersection_not_prefix_total(self):
  spans=[{'start':10,'end':20,'text_sha256':'x'},{'start':22,'end':25,'text_sha256':'y'}]
  self.assertEqual(analyze.memory_coverage(spans,15,{'x'}),{'memory_tokens':13,'previously_seen_text_tokens':10,'compatible_memory_prefix_tokens':5,'compatible_repeated_text_tokens':5})
  self.assertEqual(analyze.memory_coverage(spans,5,{'x'})['compatible_memory_prefix_tokens'],0)
 def test_text_equal_does_not_imply_id_equal(self):
  left=[(1,[('a','text')])];right=[(1,[('b','text')])]
  s=analyze.compare_sequences(left,right)
  self.assertFalse(s['same_question_and_id_text_order'])
  self.assertTrue(s['same_question_and_text_order'])
 def test_cache_replay_cost_is_paired_not_uncached_assumption(self):
  rs=[{'condition':'native','pair':1,'phase':'first','prefill_ms':100,'cached_tokens':0,'prompt_tokens':20},{'condition':'native','pair':1,'phase':'repeat','prefill_ms':30,'cached_tokens':12,'prompt_tokens':20}]
  s=analyze.replay_stats(rs)['native']
  self.assertEqual(s['pair_count'],1)
  self.assertEqual(s['repeat_hit_requests'],1)
  self.assertEqual(s['prefill_first_mean_ms']-s['prefill_repeat_mean_ms'],70)
if __name__=='__main__':unittest.main()
