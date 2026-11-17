import hashlib
import unittest

from collect_memory_spans import collect_request


class FakeEncoding:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets


class FakeTokenizer:
    def __init__(self, rendered, offsets, encoded_ids=None):
        self.rendered = rendered
        self.offsets = offsets
        self.encoded_ids = encoded_ids

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return self.rendered

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = self.encoded_ids if self.encoded_ids is not None else [1] * len(self.offsets)
        return FakeEncoding(ids, self.offsets)


class MemorySpanTest(unittest.TestCase):
    def test_returns_hash_and_overlapping_token_range_for_unique_user_memory(self):
        rendered = "<user>Context: abcde\nQuestion?"
        offsets = [(0, 6), (6, 15), (15, 18), (18, 21), (21, len(rendered))]
        row = {
            "request_id": "r1",
            "messages": [{"role": "user", "content": "Context: abcde\nQuestion?"}],
            "memories": [{"id": "m1", "text": "abcde"}],
        }
        result = collect_request(row, [1, 1, 1, 1, 1], FakeTokenizer(rendered, offsets))
        span = result["spans"][0]
        self.assertEqual((span["start"], span["end"]), (2, 4))
        self.assertEqual(span["text_sha256"], hashlib.sha256(b"abcde").hexdigest())
        self.assertTrue(result["verified"])

    def test_rank_label_disambiguates_identical_memory_text(self):
        current='Context:\n[Memory 1] same\n[Memory 2] same\nQuestion: same'
        rendered='<user>'+current
        row={'messages':[{'role':'user','content':current}],'memories':[{'id':'a','text':'same','rank':1},{'id':'b','text':'same','rank':2}]}
        offsets=[(i,i+1) for i in range(len(rendered))]
        result=collect_request(row,[1]*len(rendered),FakeTokenizer(rendered,offsets))
        self.assertTrue(result['verified'])
        self.assertLess(result['spans'][0]['end'],result['spans'][1]['start'])

    def test_rejects_ambiguous_duplicate_memory_text(self):
        row = {
            "request_id": "r1",
            "messages": [{"role": "user", "content": "x ... x"}],
            "memories": [{"id": "m1", "text": "x"}],
        }
        result = collect_request(row, [1], FakeTokenizer("<user>x ... x", [(0, 13)]))
        self.assertFalse(result["verified"])
        self.assertEqual(result["failure"], "memory_text_not_unique_in_final_user_message:m1")
        self.assertEqual(result["spans"], [])

    def test_rejects_non_roundtripping_prompt_ids_without_guessing(self):
        row = {"request_id": "r1", "messages": [{"role": "user", "content": "x"}], "memories": []}
        result = collect_request(row, [1, 2], FakeTokenizer("x", [(0, 1)], encoded_ids=[1]))
        self.assertFalse(result["verified"])
        self.assertIsNotNone(result["failure"])
        self.assertEqual(result["spans"], [])

    def test_does_not_use_occurrence_from_earlier_message(self):
        row = {
            "request_id": "r1",
            "messages": [
                {"role": "assistant", "content": "needle"},
                {"role": "user", "content": "Question?"},
            ],
            "memories": [{"memory_id": "m1", "text": "needle"}],
        }
        rendered = "<assistant>needle<user>Question?"
        result = collect_request(row, [1], FakeTokenizer(rendered, [(0, len(rendered))]))
        self.assertFalse(result["verified"])
        self.assertEqual(result["spans"], [])


if __name__ == "__main__":
    unittest.main()
