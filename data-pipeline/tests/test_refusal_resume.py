import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retry_failures as rf
import stage_a_generate as sa


def ok_call(sentence, slug, api_key, provider=None, api_model=None):
    return {"source_text": sentence, "_generator_model": slug, "translation": "t"}


class RefusalDetectionTest(unittest.TestCase):
    def test_markers(self):
        self.assertTrue(sa.looks_like_refusal("I cannot fulfill this request, as it involves ..."))
        self.assertTrue(sa.looks_like_refusal("I'm sorry, but I can't help with that."))
        self.assertFalse(sa.looks_like_refusal("Here is the analysis: the sentence uses ты"))

    def test_call_model_raises_refused(self):
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"choices": [{"message": {"content": "I cannot fulfill this request."},
                                               "finish_reason": "stop"}]}
        with mock.patch.object(sa.requests, "post", return_value=resp):
            with self.assertRaises(sa.Refused):
                sa.call_model("s", "model/a", "k")


class RetryRefusedTest(unittest.TestCase):
    def test_refused_kept_apart(self):
        def call(sentence, slug, api_key, provider=None, api_model=None):
            if slug == "model/b":
                raise sa.Refused("REFUSED by model/b: no")
            return ok_call(sentence, slug, api_key)
        refused = []
        results, still = rf.retry_pairs([("s", "model/a"), ("s", "model/b")], "k", sleep=0, call=call,
                                        check_balance=lambda _: True, refused=refused)
        self.assertEqual(len(results), 1)
        self.assertEqual(still, [])
        self.assertEqual([r["model"] for r in refused], ["model/b"])


class LoadDonePairsTest(unittest.TestCase):
    def test_reads_seed_and_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"source_text": "a", "_generator_model": "m/1"}) + "\n\n")
            self.assertEqual(rf.load_done_pairs(p), {("a", "m/1")})
            self.assertEqual(rf.load_done_pairs(os.path.join(d, "missing.jsonl")), set())


if __name__ == "__main__":
    unittest.main()
