"""Stage A gender context: read from seed meta, sent as a second system message (no network)."""
import json
import os
import tempfile
import unittest
from unittest import mock

import stage_a_generate as sa


def ok_response(payload):
    body = {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}
    return mock.Mock(ok=True, status_code=200, json=lambda: body)


class GenderHintsTest(unittest.TestCase):
    def tearDown(self):
        sa.GENDER_HINTS.clear()

    def test_load_keeps_only_known_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "seeds.meta.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"seed": "Я устал.", "speaker_gender": "male"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"seed": "Ты пришла?", "addressee_gender": "female"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"seed": "Привет.", "speaker_gender": "unknown"}, ensure_ascii=False) + "\n")
            hints = sa.load_gender_hints(p)
        self.assertEqual(hints, {"Я устал.": {"speaker_gender": "male"},
                                 "Ты пришла?": {"addressee_gender": "female"}})

    def test_hint_goes_in_a_system_message_and_onto_the_row(self):
        sa.GENDER_HINTS["I'm so tired."] = {"speaker_gender": "female"}
        reply = {"source_lang": "en", "source_text": "I'm so tired.", "translation": "Я так устала.",
                 "has_subtext": False, "category": "none", "nuance_note": ""}
        with mock.patch.object(sa.requests, "post", return_value=ok_response(reply)) as post:
            row = sa.call_model("I'm so tired.", "m/x", "key")
        msgs = post.call_args.kwargs["json"]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "system", "user"])
        self.assertIn("the speaker is a woman", msgs[1]["content"])
        self.assertEqual(msgs[2]["content"], "I'm so tired.")
        self.assertEqual(row["_speaker_gender"], "female")

    def test_no_hint_no_extra_message(self):
        reply = {"source_lang": "en", "source_text": "Hi.", "translation": "Привет.",
                 "has_subtext": False, "category": "none", "nuance_note": ""}
        with mock.patch.object(sa.requests, "post", return_value=ok_response(reply)) as post:
            row = sa.call_model("Hi.", "m/x", "key")
        self.assertEqual(len(post.call_args.kwargs["json"]["messages"]), 2)
        self.assertNotIn("_speaker_gender", row)


if __name__ == "__main__":
    unittest.main()


class TruncatedReplyTest(unittest.TestCase):
    def test_truncated_json_is_retried_with_more_tokens(self):
        cut = mock.Mock(ok=True, status_code=200, json=lambda: {
            "choices": [{"finish_reason": "length",
                         "message": {"content": '{"source_lang": "ru", "source_text": "x", "translation": "In the circ'}}]})
        full = ok_response({"source_lang": "ru", "source_text": "x", "translation": "y",
                            "has_subtext": False, "category": "none", "nuance_note": ""})
        with mock.patch.object(sa.requests, "post", side_effect=[cut, full]) as post:
            row = sa.call_model_with_retry("x", "m/x", "key")
        self.assertEqual(row["translation"], "y")
        self.assertEqual(post.call_args_list[1].kwargs["json"]["max_tokens"], 2500)
