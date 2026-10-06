import json
import unittest

from podtekst_sft.metrics import category_scores, detection_scores, summarize
from podtekst_sft.prompt import SYSTEM_PROMPT, build_messages, parse_reply, target_json, user_content
from prepare_sft import convert
from train_lora import common_prefix_len, tokenize_example

ROW = {"source_lang": "ru", "source_text": "Ты что, обиделся?", "translation": "What, are you sulking?",
       "has_subtext": True, "category": "emotional_subtext", "nuance_note": "Обиделся is hurt plus a sulk."}


class PromptTest(unittest.TestCase):
    def test_messages_and_target(self):
        m = build_messages(ROW)
        self.assertEqual([x["role"] for x in m], ["system", "user", "assistant"])
        self.assertEqual(m[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(json.loads(m[2]["content"])["category"], "emotional_subtext")
        self.assertEqual(len(build_messages(ROW, with_answer=False)), 2)

    def test_none_row_has_empty_note(self):
        t = json.loads(target_json({**ROW, "has_subtext": False, "category": "none", "nuance_note": "x"}))
        self.assertEqual((t["category"], t["nuance_note"]), ("none", ""))

    def test_gender_hint(self):
        self.assertEqual(user_content("Я устала.", "female"), "Я устала.\n(speaker: female)")
        self.assertEqual(user_content("Hi"), "Hi")
        m = build_messages({**ROW, "addressee_gender": "male"})
        self.assertIn("(addressee: male)", m[1]["content"])

    def test_parse_reply(self):
        good = '<think>\n\n</think>\n```json\n{"translation": "Hi", "has_subtext": "true", "category": "idiom", "nuance_note": "n"}\n```'
        self.assertEqual(parse_reply(good)["category"], "idiom")
        self.assertIsNone(parse_reply("Sorry, I can't"))
        self.assertIsNone(parse_reply('{"translation": 5}'))
        bad_cat = parse_reply('{"translation": "Hi", "has_subtext": true, "category": "tone"}')
        self.assertEqual((bad_cat["has_subtext"], bad_cat["category"]), (False, "none"))


class MetricsTest(unittest.TestCase):
    def test_detection(self):
        d = detection_scores([True, True, False, False], [True, False, True, False])
        self.assertEqual((d["precision"], d["recall"], d["false_positive_rate"]), (0.5, 0.5, 0.5))

    def test_category_and_summary(self):
        c = category_scores(["idiom", "none"], ["idiom", "sarcasm"])
        self.assertEqual(c["idiom"]["f1"], 1.0)
        self.assertEqual(c["accuracy"], 0.5)
        s = summarize([ROW, {**ROW, "has_subtext": False, "category": "none"}],
                      [None, {"translation": "x", "has_subtext": False, "category": "none"}])
        self.assertEqual(s["json_valid_rate"], 0.5)
        self.assertEqual(s["detection"]["fn"], 1)


class FakeTok:
    """Char-level fake chat template: system/user/assistant joined with markers."""
    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        s = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        if add_generation_prompt:
            s += "<assistant>"
        return [ord(ch) for ch in s]


class TokenizeTest(unittest.TestCase):
    def test_masks_prompt(self):
        ex = tokenize_example(FakeTok(), build_messages(ROW), 2000)
        answer = target_json(ROW)
        kept = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100)
        self.assertEqual(kept, answer)

    def test_too_long_skipped(self):
        self.assertIsNone(tokenize_example(FakeTok(), build_messages(ROW), 50))

    def test_common_prefix(self):
        self.assertEqual(common_prefix_len([1, 2, 3], [1, 2, 4, 5]), 2)


class PrepareTest(unittest.TestCase):
    def test_convert_keeps_gold(self):
        out = convert([{**ROW, "_batch_file": "x.jsonl", "speaker_gender": "male"}])
        self.assertNotIn("_batch_file", out[0]["row"])
        self.assertEqual(out[0]["row"]["speaker_gender"], "male")
        self.assertEqual(len(out[0]["messages"]), 3)


if __name__ == "__main__":
    unittest.main()
