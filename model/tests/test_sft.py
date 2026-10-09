import json
import unittest

from podtekst_sft.metrics import category_scores, detection_scores, summarize
from podtekst_sft.prompt import SYSTEM_PROMPT, build_messages, parse_reply, target_json, user_content
from prepare_sft import convert
from train_lora import common_prefix_len, oversample, parse_oversample, tokenize_example

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
    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True, **kw):
        s = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        if add_generation_prompt:
            s += "<assistant>"
        return [ord(ch) for ch in s]


class DictTok(FakeTok):
    """transformers 5.x style: apply_chat_template(tokenize=True) returns a dict."""
    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True, **kw):
        ids = super().apply_chat_template(msgs, add_generation_prompt, tokenize)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TokenizeTest(unittest.TestCase):
    def test_dict_return_from_new_transformers(self):
        ex = tokenize_example(DictTok(), build_messages(ROW), 2000)
        self.assertIsNotNone(ex)
        kept = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100)
        self.assertEqual(kept, target_json(ROW))

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



class EndpointTest(unittest.TestCase):
    def test_chat_endpoint_posts_greedy_request(self):
        from evaluate import chat_endpoint
        seen = {}

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": '{"translation": "Hi"}'}}]}

        def post(url, json, timeout):
            seen.update(url=url, body=json)
            return Resp()
        out = chat_endpoint("http://127.0.0.1:8080/v1/", build_messages(ROW, with_answer=False), 64, post=post)
        self.assertEqual(out, '{"translation": "Hi"}')
        self.assertEqual(seen["url"], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(seen["body"]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()


class OversampleTest(unittest.TestCase):
    def test_repeats_only_named_categories(self):
        rows = [{"row": {"category": "none"}}, {"row": {"category": "formality_shift"}},
                {"messages": [{"role": "assistant", "content": '{"category": "emotional_subtext"}'}]}]
        out = oversample(rows, parse_oversample("formality_shift=2, emotional_subtext=3"))
        self.assertEqual(len(out), 1 + 2 + 3)

    def test_empty_spec_is_noop(self):
        rows = [{"row": {"category": "none"}}]
        self.assertIs(oversample(rows, parse_oversample("")), rows)
