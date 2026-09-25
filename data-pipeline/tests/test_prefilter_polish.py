"""Prefilter polish recovery with OpenRouter mocked (no network, no spend).
Run from data-pipeline/:  python -m unittest discover -s tests -t .
"""
import json
import unittest
from unittest import mock

import stage_b_prefilter as pf

PRIMARY = {"slug": "z-ai/glm-5.3-flash", "provider": {"order": ["deepinfra"], "allow_fallbacks": True}}
FALLBACK = {"slug": "deepseek/deepseek-v4-flash", "provider": {"order": ["streamlake"], "allow_fallbacks": True}}
AGREEING = [{"translation": "Could you pass the menu?", "nuance_note": "Formal вы."},
            {"translation": "Would you pass the menu, please?", "nuance_note": "Polite вы."}]
GOOD = json.dumps({"translation": "Could you please pass the menu?", "nuance_note": "Formal вы: polite request."})


def reply(content, finish="stop", reasoning_tokens=20, provider="InferenceNet"):
    r = mock.Mock(ok=True, status_code=200)
    r.json.return_value = {
        "provider": provider,
        "choices": [{"finish_reason": finish, "message": {"content": content, "reasoning": "..." * 10}}],
        "usage": {"completion_tokens": 60 + reasoning_tokens,
                  "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
    }
    return r


def run(*replies):
    """Polish once with the given sequence of OpenRouter replies; returns (picked, path, logs, posted bodies)."""
    post = mock.Mock(side_effect=list(replies))
    picked, path, logs = pf.polish("Будьте добры, передайте меню.", True, "formality_shift", AGREEING,
                                   PRIMARY, FALLBACK, "key", post=post, sleep=lambda s: None)
    return picked, path, logs, [c.kwargs["json"] for c in post.call_args_list]


class PolishRecoveryTests(unittest.TestCase):
    def test_request_asks_for_json_low_reasoning_and_headroom(self):
        picked, path, logs, bodies = run(reply(GOOD))
        body = bodies[0]
        self.assertEqual(body["max_tokens"], 1500)
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertTrue(body["provider"]["require_parameters"])          # skip routes that ignore JSON mode
        self.assertEqual(path, "primary")
        self.assertEqual(logs[0]["finish_reason"], "stop")
        self.assertEqual(logs[0]["reasoning_tokens"], 20)

    def test_length_truncated_reply_is_retried(self):
        cut = '{"translation": "Could you please pass the me'
        picked, path, logs, bodies = run(reply(cut, finish="length", reasoning_tokens=380), reply(GOOD))
        self.assertEqual(path, "primary_retry")
        self.assertEqual(picked["translation"], "Could you please pass the menu?")
        self.assertEqual(logs[0]["finish_reason"], "length")
        self.assertIn("no JSON object", logs[0]["error"])

    def test_empty_reply_with_reasoning_goes_to_retry_then_fallback_model(self):
        empty = reply(None, finish="length", reasoning_tokens=472)
        picked, path, logs, bodies = run(empty, reply("", finish="length", reasoning_tokens=450),
                                         reply(GOOD, provider="StreamLake"))
        self.assertEqual(path, "fallback_model")
        self.assertEqual([b["model"] for b in bodies],
                         ["z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash", "deepseek/deepseek-v4-flash"])
        self.assertEqual(bodies[2]["provider"]["order"], ["streamlake"])
        self.assertEqual(logs[0]["error"], "empty content")
        self.assertEqual(logs[0]["reasoning_tokens"], 472)
        self.assertEqual(logs[-1]["model"], "deepseek/deepseek-v4-flash")

    def test_malformed_json_everywhere_ends_unpolished(self):
        bad = '{"translation": "Could you pass the menu?", "nuance_note": }'
        picked, path, logs, bodies = run(reply(bad), reply(bad), reply("Sure! Here you go."))
        self.assertIsNone(picked)
        self.assertEqual(path, "unpolished")
        self.assertEqual(len(bodies), 3)
        self.assertTrue(all(l.get("error") for l in logs))

    def test_http_error_follows_the_same_chain(self):
        err = mock.Mock(ok=False, status_code=429, text="rate-limited upstream")
        picked, path, logs, _ = run(err, reply(GOOD))
        self.assertEqual(path, "primary_retry")
        self.assertIn("HTTP 429", logs[0]["error"])

    def test_config_supplies_primary_and_fallback(self):
        primary, fallback = pf.load_polish_models("config/models.json")
        self.assertEqual(primary["slug"], "z-ai/glm-5.3-flash")
        self.assertEqual(primary["reasoning"], {"effort": "low"})
        self.assertEqual(fallback["slug"], "deepseek/deepseek-v4-flash")


class PrefilterMainTests(unittest.TestCase):
    def test_parallel_run_keeps_order_and_routes_suspect_rows_to_cowork(self):
        import os
        import sys
        import tempfile

        def cand(src, model, sub=True, cat="sarcasm", **extra):
            return {"source_lang": "ru", "source_text": src, "translation": f"{src} (EN)", "has_subtext": sub,
                    "category": cat, "nuance_note": "n", "_generator_model": model, **extra}
        srcs = [f"Предложение {i}." for i in range(8)]
        rows = [cand(s, m) for s in srcs for m in ("a", "b")]
        rows += [cand("Спорное.", "a"), cand("Спорное.", "b", sub=False)]                       # contested
        rows += [cand("Спасибо, доктор.", "a", _translation_suspect="partial echo")]            # suspect only

        def post(url, headers=None, json=None, timeout=None):
            src = json["messages"][1]["content"].split("\n")[0].removeprefix("Source sentence: ")
            return reply(pf_json.dumps({"translation": f"polished {src}", "nuance_note": "ok"}))
        pf_json = __import__("json")
        with tempfile.TemporaryDirectory() as d:
            inp, res, cow = (os.path.join(d, n) for n in ("in.jsonl", "res.jsonl", "cow.jsonl"))
            pf.write_jsonl(inp, rows)
            argv = ["stage_b_prefilter.py", "--in", inp, "--resolved", res, "--needs-cowork", cow, "--workers", "4"]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}), \
                    mock.patch.object(pf.requests, "post", side_effect=post), mock.patch("builtins.print"):
                pf.main()
            resolved, cowork = pf.load_jsonl(res), pf.load_jsonl(cow)
        self.assertEqual([r["source_text"] for r in resolved], srcs)                             # input order
        self.assertTrue(all(r["translation"] == f"polished {r['source_text']}" for r in resolved))
        self.assertEqual({r["_stage_b_polish_path"] for r in resolved}, {"primary"})
        self.assertEqual({r["source_text"] for r in cowork}, {"Спорное.", "Спасибо, доктор."})


if __name__ == "__main__":
    unittest.main()
