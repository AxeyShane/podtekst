import json
import tempfile
import unittest
from pathlib import Path

from score_device import load_results, score

ROW = {"source_lang": "ru", "source_text": "Ну спасибо, конечно.", "translation": "Well, thanks a lot.",
       "has_subtext": True, "category": "sarcasm", "nuance_note": "Ironic thanks."}
PLAIN = {"source_lang": "en", "source_text": "I missed you.", "translation": "Я скучал по тебе.",
         "has_subtext": False, "category": "none", "nuance_note": ""}
CRUDE = {"source_lang": "ru", "source_text": "Мне похуй, что они думают.",
         "translation": "I don't give a fuck what they think.", "has_subtext": False, "category": "none",
         "nuance_note": ""}


def reply(translation, has, cat):
    return json.dumps({"translation": translation, "has_subtext": has, "category": cat, "nuance_note": ""},
                      ensure_ascii=False)


def write(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for row, text in items:
            f.write(json.dumps({"row": row, "reply": text, "parsed": None, "ms": {"total": 900}}, ensure_ascii=False) + "\n")


class ScoreDeviceTest(unittest.TestCase):
    def test_groups_by_probe_file_and_scores(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            test = d / "podtekst-rua-q3-int4__sft_test__20261016-0300.jsonl"
            probe = d / "podtekst-rua-q3-int4__crude_probe__20261016-0310.jsonl"
            write(test, [(ROW, reply("Well, thanks a lot.", True, "sarcasm")),
                         (PLAIN, "not json at all")])
            write(probe, [(CRUDE, reply("I don't give a fuck what they think.", False, "none"))])

            groups = load_results([test, probe])
            self.assertEqual([len(groups[k][0]) for k in ("test", "probe", "prof")], [2, 1, 0])

            report = score([str(d / "*.jsonl")], "phone-check", out_dir=d / "out")
            self.assertEqual(report["n"], 2)
            self.assertEqual(report["json_valid_rate"], 0.5)
            self.assertEqual(report["crude"]["probe_rows"], 1)
            self.assertTrue((d / "out" / "eval_phone-check.json").is_file())
            self.assertEqual(report["base"], "device:podtekst-rua-q3-int4")

    def test_unknown_file_names_count_as_test_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "something.jsonl"
            write(p, [(PLAIN, reply("Я скучал по тебе.", False, "none"))])
            self.assertEqual(len(load_results([p])["test"][0]), 1)


if __name__ == "__main__":
    unittest.main()
