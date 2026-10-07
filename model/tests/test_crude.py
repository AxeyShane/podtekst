import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from podtekst_sft.crude import crude_report, is_crude, masked, refused, target_crude  # noqa: E402


class StemTest(unittest.TestCase):
    def test_russian_hits_with_prefixes(self):
        for t in ["Бля, опять", "мне похуй", "заебал уже", "охуеть", "иди нахер", "отъебись", "ну ты жопа"]:
            self.assertTrue(is_crude(t, "ru"), t)

    def test_russian_lookalikes_do_not_count(self):
        for t in ["Купи хлеба", "кошки на душе скребут", "животный страх", "сто рублей", "напеки блинов"]:
            self.assertFalse(is_crude(t, "ru"), t)

    def test_english(self):
        for t in ["worked your asses off", "I'll rip you a new one", "fucking rain", "screw this"]:
            self.assertTrue(is_crude(t, "en"), t)
        for t in ["class", "assess", "grass", "hello", "Dickens"]:
            self.assertFalse(is_crude(t, "en"), t)

    def test_target_is_other_language(self):
        self.assertTrue(target_crude("This fucking rain", "ru"))
        self.assertFalse(target_crude("This rain", "ru"))

    def test_masked_and_refused(self):
        self.assertTrue(masked("f*** this"))
        self.assertTrue(masked("б**ть"))
        self.assertFalse(masked("plain text"))
        self.assertTrue(refused("I'm sorry, but I can't help with that."))
        self.assertFalse(refused('{"translation": "Fuck off"}'))


class ReportTest(unittest.TestCase):
    def test_probe_rows_always_counted(self):
        rows = [{"source_text": "Привет", "source_lang": "ru", "translation": "Hi"},
                {"source_text": "Привет ещё", "source_lang": "ru", "translation": "Fucking hi"}]
        preds = [{"translation": "Hi"}, {"translation": "f*** hi"}]
        rep = crude_report(rows, ["{}", "{}"], preds, always={1})
        self.assertEqual(rep["n"], 1)
        self.assertEqual(rep["profanity_kept_rate"], 0.0)
        self.assertEqual(rep["masked_rate"], 1.0)

    def test_none_when_empty(self):
        self.assertIsNone(crude_report([{"source_text": "Привет", "translation": "Hi"}], ["{}"], [None]))

    def test_probe_file_is_consistent(self):
        rows = [json.loads(l) for l in open(HERE / "probes" / "crude_probe.jsonl", encoding="utf-8")]
        self.assertGreaterEqual(len(rows), 20)
        for r in rows:
            self.assertTrue(is_crude(r["source_text"], r["source_lang"]), r["source_text"])
            self.assertTrue(target_crude(r["translation"], r["source_lang"]), r["translation"])


if __name__ == "__main__":
    unittest.main()
