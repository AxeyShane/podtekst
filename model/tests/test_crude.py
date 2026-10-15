import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from podtekst_sft.crude import crude_report, is_crude, masked, refused, target_crude, tier  # noqa: E402


class TierTest(unittest.TestCase):
    def test_russian_tiers(self):
        for t, want in [("пиздец как холодно", 3), ("мне похуй", 3), ("иди нахуй", 3), ("заебал", 3), ("Сука, опять", 3),
                        ("хрен его знает", 2), ("какого хрена", 2), ("ну ты жопа", 2), ("хитрожопый", 2), ("сволочь", 2),
                        ("блин, опять", 1), ("какого чёрта", 1), ("мне пофиг", 1), ("офигеть", 1),
                        ("проебал дедлайн", 3), ("ахуеть", 3), ("ну ты и мудила", 3), ("напился до усрачки", 2),
                        ("просрал всё", 2), ("тебе", 0), ("фигура", 0), ("напеки блинов", 0), ("животный страх", 0),
                        ("сук", 0), ("проблема", 0), ("абляция", 0), ("профиль", 0)]:
            self.assertEqual(tier(t, "ru"), want, t)

    def test_english_tiers(self):
        for t, want in [("fucking cold", 3), ("motherfucker", 3), ("bullshit", 3), ("I don't give a shit", 3),
                        ("you bastard", 2), ("dumbass", 2), ("kiss my ass", 2), ("damn it", 1), ("what the hell", 1),
                        ("crap", 1), ("class", 0), ("hello", 0), ("assess", 0), ("shell", 0)]:
            self.assertEqual(tier(t, "en"), want, t)

    def test_kept_by_tier_compares_with_the_reference(self):
        rows = [{"source_text": "Бля", "source_lang": "ru", "translation": "Fuck"},
                {"source_text": "Блин", "source_lang": "ru", "translation": "Damn"}]
        rep = crude_report(rows, ["{}", "{}"], [{"translation": "Damn"}, {"translation": "Damn"}])
        self.assertEqual(rep["profanity_kept_rate_by_tier"], {"mild": 1.0, "medium": None, "strong": 0.0})
        self.assertEqual(rep["profanity_kept_rate"], 0.5)

    def test_kept_is_within_one_tier_same_tier_is_strict(self):
        rows = [{"source_text": "Бля", "source_lang": "ru", "translation": "Fuck"},        # strong -> medium: kept
                {"source_text": "Херня", "source_lang": "ru", "translation": "Bollocks"},  # medium -> strong: kept
                {"source_text": "Жопа", "source_lang": "ru", "translation": "Ass"},        # medium -> none: lost
                {"source_text": "Блин", "source_lang": "ru", "translation": "Damn"}]       # mild -> none: kept
        preds = [{"translation": t} for t in ("Bloody hell", "Shit", "Oh no", "Ugh")]
        rep = crude_report(rows, ["{}"] * 4, preds)
        self.assertEqual(rep["profanity_kept_rate"], 0.75)
        self.assertEqual(rep["profanity_kept_rate_by_tier"], {"mild": 1.0, "medium": 0.5, "strong": 1.0})
        self.assertEqual(rep["profanity_same_tier_rate"], 0.0)
        self.assertEqual(rep["profanity_same_tier_rate_by_tier"], {"mild": 0.0, "medium": 0.0, "strong": 0.0})
        self.assertEqual(rep["profanity_present_rate"], 0.25)   # only "Shit" is in the crude stem list


class StemTest(unittest.TestCase):
    def test_russian_hits_with_prefixes(self):
        for t in ["Бля, опять", "мне похуй", "заебал уже", "охуеть", "иди нахер", "отъебись", "ну ты жопа",
                  "проебал дедлайн", "ахуеть", "мудила", "до усрачки"]:
            self.assertTrue(is_crude(t, "ru"), t)

    def test_russian_lookalikes_do_not_count(self):
        for t in ["Купи хлеба", "кошки на душе скребут", "животный страх", "сто рублей", "напеки блинов",
                  "проблема", "абляция"]:
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
        self.assertEqual(rep["profanity_present_rate"], 0.0)
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
