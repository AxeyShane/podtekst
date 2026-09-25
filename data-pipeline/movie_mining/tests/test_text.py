"""Model-free tests for the subtitle miner. Run from data-pipeline/:
    python -m unittest discover -s movie_mining/tests -t .
"""
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from movie_mining import mine_subtitles as ms, origin
from movie_mining.cues import _morph, address_register, idiom_hits, ru_fluency_issue
from movie_mining.text_utils import clean_line, film_id_from_ids_line, is_multi_speaker, pair_passes

RU = [
    "- Какого чёрта ты тут делаешь?",
    "<i>Я сегодня очень устал на работе.</i>",
    "Какого чёрта ты тут делаешь?",           # duplicate after cleaning
    "Спасибо.",                               # too short
    "Вы не подскажете, где здесь вокзал?",
    "Субтитры сделаны командой ABC",          # credits
    "- Привет.\n- Привет, как дела?",         # two speakers
]
EN = [
    "What the hell are you doing here?",
    "I'm really tired from work today.",
    "What the hell are you doing here?",
    "Thanks.",
    "Excuse me, where is the station?",
    "Subtitles by team ABC",
    "Hi. Hi, how are you?",
]
IDS = [f"en/2001/{100 + i // 3}/x.xml.gz\tru/2001/{100 + i // 3}/y.xml.gz\t1\t1" for i in range(len(RU))]


def make_zip(tmp: Path) -> Path:
    z = tmp / "en-ru.txt.zip"
    with zipfile.ZipFile(z, "w") as zf:
        # moses files are one pair per line; the multi-speaker cue is on one line in real data
        zf.writestr("OpenSubtitles.en-ru.ru", "\n".join(r.replace("\n", " ") for r in RU) + "\n")
        zf.writestr("OpenSubtitles.en-ru.en", "\n".join(EN) + "\n")
        zf.writestr("OpenSubtitles.en-ru.ids", "\n".join(IDS) + "\n")
    return z


class FakeModels:
    """Bag-of-characters 'embeddings' and canned literal MT."""
    LITERAL = {
        "Какого чёрта ты тут делаешь?": "What devil are you doing here?",
        "Я сегодня очень устал на работе.": "I'm really tired from work today.",
        "Вы не подскажете, где здесь вокзал?": "You will not tell me where the station is here?",
    }

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(8)
            v[0] = 1.0
            v[1 + len(t) % 7] = 0.2
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    def translate(self, texts, direction):
        return [self.LITERAL.get(t, t) for t in texts]


class TextTests(unittest.TestCase):
    def test_clean_and_filters(self):
        self.assertEqual(clean_line("- <i>Привет, как дела?</i> [музыка]"), "Привет, как дела?")
        self.assertTrue(is_multi_speaker("- Привет.\n- Как дела?"))
        self.assertFalse(is_multi_speaker("- Привет, как дела?"))
        self.assertEqual(pair_passes("Спасибо.", "Thanks.")[1], "length")
        self.assertEqual(pair_passes("Я купил новый iPhone вчера вечером", "I bought a new iPhone last night")[0], True)
        self.assertEqual(pair_passes("Субтитры сделаны командой ABC тут", "Subtitles by team ABC here")[1], "credits")
        self.assertEqual(film_id_from_ids_line("en/1999/12345/6.xml.gz\tru/1999/12345/7.xml.gz"), "1999/12345")

    def test_mojibake_and_interjection_filters(self):
        rejected = {("Ќа каком основании?", "On what grounds?"): "mojibake",
                    ("ƒа, да. я сейчас.", "Yes Yes. I am now."): "mojibake",
                    ("Я знаю, что делать.", "I knoƒ what to do."): "mojibake",      # ƒ in EN
                    ("О, о-о-о, ну.", "Oh, oh-oh-oh, well."): "interjection",
                    ("Хе-хе-хе.", "Heh- heh-heh ."): "interjection",
                    ("Не-не-не!", "No-no - no !"): "interjection",
                    ('"Б, А". "Б, А".', '"B , A". "B , A".'): "interjection"}
        for (ru, en), why in rejected.items():
            self.assertEqual(pair_passes(ru, en), (False, why), ru)
        for ru, en in [("Кому ж ещё?", "For whom else?"), ("Ну, ну, ну, не надо.", "Now, now, don't."),
                       ("Встретимся в кафе у Пьера.", "Meet me at the café by Pierre."),
                       ("Сегодня 22 июня!", "Today is June 22!")]:
            self.assertEqual(pair_passes(ru, en), (True, ""), ru)

    def test_cues(self):
        self.assertEqual(address_register("Я тебе говорил"), "ty")
        self.assertEqual(address_register("Вам помочь?"), "vy")
        self.assertIsNone(address_register("Идёт дождь"))
        self.assertIn("какого черта", idiom_hits("Какого чёрта ты тут?"))

    def test_end_to_end_with_fake_models(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pool, stats = ms.collect_pool(ms.iter_pairs(make_zip(tmp)), 100, random.Random(1), 3, 25)
            self.assertEqual(stats["read"], len(RU))
            self.assertEqual(stats["reject_duplicate"], 1)
            self.assertEqual(stats["reject_length"], 1)
            self.assertEqual(stats["reject_credits"], 1)
            self.assertEqual(stats["reject_multi_speaker"], 1)
            self.assertEqual(len(pool), 3)

            cands, s2 = ms.score_pool(pool, FakeModels(), ["ru-en"], min_align=0.5, max_chrf=45, log=lambda *_: None)
            srcs = {c["source"] for c in cands}
            self.assertIn("Какого чёрта ты тут делаешь?", srcs)                 # idiomatic -> divergent
            self.assertNotIn("Я сегодня очень устал на работе.", srcs)          # literal == human
            self.assertEqual(s2["reject_literal_close_ru-en"], 1)

            ranked = ms.rank(cands, min_sem=0.0)
            by_src = {c["source"]: c for c in ranked}
            idiom = by_src["Какого чёрта ты тут делаешь?"]
            self.assertEqual(idiom["divergence_kind"], "local_swap")      # chrF ~67, 2-word swap
            self.assertEqual(idiom["address"], "ty")
            self.assertTrue(idiom["idioms"])
            self.assertEqual(by_src["Вы не подскажете, где здесь вокзал?"]["divergence_kind"], "rewrite")
            sel = ms.select(ranked, target=10, max_per_film=1)
            self.assertEqual(len({c["film"] for c in sel}), len(sel))

            seeds = tmp / "seeds.txt"
            ms.write_outputs(sel, stats, "t", tmp / "out", seeds, seed_count=5)
            self.assertEqual(seeds.read_text(encoding="utf-8").splitlines()[0], sel[0]["source"])
            rec = json.loads((tmp / "out" / "subs_candidates_t.jsonl").read_text(encoding="utf-8").splitlines()[0])
            for key in ("human_translation", "literal_mt", "chrf_literal_vs_human", "score", "film", "bucket"):
                self.assertIn(key, rec)

    @unittest.skipUnless(_morph(), "pymorphy3 not installed")
    def test_fluency(self):
        self.assertEqual(ru_fluency_issue("Мможет я помогу?"), "doubled_capital")
        self.assertIsNone(ru_fluency_issue("Ссора была глупой."))              # real word, doubled letter
        self.assertIsNone(ru_fluency_issue("Вы знакомы с Даниелли?"))          # one unknown name is fine
        self.assertEqual(ru_fluency_issue("Тудым-сюдым шмяк бдыщ"), "unknown_words")

    def test_address_bucket_is_capped_and_unboosted(self):
        def cand(i, ru, chrf):
            return {"ru": ru, "source": f"line {i}", "film": f"2001/{i}", "align_cos": 1.0,
                    "chrf_literal_vs_human": chrf, "novelty": 0.0}
        cands = [cand(i, "Ты где был вчера?", 10.0) for i in range(8)] + \
                [cand(100 + i, "Идёт сильный дождь.", 20.0) for i in range(8)]
        ranked = ms.rank(cands, min_sem=0.0)
        self.assertEqual({c["bucket"] for c in ranked[:8]}, {"address"})          # higher divergence only
        self.assertAlmostEqual(ranked[0]["score"], 0.9)                           # no ты/вы bonus
        sel = ms.select(ranked, target=10, max_per_film=5, address_share=0.3)
        self.assertEqual(sum(c["bucket"] == "address" for c in sel), 3)
        self.assertEqual(sum(c["bucket"] == "general" for c in sel), 7)

    def test_origin_filter(self):
        self.assertEqual(origin.imdb_id("1979/79679"), "tt0079679")
        calls = []

        def fake_fetch(ids):
            calls.append(list(ids))
            return {"tt0000100": ["Q7737"], "tt0000101": ["Q1860"]}             # 102: unknown to Wikidata
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            z = make_zip(tmp)
            keys = origin.film_keys(z)
            self.assertEqual(keys, {"2001/100", "2001/101", "2001/102"})
            cache = tmp / "film_lang.json"
            keep, st = origin.films_with_origin(keys, cache, fetch=fake_fetch, log=lambda *_: None)
            self.assertEqual(keep, {"2001/100"})
            self.assertEqual(st["origin_films_with_language"], 2)
            origin.films_with_origin(keys, cache, fetch=fake_fetch, log=lambda *_: None)
            self.assertEqual(len(calls), 1)                                        # second run served from cache
            pool, stats = ms.collect_pool(ms.iter_pairs(z), 100, random.Random(1), 3, 25, films=keep)
            self.assertEqual(stats["reject_origin"], 4)
            self.assertTrue(all(p["film"] == "2001/100" for p in pool))


if __name__ == "__main__":
    unittest.main()
