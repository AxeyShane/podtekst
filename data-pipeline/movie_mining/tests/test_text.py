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

from movie_mining import mine_subtitles as ms
from movie_mining.cues import address_register, idiom_hits
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
            for key in ("human_translation", "literal_mt", "chrf_literal_vs_human", "score", "film"):
                self.assertIn(key, rec)


if __name__ == "__main__":
    unittest.main()
