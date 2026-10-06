"""Emotion-word mining (--emotions): lexicon matching, pool filter, no-divergence scoring, caps.
No models: embeddings and MT are faked, chrF is patched."""
import random
import unittest
from unittest import mock

import numpy as np

from movie_mining import mine_subtitles as ms
from movie_mining.cues import emotion_hits


class FakeModels:
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])      # every pair "aligned", every literal "same meaning"

    def translate(self, texts, direction):
        return ["I'm so offended." if "обидно" in t else "Literal." for t in texts]


class EmotionHitsTest(unittest.TestCase):
    def test_inflected_forms_match(self):
        self.assertEqual(emotion_hits("Она на меня обиделась."), ["obida"])
        self.assertEqual(emotion_hits("Тоска зелёная."), ["toska"])
        self.assertEqual(emotion_hits("Мне как-то не по себе."), ["ne_po_sebe"])

    def test_near_misses_do_not_match(self):
        for line in ("Поедем в Тоскану.", "Жаль, что так вышло.", "Он жалкий тип.", "Он умилостивил бога."):
            self.assertEqual(emotion_hits(line), [], line)


class EmotionModeTest(unittest.TestCase):
    PAIRS = [("Мне так обидно, что ты не пришёл.", "It hurts that you didn't come.", "f1"),
             ("Я сегодня очень устал на работе.", "I'm really tired from work today.", "f1"),
             ("Тоска такая, хоть волком вой.", "I'm so miserable I could howl.", "f2")]

    def test_pool_keeps_only_emotion_lines(self):
        pool, stats = ms.collect_pool(iter(self.PAIRS), 100, random.Random(0), 3, 25,
                                      keep=lambda ru: bool(emotion_hits(ru)))
        self.assertEqual(sorted(p["film"] for p in pool), ["f1", "f2"])
        self.assertEqual(stats["reject_not_wanted"], 1)

    def test_scoring_keeps_literal_lines_when_divergence_not_required(self):
        pool = [{"ru": ru, "en": en, "film": f} for ru, en, f in self.PAIRS if emotion_hits(ru)]
        with mock.patch.object(ms, "_chrf", return_value=90.0):        # literal MT ~ human line
            strict, _ = ms.score_pool(pool, FakeModels(), ["ru-en"], 0.5, 45, log=lambda *_: None)
            loose, _ = ms.score_pool(pool, FakeModels(), ["ru-en"], 0.5, 45, log=lambda *_: None,
                                     require_divergence=False)
        self.assertEqual(strict, [])
        self.assertEqual(len(loose), 2)
        self.assertEqual({c["divergence_kind"] for c in loose}, {"literal"})
        ranked = ms.rank(loose, 0.5, emotions=True)
        self.assertEqual({c["bucket"] for c in ranked}, {"emotion"})

    def test_per_emotion_cap(self):
        ranked = [{"source": f"обида {i}", "film": f"f{i}", "emotions": ["obida"], "bucket": "emotion"} for i in range(5)]
        ranked += [{"source": "тоска", "film": "fx", "emotions": ["toska"], "bucket": "emotion"}]
        out = ms.select(ranked, 10, 15, max_per_emotion=2)
        self.assertEqual([c["emotions"][0] for c in out], ["obida", "obida", "toska"])

class EmotionKeepTest(unittest.TestCase):
    def test_only_and_exclude(self):
        import json
        import os
        import tempfile
        from movie_mining import mine_subtitles as ms
        keep = ms.emotion_keep({"obida", "toska"})
        self.assertTrue(keep("Ты его обидела?"))
        self.assertFalse(keep("Мне очень стыдно."))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ru": "Ты его обидела?"}, ensure_ascii=False) + "\n")
            keep2 = ms.emotion_keep({"obida"}, ms.load_exclude_keys([p]))
        self.assertFalse(keep2("Ты его обидела?"))
        self.assertTrue(keep2("Он меня обидел."))
        self.assertTrue(ms.emotion_keep()("Мне очень стыдно."))


if __name__ == "__main__":
    unittest.main()
