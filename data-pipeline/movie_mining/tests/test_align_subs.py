"""align_subs tests: subtitle cleaning and the alignment DP with a fake similarity (no models)."""
import tempfile
import unittest
from pathlib import Path

from movie_mining import align_subs

RU = """1
00:00:00,040 --> 00:00:00,960
25.000000

2
00:00:00,120 --> 00:00:00,960
— Что ты на меня смотришь?

3
00:00:01,000 --> 00:00:02,000
Юра, я с кем разговариваю?

4
00:00:03,400 --> 00:00:03,680
(хлопок)

5
00:00:04,000 --> 00:00:05,400
Эй, бродяги, хватит воевать. ♪ ля-ля ♪
"""
EN = """1
00:00:00,000 --> 00:00:02,000
Why are you looking at me? Yura, I'm talking to you.

2
00:00:04,000 --> 00:00:05,000
Hey, drifters, stop fighting!
"""

def fake_sim(en, ru):
    pairs = {("Why", "Что ты"), ("Yura", "Юра"), ("drifters", "бродяги")}
    hits = sum(1 for e, r in pairs if e in en and r in ru)
    total = sum(1 for e, _ in pairs if e in en)
    return hits / max(1, total)

class AlignSubsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "v.ru.srt").write_text(RU, encoding="utf-8")
        (self.tmp / "v.en.srt").write_text(EN, encoding="utf-8")

    def test_read_srt_cleans(self):
        ru = align_subs.read_srt(self.tmp / "v.ru.srt")
        self.assertEqual([u["text"] for u in ru],
                         ["Что ты на меня смотришь?", "Юра, я с кем разговариваю?", "Эй, бродяги, хватит воевать."])

    def test_align_groups_two_russian_cues_with_one_english(self):
        ru = align_subs.read_srt(self.tmp / "v.ru.srt")
        en = align_subs.read_srt(self.tmp / "v.en.srt")
        rows = align_subs.align(en, ru, fake_sim)
        self.assertEqual([r[0] for r in rows], ["1:2", "1:1"])
        self.assertEqual(rows[0][2], "1+2")

if __name__ == "__main__":
    unittest.main()
