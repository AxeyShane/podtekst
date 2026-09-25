"""merge_seeds.py: source tagging, sidecar metadata, cross-source dedupe, interleaving."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import merge_seeds as ms


class MergeSeedsTests(unittest.TestCase):
    def test_tags_sources_carries_metadata_and_dedupes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "syn.txt").write_text("Ты где был?\nКак дела?\n", encoding="utf-8")
            (d / "film.txt").write_text("Как дела ?\nНу ты даёшь!\n", encoding="utf-8")          # dup of syn
            (d / "film.meta.jsonl").write_text(json.dumps({"seed": "Ну ты даёшь!", "film": "afonya",
                                                           "clip": "clips/a.wav"}, ensure_ascii=False),
                                               encoding="utf-8")
            parts = [("synthetic", ms.load_part(str(d / "syn.txt"), "synthetic")),
                     ("film", ms.load_part(str(d / "film.txt"), "film"))]
            rows, dropped = ms.merge(parts, random.Random(0))
        self.assertEqual(dropped, 1)
        by_seed = {r["seed"]: r for r in rows}
        self.assertEqual(set(by_seed), {"Ты где был?", "Как дела?", "Ну ты даёшь!"})
        self.assertEqual(by_seed["Как дела?"]["source"], "synthetic")           # earlier part wins
        self.assertEqual(by_seed["Ну ты даёшь!"]["film"], "afonya")            # sidecar metadata kept
        self.assertEqual(by_seed["Ну ты даёшь!"]["source"], "film")

    def test_interleaves_sources(self):
        parts = [(s, [{"seed": f"{s} {i}", "source": s} for i in range(50)]) for s in ("a", "b")]
        rows, _ = ms.merge(parts, random.Random(1))
        first = {r["source"] for r in rows[:10]}
        self.assertEqual(first, {"a", "b"})                                     # not all of one source first


if __name__ == "__main__":
    unittest.main()
