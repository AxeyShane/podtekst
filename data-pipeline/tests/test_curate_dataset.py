"""curate_dataset: file selection, exact dedupe, near-duplicate grouping, leak-free split (no models)."""
import json
import os
import tempfile
import unittest

import numpy as np

import curate_dataset as cd


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row(text, cat="none", lang="ru"):
    return {"source_lang": lang, "source_text": text, "translation": "t", "has_subtext": cat != "none",
            "category": cat, "nuance_note": "" if cat == "none" else "n"}


class CurateTest(unittest.TestCase):
    def test_final_files_skip_intermediates(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("stage_b_batch1.jsonl", "stage_b_batch5_cowork_resolved.jsonl",
                      "stage_b_batch6b_prefilter_resolved.jsonl", "stage_b_batch3_resolved.jsonl",
                      "stage_b_podcast1.jsonl", "stage_b_batch4_cowork_dropped.jsonl"):
                open(os.path.join(d, n), "w").close()
            names = [os.path.basename(f) for f in cd.final_files(d)]
        self.assertEqual(names, ["stage_b_batch1.jsonl", "stage_b_podcast1.jsonl"])

    def test_exact_duplicates_last_file_wins(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            write(a, [row("Привет!"), row("Как дела?")])
            write(b, [row("  привет! ", "idiom")])
            rows, dups = cd.load_rows([a, b])
        self.assertEqual(dups, 1)
        self.assertEqual(sorted(r["category"] for r in rows), ["idiom", "none"])

    def test_near_pairs_respect_language(self):
        rows = [row("a"), row("b"), row("c", lang="en")]
        emb = np.array([[1.0, 0.0], [0.99, 0.141], [1.0, 0.0]])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        pairs, neigh = cd.near_duplicate_pairs(rows, emb, 0.95)
        self.assertEqual([(i, j) for i, j, _ in pairs], [(0, 1)])
        self.assertEqual(neigh[2], 0)

    def test_split_keeps_groups_together_and_stratifies(self):
        rows = [row(f"s{i}", "sarcasm" if i < 20 else "none") for i in range(40)]
        groups = {i: [i] for i in range(40)}
        groups[0] = [0, 1]; del groups[1]
        test = cd.split_groups(rows, groups, 0.25, seed=1)
        self.assertEqual((0 in test), (1 in test))
        self.assertTrue(4 <= sum(1 for i in test if i < 20) <= 6)
        self.assertTrue(4 <= sum(1 for i in test if i >= 20) <= 6)


if __name__ == "__main__":
    unittest.main()
