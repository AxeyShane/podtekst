"""Curate the verified Stage B rows into a fine-tuning set: dedupe, near-duplicates, leak-free split.

    python curate_dataset.py                                  # LaBSE, writes dataset/
    python curate_dataset.py --model google/embeddinggemma-2  # any sentence-transformers model
    python curate_dataset.py --near 0.93 --test-share 0.1

Steps:
  1. Load every final stage_b_<batch>.jsonl (not the cowork/prefilter/dropped intermediates).
  2. Exact duplicates (same source text after lowercasing and whitespace folding): keep the
     row from the latest batch file.
  3. Near-duplicates: embed source texts; pairs at cosine >= --near (same source language)
     are joined into groups. Groups stay together in one split, so a test sentence never has a
     near-copy in training. With --drop-near, only one row per group is kept for training.
  4. Split: about --test-share of groups go to test, stratified by category so each category
     appears in both splits in the same proportion.
  5. Report: counts by category, source language and batch, the near-duplicate groups, and the
     sentences with the most close neighbours (the most templated patterns).

Outputs (git-ignored, they hold dataset text): dataset/train.jsonl, dataset/test.jsonl,
dataset/near_duplicates.tsv, dataset/curation_report.json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

SKIP = re.compile(r"cowork|prefilter|dropped|resolved|before_gap|repolished")
LABEL_KEYS = ("source_lang", "source_text", "translation", "has_subtext", "category", "nuance_note")


def final_files(folder: str = ".") -> list[str]:
    return [f for f in sorted(glob.glob(os.path.join(folder, "stage_b_*.jsonl")))
            if not SKIP.search(os.path.basename(f))]


def key(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def load_rows(files: list[str]) -> tuple[list[dict], int]:
    """Rows from all files; for exact-duplicate source texts the last file wins."""
    by_key: dict[str, dict] = {}
    total = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("needs_human_review") or r.get("_stage_b_needs_human_review"):
                    continue
                total += 1
                row = {k: r[k] for k in LABEL_KEYS if k in r}
                row["_batch_file"] = os.path.basename(f)
                for extra in ("speaker_gender", "addressee_gender", "_speaker_gender", "_addressee_gender"):
                    if extra in r:
                        row[extra.lstrip("_")] = r[extra]
                by_key[key(r["source_text"])] = row
    return list(by_key.values()), total - len(by_key)


class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def near_duplicate_pairs(rows: list[dict], emb, threshold: float, block: int = 1024) -> tuple[list[tuple], list[int]]:
    """(i, j, cos) for same-language pairs at cos >= threshold, plus each row's neighbour count at
    cos >= threshold - 0.1 (a softer 'how templated is this' signal). emb: L2-normalised rows."""
    import numpy as np
    n = len(rows)
    lang = np.array([r.get("source_lang", "") for r in rows])
    pairs, neighbours = [], [0] * n
    for s in range(0, n, block):
        sims = emb[s:s + block] @ emb.T
        for bi in range(sims.shape[0]):
            i = s + bi
            row = sims[bi]
            same = lang == lang[i]
            soft = np.where((row >= threshold - 0.1) & same)[0]
            neighbours[i] = int(len(soft) - 1)
            for j in np.where((row >= threshold) & same)[0]:
                if j > i:
                    pairs.append((i, int(j), float(row[j])))
    return pairs, neighbours


def split_groups(rows: list[dict], groups: dict[int, list[int]], test_share: float, seed: int) -> set[int]:
    """Pick groups for test so each category gets about test_share of its rows. Returns test row ids."""
    rng = random.Random(seed)
    by_cat: dict[str, list[list[int]]] = defaultdict(list)
    for members in groups.values():
        cat = Counter(rows[m]["category"] for m in members).most_common(1)[0][0]
        by_cat[cat].append(members)
    test: set[int] = set()
    for cat, gs in sorted(by_cat.items()):
        rng.shuffle(gs)
        target = round(test_share * sum(len(g) for g in gs))
        taken = 0
        for g in gs:
            if taken >= target:
                break
            test.update(g); taken += len(g)
    return test


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--near", type=float, default=0.93, help="Cosine at or above which two sources are near-duplicates")
    ap.add_argument("--test-share", type=float, default=0.1)
    ap.add_argument("--drop-near", action="store_true", help="Keep one row per near-duplicate group in train")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="dataset")
    args = ap.parse_args()

    files = final_files()
    rows, exact_dups = load_rows(files)
    print(f"{len(files)} files, {len(rows)} unique rows ({exact_dups} exact duplicates removed)")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    emb = model.encode([r["source_text"] for r in rows], batch_size=128, normalize_embeddings=True,
                       show_progress_bar=True)
    pairs, neighbours = near_duplicate_pairs(rows, emb, args.near)

    uf = UnionFind(len(rows))
    for i, j, _ in pairs:
        uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    test_ids = split_groups(rows, groups, args.test_share, args.seed)

    train, test = [], []
    dropped_near = 0
    for root, members in groups.items():
        if args.drop_near and len(members) > 1 and not (set(members) & test_ids):
            keep = [max(members, key=lambda m: rows[m].get("has_subtext", False))]
            dropped_near += len(members) - 1
        else:
            keep = members
        for m in keep:
            (test if m in test_ids else train).append(rows[m])

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)
    with open(os.path.join(args.out_dir, "near_duplicates.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cosine", "category_a", "category_b", "source_a", "source_b"])
        for i, j, c in sorted(pairs, key=lambda p: -p[2]):
            w.writerow([round(c, 3), rows[i]["category"], rows[j]["category"], rows[i]["source_text"], rows[j]["source_text"]])

    multi = [g for g in groups.values() if len(g) > 1]
    templated = sorted(range(len(rows)), key=lambda i: -neighbours[i])[:25]
    report = {
        "model": args.model, "near_threshold": args.near, "files": files,
        "rows_unique": len(rows), "exact_duplicates_removed": exact_dups,
        "near_duplicate_pairs": len(pairs), "near_duplicate_groups": len(multi),
        "rows_in_near_groups": sum(len(g) for g in multi),
        "near_groups_with_mixed_labels": sum(1 for g in multi if len({rows[m]["category"] for m in g}) > 1),
        "dropped_as_near_duplicates": dropped_near,
        "train": len(train), "test": len(test),
        "train_by_category": dict(Counter(r["category"] for r in train)),
        "test_by_category": dict(Counter(r["category"] for r in test)),
        "by_source_lang": dict(Counter(r.get("source_lang") for r in rows)),
        "by_batch_file": dict(Counter(r["_batch_file"] for r in rows)),
        "most_templated": [{"source": rows[i]["source_text"], "category": rows[i]["category"],
                            "neighbours": neighbours[i]} for i in templated],
    }
    with open(os.path.join(args.out_dir, "curation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"near-duplicate pairs {len(pairs)} in {len(multi)} groups "
          f"({report['near_groups_with_mixed_labels']} with mixed labels)")
    print(f"train {len(train)}  test {len(test)}  -> {args.out_dir}/")
    print("train:", report["train_by_category"]); print("test: ", report["test_by_category"])


if __name__ == "__main__":
    main()
