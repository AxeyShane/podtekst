"""Merge seed files from different sources into one batch seed file, tagging each seed's source.

    python merge_seeds.py --out seeds_batch6.txt \
        --part synthetic=seeds_batch6_synthetic.txt \
        --part film=seeds_batch6_film.txt \
        --part subs=seeds_batch6_subs.txt \
        --part carryover=seeds_batch5_carryover.txt,seeds_batch6_carryover.txt,stage_a_batch4_next_seeds.txt

Writes --out (one seed per line, what stage_a_generate.py / run_batch.py read) and <out>.meta.jsonl
with {"seed", "source", ...} per seed -- extra fields (film, clip, ...) come from each part's own
<part>.meta.jsonl when it exists. Stage A/B rows are keyed by source_text, so joining on "seed"
later lets verification outcomes be compared by source.

Seeds are deduped on a normalised key (first part listed wins) and interleaved across sources with
a fixed shuffle, so a run the balance guard stops partway still covers every source.
"""
import argparse
import json
import random
from pathlib import Path

from movie_mining.text_utils import normalize_key


def load_part(paths: str, source: str) -> list[dict]:
    rows = []
    for p in (x.strip() for x in paths.split(",") if x.strip()):
        meta_path = Path(p).with_suffix(".meta.jsonl")
        meta = {}
        if meta_path.exists():
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    m = json.loads(line)
                    meta[m["seed"]] = m
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                seed = line.strip()
                rows.append({**meta.get(seed, {}), "seed": seed, "source": source, "file": Path(p).name})
    return rows


def merge(parts: list[tuple[str, list[dict]]], rng: random.Random) -> tuple[list[dict], int]:
    """Dedupe across parts (earlier part wins), then interleave. Returns (rows, duplicates dropped)."""
    seen, kept, dropped = set(), [], 0
    for _, rows in parts:
        for r in rows:
            key = normalize_key(r["seed"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(r)
    rng.shuffle(kept)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", action="append", required=True, help="source=file[,file...]; repeatable")
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()

    parts = []
    for spec in args.part:
        source, _, paths = spec.partition("=")
        parts.append((source, load_part(paths, source)))
    rows, dropped = merge(parts, random.Random(args.seed))
    Path(args.out).write_text("".join(r["seed"] + "\n" for r in rows), encoding="utf-8")
    meta = Path(args.out).with_suffix(".meta.jsonl")
    meta.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {s: sum(r["source"] == s for r in rows) for s, _ in parts}
    print(f"{len(rows)} seeds -> {args.out} (+ {meta.name}); {dropped} duplicates dropped; by source {counts}")


if __name__ == "__main__":
    main()
