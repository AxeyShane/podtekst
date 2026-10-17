"""
Seeds for batch idioms1: idioms whose meaning is lost word for word, RU->EN and EN->RU.
See seeds_idioms1_spec.md for why and for the targets; the idioms live in idioms1_list.txt.

    python generate_idiom_seeds.py --check                                   # validate the list, no calls
    python generate_idiom_seeds.py --split train --out seeds_idioms1.txt
    python generate_idiom_seeds.py --split test  --out seeds_idioms1_test.txt
    python generate_idiom_seeds.py --split train --out x.txt --dry-run       # print the grid, no calls

One cell per idiom. Train idioms get USES["idiom"] uses; idioms marked L also get literal-use
negatives; equiv idioms (same image in both languages) get "none" negatives. Every kept line is
4-20 words, in the right script, contains the idiom (its key regex), and is new against every
seeds_*.txt, the dataset and the eval probes. Writes <out> plus <out>.meta.jsonl
({seed, lang, idiom, kind, split}).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import generate_seeds as gs
from generate_register_seeds import CHATTER, existing_texts, norm

LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idioms1_list.txt")
USES = {"idiom": 2, "test": 2, "equiv": 3, "literal": 2}

# Idioms of the 53-sentence comparison with DeepL / Yandex / Google (in dataset/test.jsonl and the
# probes). They are never seeded, so a gain there comes from generalisation, not from memorising them.
BENCHMARK_EXCLUDE = [
    r"веша\w* нос", r"в рубашке", r"на носу.*заруби|заруби\w* себе", r"чеса\w* язык", r"ногу отдавил",
    r"дело в шляпе", r"почву из.под ног", r"\bдостал",
    r"red.handed", r"chip on (his|her|my|your) shoulder", r"wit'?s end", r"fed up", r"loose cannon",
    r"in the bag", r"rug out", r"right on cue",
]

IDIOM_SYSTEM = """You write seed sentences for a dataset that teaches a phone keyboard to translate idioms
between Russian and English by meaning, not word for word. Write ONLY in {lang_name}.

Every sentence must use this idiom in its idiomatic sense: {idiom}  (meaning: {meaning})

Write exactly {n} sentences, one per line, no numbering, no quotes, no commentary. Each is 4-20 words, a
natural chat or spoken message a real person would type to a friend, family member or colleague. Vary the
situation, the tense and the grammatical form of the idiom; at most one line may be a question. No personal
names, no real people, no lines from films, songs or books."""

LITERAL_SYSTEM = """You write seed sentences for a dataset that teaches a phone keyboard NOT to see an idiom
where there is none. Write ONLY in {lang_name}.

Every sentence must contain the words of "{idiom}" but in their plain, literal, physical sense, so that a
word-for-word translation is correct and nothing is lost. (The idiom means "{meaning}"; do NOT use that
meaning.)

Write exactly {n} sentences, one per line, no numbering, no quotes, no commentary. Each is 4-20 words, a
natural chat message. No personal names, no real people, no lines from films, songs or books."""


def load_list(path=LIST) -> list[dict]:
    rows, sec = [], None
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if line.startswith("## "):
            split, kind, lang = line[3:].split()
            sec = {"split": split, "kind": kind, "lang": lang}
            continue
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) not in (3, 4) or sec is None:
            sys.exit(f"{path}:{i}: bad line: {line}")
        key = parts[0].replace("ё", "е").replace("/", "|")
        rows.append({**sec, "key": re.compile(key), "key_src": parts[0], "idiom": parts[1],
                     "meaning": parts[2], "literal": len(parts) == 4 and parts[3] == "L", "line": i})
    return rows


def has(row, text: str) -> bool:
    return bool(row["key"].search(text.lower().replace("ё", "е")))


def grid(rows, split: str) -> list[dict]:
    cells = []
    for r in rows:
        if r["split"] != split:
            continue
        lang_name = gs.LANG_NAMES[r["lang"]]
        if r["kind"] == "idiom":
            n = USES["idiom" if split == "train" else "test"]
            cells.append({**r, "use": "idiom", "n": n,
                          "system": IDIOM_SYSTEM.format(lang_name=lang_name, idiom=r["idiom"], meaning=r["meaning"], n="{n}")})
        if r["kind"] == "equiv":
            cells.append({**r, "use": "equiv", "n": USES["equiv"],
                          "system": IDIOM_SYSTEM.format(lang_name=lang_name, idiom=r["idiom"], meaning=r["meaning"], n="{n}")})
        if r["literal"] and split == "train":
            cells.append({**r, "use": "literal", "n": USES["literal"],
                          "system": LITERAL_SYSTEM.format(lang_name=lang_name, idiom=r["idiom"], meaning=r["meaning"], n="{n}")})
    return cells


def accept(line: str, cell: dict) -> str | None:
    if CHATTER.search(line):
        return "chatter"
    if not 4 <= len(line.split()) <= 20:
        return "length"
    cyr = bool(re.search(r"[а-яё]", line, re.I))
    if (cell["lang"] == "ru") != cyr:
        return "language"
    if not has(cell, line):
        return "no idiom"
    if any(re.search(p, line.lower()) for p in BENCHMARK_EXCLUDE):
        return "benchmark idiom"
    return None


def corpus() -> dict[str, list[str]]:
    out = {}
    for p in glob.glob("dataset/*.jsonl") + glob.glob("../model/probes/*.jsonl"):
        out[p] = [json.loads(l)["source_text"] for l in open(p, encoding="utf-8") if l.strip()]
    return out


def check(rows):
    """Validate the list: counts, keys that match their own idiom, collisions, leakage."""
    ok = True
    from collections import Counter
    c = Counter((r["split"], r["kind"], r["lang"]) for r in rows)
    for k, n in sorted(c.items()):
        print(f"  {' '.join(k):18} {n}")
    print(f"  literal-capable: {sum(r['literal'] for r in rows)}")
    names = Counter((r["lang"], r["idiom"].lower()) for r in rows)
    for k, n in names.items():
        if n > 1:
            print(f"DUPLICATE idiom {k}"); ok = False
    for r in rows:   # another idiom's key must not fire on this idiom's text
        for o in rows:
            if o is not r and o["lang"] == r["lang"] and has(o, r["idiom"]) and o["split"] != r["split"]:
                print(f"TRAIN/TEST COLLISION: key {o['key_src']!r} (line {o['line']}) matches {r['idiom']!r}"); ok = False
        for p in BENCHMARK_EXCLUDE:
            if re.search(p, r["idiom"].lower()):
                print(f"BENCHMARK idiom in list: {r['idiom']!r}"); ok = False
    texts = corpus()
    test_texts = [t for p, ts in texts.items() if "train" not in p for t in ts]
    train_texts = texts.get(os.path.join("dataset", "train.jsonl"), [])
    seeds = []
    for p in glob.glob("seeds_*.txt"):
        if "idioms1" not in p:
            seeds += [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    for r in rows:
        in_test = [t for t in test_texts if has(r, t)]
        if r["split"] == "train" and in_test:
            print(f"in eval sets (train idiom, line {r['line']}): {r['idiom']!r} -> {in_test[:2]}"); ok = False
        if r["split"] == "test":
            leaks = [t for t in train_texts + seeds if has(r, t)]
            if leaks:
                print(f"TEST idiom already in train/seeds (line {r['line']}): {r['idiom']!r} -> {leaks[:2]}"); ok = False
    covered = sum(any(has(r, t) for t in train_texts) for r in rows if r["split"] == "train")
    print(f"  train idioms already present in train.jsonl at least once: {covered}")
    print("OK" if ok else "PROBLEMS above")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"])
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", default=None, help="OpenRouter slug; default: config seed_generator_model")
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--max-calls-per-cell", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_list()
    if args.check:
        sys.exit(0 if check(rows) else 1)
    if not (args.split and args.out):
        ap.error("--split and --out are required unless --check")
    cells = grid(rows, args.split)
    from collections import Counter
    by = Counter()
    for c in cells:
        by[c["lang"], c["use"]] += c["n"]
    for (lang, use), n in sorted(by.items()):
        print(f"  {lang} {use:8} {n}")
    print(f"{len(cells)} cells, {sum(c['n'] for c in cells)} seeds")
    if args.dry_run:
        print(cells[0]["system"].replace("{n}", str(cells[0]["n"])))
        return

    api_key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("Set OPENROUTER_API_KEY first.")
    config = json.load(open(args.config, encoding="utf-8"))
    cfg = config.get("seed_generator_model", {})
    if args.model:
        cfg = next((g for g in config["stage_a_generators"] if g["slug"] == args.model), {"slug": args.model})
    slug = cfg.get("slug", gs.DEFAULT_MODEL)
    provider, api_model = cfg.get("provider"), cfg.get("api_model")
    seen = existing_texts(args.out)
    print(f"model {slug}; avoiding {len(seen)} existing texts")

    kept, rejects, short = [], Counter(), []
    for c in cells:
        got = []
        for _ in range(args.max_calls_per_cell):
            if len(got) >= c["n"]:
                break
            want = c["n"] + 2
            try:
                lines = gs.call_with_retry("idiom", c["lang"], want, slug, api_key, provider=provider,
                                           api_model=api_model, system=c["system"].replace("{n}", str(want)))
            except Exception as e:
                print(f"    {c['idiom']}: call failed ({e})")
                continue
            for line in lines:
                why = accept(line, c) or ("duplicate" if norm(line) in seen else None)
                if why:
                    rejects[why] += 1
                    continue
                seen.add(norm(line))
                got.append(line)
                if len(got) >= c["n"]:
                    break
            time.sleep(0.3)
        if len(got) < c["n"]:
            short.append((c["lang"], c["use"], c["idiom"], len(got), c["n"]))
        kept += [{"seed": s, "lang": c["lang"], "idiom": c["idiom"], "kind": c["use"], "split": args.split}
                 for s in got]
        print(f"  {c['lang']} {c['use']:8} {c['idiom'][:40]:40} {len(got)}/{c['n']}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(k["seed"] + "\n" for k in kept)
    with open(args.out.removesuffix(".txt") + ".meta.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(k, ensure_ascii=False) + "\n" for k in kept)
    print(f"\nWrote {len(kept)} seeds to {args.out}. Rejected lines: {dict(rejects)}")
    for s in short:
        print(f"  short: {s}")


if __name__ == "__main__":
    main()
