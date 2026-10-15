"""
Deterministic checks for batch register1, run on the Stage A clean file before Stage B. No model calls.

    python register_checks.py --in stage_a_register1.clean.jsonl --seeds seeds_register1.txt \
        --kept stage_a_register1.kept.jsonl

Reports: rows with a profane source; how many translations keep the force tier (model/podtekst_sft/crude.py
tiers), per tier and per generator; masked translations (must be 0); duplicate rate; length stats.
"kept" is a translation within one tier of the source: exact matching is too strict across RU/EN ("shit" ->
«херня», «какого хрена» -> "what the hell" are correct). Only jumps of two or more tiers (strong -> mild or
none, medium -> none) are flagged. "same tier" is the strict number, reported next to it. Exits 2 when the
kept rate is under --min-kept (default 0.85): stop and look before Stage B.

--kept writes the input with, per sentence, only the kept candidates -- so the prefilter cannot pick a
flagged one -- or all candidates when none is kept (Cowork decides those).
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from podtekst_sft.crude import TIER_NAMES, masked, other, tier, tier_kept  # noqa: E402
from deterministic_checks import MASK_RE, load_jsonl  # noqa: E402
from generate_register_seeds import existing_texts, norm  # noqa: E402


def pct(a, b):
    return f"{a}/{b} ({a / b:.1%})" if b else "0/0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="Stage A clean JSONL")
    ap.add_argument("--seeds", required=True, help="Seed file of the batch (duplicate check)")
    ap.add_argument("--kept", default=None, help="Write the tier-filtered candidates here")
    ap.add_argument("--min-kept", type=float, default=0.85)
    args = ap.parse_args()

    rows = load_jsonl(args.infile)
    for r in rows:
        r["_src_tier"] = tier(r["source_text"], r["source_lang"])
        r["_tr_tier"] = tier(r.get("translation", ""), other(r["source_lang"]))
    prof = [r for r in rows if r["_src_tier"]]
    same = [r for r in prof if r["_tr_tier"] == r["_src_tier"]]
    print(f"Candidate rows: {len(rows)} ({len({r['source_text'] for r in rows})} sentences); "
          f"profane source: {len(prof)}")
    ok = [r for r in prof if tier_kept(r["_tr_tier"], r["_src_tier"])]
    sents = {r["source_text"] for r in prof}
    print(f"Kept (within one tier), per candidate: {pct(len(ok), len(prof))}")
    print(f"Kept (within one tier), per sentence (any candidate keeps it): "
          f"{pct(len({r['source_text'] for r in ok}), len(sents))}")
    print(f"Same tier (strict), per candidate: {pct(len(same), len(prof))}")
    print(f"Flagged (jump of two or more tiers): {pct(len(prof) - len(ok), len(prof))}")
    print(f"Softened: {pct(sum(r['_tr_tier'] < r['_src_tier'] for r in prof), len(prof))}   "
          f"harsher: {pct(sum(r['_tr_tier'] > r['_src_tier'] for r in prof), len(prof))}")
    for t, name in TIER_NAMES.items():
        for lang in ("ru", "en"):
            sub = [r for r in prof if r["_src_tier"] == t and r["source_lang"] == lang]
            print(f"  {lang} {name:6} kept {pct(sum(tier_kept(r['_tr_tier'], t) for r in sub), len(sub))}   "
                  f"same tier {pct(sum(r['_tr_tier'] == t for r in sub), len(sub))}")
    for m in sorted({r.get("_generator_model") for r in prof}):
        sub = [r for r in prof if r.get("_generator_model") == m]
        print(f"  {m:32} kept {pct(sum(tier_kept(r['_tr_tier'], r['_src_tier']) for r in sub), len(sub))}   "
              f"same tier {pct(sum(r['_tr_tier'] == r['_src_tier'] for r in sub), len(sub))}")

    mask = [r for r in rows if masked(r.get("translation", "")) or MASK_RE.search(r.get("translation", ""))]
    print(f"Masked translations: {len(mask)}")
    for r in mask[:5]:
        print(f"    {r['translation']}")

    seeds = [l.strip() for l in open(args.seeds, encoding="utf-8") if l.strip()]
    normed = Counter(norm(s) for s in seeds)
    old = existing_texts(args.seeds)
    print(f"Duplicates: {sum(n - 1 for n in normed.values())} within {len(seeds)} seeds, "
          f"{sum(k in old for k in normed)} against earlier seeds / dataset / probes")

    words = [len(s.split()) for s in seeds]
    ratio = [len(r.get("translation", "")) / max(1, len(r["source_text"])) for r in rows]
    print(f"Seed length (words): min {min(words)}, median {statistics.median(words)}, max {max(words)}; "
          f"outside 4-18: {sum(not 4 <= w <= 18 for w in words)}")
    print(f"Translation/source length ratio: median {statistics.median(ratio):.2f}, "
          f"under 0.5: {sum(x < 0.5 for x in ratio)}, over 2.0: {sum(x > 2 for x in ratio)}")

    if args.kept:
        groups = defaultdict(list)
        for r in rows:
            groups[r["source_text"]].append(r)
        out, none_kept = [], 0
        for cands in groups.values():
            good = [r for r in cands if tier_kept(r["_tr_tier"], r["_src_tier"])]
            none_kept += not good
            out += good or cands
        with open(args.kept, "w", encoding="utf-8", newline="\n") as f:
            for r in out:
                r = {k: v for k, v in r.items() if k not in ("_src_tier", "_tr_tier")}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"--kept: {len(out)} candidates -> {args.kept}; {none_kept} sentences have no kept "
              f"candidate (all candidates kept for Cowork)")

    rate = len(ok) / len(prof) if prof else 1.0
    if rate < args.min_kept:
        print(f"\nSTOP: kept rate (within one tier) {rate:.1%} is under {args.min_kept:.0%}.")
        sys.exit(2)


if __name__ == "__main__":
    main()
