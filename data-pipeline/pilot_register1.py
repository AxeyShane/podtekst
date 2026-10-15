"""
Generator pilot for batch register1: how each model handles profanity, before choosing who writes the seeds.

    python pilot_register1.py                       # every Stage A generator in config/models.json
    python pilot_register1.py --models a/b,c/d      # any OpenRouter slugs (e.g. candidates outside the roster)

Per model: 40 seed-writing requests (generate_register_seeds.py prompts, 3 lines each, tiers weighted to
strong) and 40 Stage A translations of pilot_register1_sentences.txt with the "Profanity and slang register"
rules. Reports refusal rate and, per output, kept (same tier) / softened (lower tier or none) / masked, with
the strong tier on its own. Raw replies go to pilot_register1_results.jsonl.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import generate_register_seeds as reg
import generate_seeds as gs
import stage_a_generate as sa
from podtekst_sft.crude import masked, other, tier  # path set up by generate_register_seeds

RULES = "Profanity and slang register"
CELLS = [c for lang in ("ru", "en") for c in
         [{"lang": lang, "tier": t, "function": f} for t, k in ((3, 10), (2, 5), (1, 5))
          for f in list(reg.FUNCTIONS)[:k]]]                        # 20 per language, 40 in all


def seed_pilot(m, api_key):
    out = []
    for c in CELLS:
        system = reg.SYSTEM.format(lang_name=gs.LANG_NAMES[c["lang"]], tier=reg.TIER_DESC[c["lang"], c["tier"]],
                                   function=reg.FUNCTIONS[c["function"]], n=3)
        try:
            lines = gs.call_model(c["function"], c["lang"], 3, m["slug"], api_key, provider=m.get("provider"),
                                  api_model=m.get("api_model"), system=system)
            refused = not lines or sa.looks_like_refusal(lines[0])
            out.append({**c, "kind": "seed", "model": m["slug"], "refused": refused, "lines": [] if refused else lines})
        except Exception as e:
            out.append({**c, "kind": "seed", "model": m["slug"], "refused": sa.looks_like_refusal(str(e)) or
                        "REFUSED" in str(e), "error": str(e)[:300], "lines": []})
    return out


def translate_pilot(m, api_key, sentences):
    out = []
    for i, s in enumerate(sentences):
        lang = "ru" if i < 20 else "en"
        row = {"kind": "translate", "model": m["slug"], "lang": lang, "tier": tier(s, lang), "source": s}
        try:
            r = sa.call_model_with_retry(s, m["slug"], api_key, provider=m.get("provider"), api_model=m.get("api_model"))
            out.append({**row, "refused": False, "translation": r.get("translation", "")})
        except sa.Refused as e:
            out.append({**row, "refused": True, "error": str(e)[:300]})
        except Exception as e:
            out.append({**row, "refused": False, "error": str(e)[:300], "translation": None})
    return out


def scores(rows):
    """{scope: (n, refusal, kept, softened, masked)} over seed lines or translations, all tiers and strong only."""
    res = {}
    for kind in ("seed", "translate"):
        rs = [r for r in rows if r["kind"] == kind]
        for scope, pick in (("all", rs), ("strong", [r for r in rs if r["tier"] == 3])):
            ref = sum(r["refused"] for r in pick) / len(pick) if pick else 0
            outs = ([(l, r["lang"], r["tier"]) for r in pick for l in r.get("lines", [])] if kind == "seed" else
                    [(r["translation"], other(r["lang"]), r["tier"]) for r in pick if r.get("translation")])
            n = len(outs) or 1
            res[kind, scope] = (len(outs), ref, sum(tier(t, l) == want for t, l, want in outs) / n,
                                sum(tier(t, l) < want for t, l, want in outs) / n, sum(masked(t) for t, _, _ in outs) / n)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated slugs; default: the Stage A roster")
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--out", default="pilot_register1_results.jsonl")
    args = ap.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("Set OPENROUTER_API_KEY first.")
    roster = json.load(open(args.config, encoding="utf-8"))["stage_a_generators"]
    models = ([{"slug": s.strip()} for s in args.models.split(",")] if args.models else roster)
    sa.EXTRA_RULES.append(sa.guideline_section(RULES))
    sentences = [l.strip() for l in open("pilot_register1_sentences.txt", encoding="utf-8") if l.strip()]

    with ThreadPoolExecutor(max_workers=2 * len(models)) as pool:
        futs = [pool.submit(seed_pilot, m, api_key) for m in models] + \
               [pool.submit(translate_pilot, m, api_key, sentences) for m in models]
        rows = [r for f in futs for r in f.result()]
    with open(args.out, "a", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    print(f"\n{'model':32} {'task':10} {'scope':7} {'n':>4} {'refusal':>8} {'kept':>6} {'softened':>9} {'masked':>7}")
    for m in models:
        for (kind, scope), (n, ref, kept, soft, mask) in scores([r for r in rows if r["model"] == m["slug"]]).items():
            print(f"{m['slug']:32} {kind:10} {scope:7} {n:4} {ref:8.0%} {kept:6.0%} {soft:9.0%} {mask:7.0%}")
    errors = [r for r in rows if r.get("error") and not r["refused"]]
    print(f"\n{len(errors)} calls failed for other reasons (see {args.out})")


if __name__ == "__main__":
    main()
