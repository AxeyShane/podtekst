"""
Seeds for batch register1: swearing and current slang, by force tier x function x language.
See seeds_register1_spec.md for why and for the targets. Same OpenRouter call as generate_seeds.py,
with a register prompt per grid cell instead of a category brief.

    python generate_register_seeds.py --split train --out seeds_register1.txt
    python generate_register_seeds.py --split test  --out seeds_register1_test.txt   # independent held-out seeds
    python generate_register_seeds.py --split train --out x.txt --dry-run            # print the grid, no calls

Every kept line is 4-18 words, carries profanity of exactly the requested tier (model/podtekst_sft/crude.py
tiers), is unmasked, has no slur, and is new against every seeds_*.txt, the current dataset and the eval
probes. Writes <out> plus <out>.meta.jsonl ({seed, lang, tier, function, split}).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import generate_seeds as gs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from podtekst_sft.crude import TIER_NAMES, masked, tier  # noqa: E402

TIER_DESC = {
    ("ru", 3): "strong mat: words from the хуй / пизда / ебать / блядь / сука families (пиздец, нахуй, охуеть, "
               "заебал, ёбаный, блядь, сука, похуй, мудак)",
    ("ru", 2): "medium, not mat: хрен, хер, жопа, задница, сволочь, говно, дерьмо, срать, хитрожопый. "
               "No хуй / пизд / еб / бля / сука words",
    ("ru", 1): "mild: блин, чёрт, фиг (пофиг, нафиг, офигеть), капец. Nothing stronger",
    ("en", 3): "strong: fuck, fucking, shit, bullshit, motherfucker, asshole",
    ("en", 2): "medium: ass, bastard, dick, prick, piss, pissed, bloody, bollocks, goddamn. No fuck / shit",
    ("en", 1): "mild: damn, crap, hell, heck, freaking. Nothing stronger",
}
FUNCTIONS = {
    "interjection": "an interjection of surprise or annoyance (Блин, опять дождь / Damn, it's raining again)",
    "intensifier": "the swear word intensifying an adjective, adverb or verb (пиздец как холодно / it's fucking cold)",
    "insult_person": "an insult aimed at a person (ты такой мудак / you absolute bastard)",
    "insult_thing": "an insult aimed at a thing or situation (этот ёбаный принтер / this shitty printer)",
    "dismissal": "dismissal or not caring (мне похуй / забей нахер / I don't give a shit)",
    "idiom": "an idiomatic expression built on the swear word whose literal words mislead "
             "(хуй знает, какого хрена, до пизды, через жопу / fuck knows, ass backwards, who gives a damn)",
    "banter": "friendly banter between close friends, affectionate rather than hostile "
              "(ну ты и сучара, красава / you lucky bastard)",
    "sarcasm": "swearing inside a sarcastic remark (Охуенно, опять пробка / Oh fucking great, another meeting)",
    "formal_vs_informal": "swearing where the setting matters: half the lines in a work or formal setting (to or "
                          "about a boss, client, teacher), half to a close friend, so the register contrast is visible",
}
SLANG = ["загоны / без загонов", "по красоте", "ору / это ор", "рофл", "треш", "пофиг / побоку / без разницы",
         "красава", "на вайбе", "запарка", "збс"]

# Train grid: RU 55 / EN 45, tiers weighted toward strong (where the fine-tune lost most).
TRAIN_TOTAL = {"ru": 545, "en": 495}
TIER_WEIGHT = {3: 0.45, 2: 0.30, 1: 0.25}
SLANG_PER_TERM = 6                       # 10 terms -> 60 RU slang rows
TEST_PER_TIER = {3: 22, 2: 15, 1: 13}    # per language -> 50 RU + 50 EN

SYSTEM = """You write seed sentences for a dataset that teaches a private phone keyboard to translate swearing
and slang between Russian and English faithfully. People write crude messages to friends and the translator
must keep them crude, so the swearing is wanted here. Write ONLY in {lang_name}.

Force: {tier}.
Function: {function}.

Write exactly {n} sentences, one per line, no numbering, no quotes, no commentary. Each is 4-18 words, a
natural chat or spoken message that a real person would type. Spell the swear words out in full, never with
asterisks. Vary the wording and the swear words across lines. No personal names, no real people, no lines
from films, songs or books, no slurs about ethnicity, nationality, religion, sexuality, gender or disability."""

SLANG_SYSTEM = """You write seed sentences for a dataset that teaches a phone keyboard current Russian chat
slang. Write ONLY in Russian, the way people in their twenties actually text in 2026.

Every sentence must use: {term}

Write exactly {n} sentences, one per line, no numbering, no quotes, no commentary. Each is 4-18 words, a
natural chat message. Vary the situation. No personal names, no real people, no lines from films or books."""

TEST_FUNCTIONS = "; ".join(f"{k}: {v.split(' (')[0]}" for k, v in FUNCTIONS.items())
SLUR = re.compile(r"(?<![а-яё])(?:пид[оа]р|педик|хач|чурк|жид|нигер|ниггер)|\b(?:fag|nigg|retard|tranny|kike|spic)\w*",
                  re.I)


# The model's own commentary on a line ("*   11: ... (12 words) - OK").
CHATTER = re.compile(r"^\s*[*#\-]|\(\d+ words?\)|\d+\s*слов|^\s*\d+[.:)]")


def norm(text: str) -> str:
    return re.sub(r"[^\wа-яё]+", " ", text.lower()).strip()


def grid(split: str) -> list[dict]:
    """Cells: {lang, tier, function, n, system}. tier 0 = slang (no profanity check)."""
    cells = []
    if split == "train":
        for lang, total in TRAIN_TOTAL.items():
            for t, w in TIER_WEIGHT.items():
                n = round(total * w / len(FUNCTIONS))
                for f, fdesc in FUNCTIONS.items():
                    cells.append({"lang": lang, "tier": t, "function": f, "n": n,
                                  "system": SYSTEM.format(lang_name=gs.LANG_NAMES[lang], tier=TIER_DESC[lang, t],
                                                          function=fdesc, n="{n}")})
        for term in SLANG:
            cells.append({"lang": "ru", "tier": 0, "function": "slang", "n": SLANG_PER_TERM,
                          "system": SLANG_SYSTEM.format(term=term, n="{n}")})
    else:
        for lang in ("ru", "en"):
            for t, n in TEST_PER_TIER.items():
                fdesc = "mixed: use a different one of these on each line -- " + TEST_FUNCTIONS
                cells.append({"lang": lang, "tier": t, "function": "mixed", "n": n,
                              "system": SYSTEM.format(lang_name=gs.LANG_NAMES[lang], tier=TIER_DESC[lang, t],
                                                     function=fdesc, n="{n}")})
    return cells


def accept(line: str, cell: dict) -> str | None:
    """Why `line` is rejected for `cell`, or None when it is kept."""
    if CHATTER.search(line):
        return "chatter"
    words = len(line.split())
    if not 4 <= words <= 18:
        return "length"
    if masked(line):
        return "masked"
    if SLUR.search(line):
        return "slur"
    if cell["lang"] == "ru" and not re.search(r"[а-яё]", line, re.I):
        return "language"
    if cell["lang"] == "en" and re.search(r"[а-яё]", line, re.I):
        return "language"
    if cell["tier"] and tier(line, cell["lang"]) != cell["tier"]:
        return f"tier {tier(line, cell['lang'])}"
    return None


def existing_texts(out_path: str) -> set[str]:
    """Normalised text of every seed file, the dataset, the eval probes and the hand-written rows pending
    native review (native_review_*.jsonl, train-only), so nothing repeats."""
    seen = gs.load_existing_seeds(gs.resolve_avoid_paths("seeds_*.txt,pilot_register1_*.txt", out_path))
    for p in glob.glob("dataset/*.jsonl") + glob.glob("../model/probes/*.jsonl") + glob.glob("native_review_*.jsonl"):
        with open(p, encoding="utf-8") as f:
            seen.update(json.loads(l)["source_text"] for l in f if l.strip())
    return {norm(s) for s in seen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="OpenRouter slug; default: config seed_generator_model")
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--max-calls-per-cell", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = grid(args.split)
    by = {}
    for c in cells:
        by[c["lang"], c["tier"]] = by.get((c["lang"], c["tier"]), 0) + c["n"]
    for (lang, t), n in sorted(by.items()):
        print(f"  {lang} {TIER_NAMES.get(t, 'slang'):6} {n}")
    print(f"{len(cells)} cells, {sum(c['n'] for c in cells)} seeds")
    if args.dry_run:
        print(cells[0]["system"].replace("{n}", str(cells[0]["n"])))
        return

    api_key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("Set OPENROUTER_API_KEY first.")
    config = json.load(open(args.config, encoding="utf-8"))
    cfg = config.get("seed_generator_model", {})
    if args.model:   # a roster model keeps its routing (provider pin, :floor tier)
        cfg = next((g for g in config["stage_a_generators"] if g["slug"] == args.model), {"slug": args.model})
    slug = cfg.get("slug", gs.DEFAULT_MODEL)
    provider, api_model = cfg.get("provider"), cfg.get("api_model")
    seen = existing_texts(args.out)
    print(f"model {slug}; avoiding {len(seen)} existing texts")

    kept, rejects, short = [], {}, []
    for c in cells:
        got = []
        for _ in range(args.max_calls_per_cell):
            if len(got) >= c["n"]:
                break
            want = min(args.chunk_size, c["n"] - len(got) + 3)
            try:
                lines = gs.call_with_retry(c["function"], c["lang"], want, slug, api_key, provider=provider,
                                           api_model=api_model, system=c["system"].replace("{n}", str(want)))
            except Exception as e:
                print(f"    {c['lang']} {c['tier']} {c['function']}: call failed ({e})")
                continue
            for line in lines:
                why = accept(line, c) or ("duplicate" if norm(line) in seen else None)
                if why:
                    rejects[why] = rejects.get(why, 0) + 1
                    continue
                seen.add(norm(line))
                got.append(line)
                if len(got) >= c["n"]:
                    break
            time.sleep(0.3)
        if len(got) < c["n"]:
            short.append((c["lang"], c["tier"], c["function"], len(got), c["n"]))
        kept += [{"seed": s, "lang": c["lang"], "tier": TIER_NAMES.get(c["tier"], "slang"),
                  "function": c["function"], "split": args.split} for s in got]
        print(f"  {c['lang']} {TIER_NAMES.get(c['tier'], 'slang'):6} {c['function']:18} {len(got)}/{c['n']}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(k["seed"] + "\n" for k in kept)
    with open(args.out.removesuffix(".txt") + ".meta.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(k, ensure_ascii=False) + "\n" for k in kept)
    print(f"\nWrote {len(kept)} seeds to {args.out}. Rejected lines: {rejects}")
    for s in short:
        print(f"  short: {s}")


if __name__ == "__main__":
    main()
