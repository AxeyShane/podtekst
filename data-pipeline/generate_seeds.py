"""
Seed generation -- delegates writing seed sentences to an OpenRouter model
instead of hand-writing them (see docs/DESIGN.md). This is step 0, upstream
of stage_a_generate.py: it produces the seeds_batchN.txt file that Stage A
then translates/annotates.

Rationale: hand-writing seeds doesn't scale past a couple of batches, and
this doesn't need Cowork-level judgment -- it's generation, not verification,
so it belongs on the same "scriptable, can run unattended" side of the
pipeline as Stage A and Stage C, not the Cowork side.

Deliberately targets category counts per batch rather than pulling randomly
from a natural corpus (Tatoeba etc.), because natural corpora skew toward
plain sentences -- see docs/PRODUCT.md and the batch-1 lesson that idiom
seeds crowded out formality_shift/sarcasm coverage. Adjust --targets each
batch to correct whatever the accumulating dataset is short on (Stage C's
coverage pass should inform this).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python generate_seeds.py --out seeds_batch3.txt \
        --targets formality_shift:40,sarcasm:30,idiom:15,emotional_subtext:25,none:50

--avoid defaults to auto-discovering every seeds_batch*.txt already in the current directory,
so you don't need to list prior batches by hand each time -- pass --avoid explicitly only to
override that (a narrower file list, a different glob, or --avoid "" to disable it).

Each (category, language) pair is requested as its own call in chunks of
--chunk-size sentences, so output quality doesn't degrade the way one giant
"write me 160 sentences" request tends to (repetition, drift). Both ru and
en are requested per category (roughly evenly split by default; override
with --lang-split).
"""

import argparse
import glob
import json
import os
import random
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up OPENROUTER_API_KEY from a .env file in the cwd, if present
except ImportError:
    pass  # python-dotenv not installed -- fall back to a real exported env var

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"  # best-performing model in this project so far

CATEGORY_BRIEFS = {
    "formality_shift": (
        "Sentences where Russian's ты/вы (informal/formal 'you') distinction, or an "
        "informal/formal imperative verb form, carries real social weight -- addressing a "
        "stranger, boss, doctor, in-law, customer vs. a close friend, sibling, or partner. "
        "Vary the context: family, dating, work, customer service, strangers on the street. "
        "For English sentences, include some where 'you'/'your' gives NO cue at all about "
        "which Russian register fits (genuinely ambiguous), and some where the English phrasing "
        "already makes the register obvious (e.g. clearly formal or clearly slangy)."
    ),
    "sarcasm": (
        "Sentences that only make sense as sarcastic/ironic in a plausible everyday context -- "
        "the literal words say the opposite of what's meant. Vary the trigger (bad luck, "
        "unreliability, disappointment, mock praise) and vary intensity, not just 'oh great, X "
        "again' every time."
    ),
    "idiom": (
        "Sentences built around a genuine idiom whose literal word-for-word translation would "
        "be confusing or meaningless in the other language. Mix idioms about emotion, work, "
        "luck, and character -- avoid the most overused examples (raining cats and dogs, jack "
        "of all trades, on cloud nine, seventh heaven, break a leg, drop in the ocean/bucket) "
        "since those are already well covered in this dataset."
    ),
    "emotional_subtext": (
        "Sentences carrying real emotional subtext beyond their literal content that ISN'T "
        "sarcasm or a fixed idiom -- backhanded compliments, passive-aggressive politeness, "
        "understatement, resigned acceptance, forced cheerfulness masking annoyance."
    ),
    "none": (
        "Plain, literal sentences with NO hidden nuance at all -- straightforward statements "
        "of fact or simple requests. Vary the life domain: family logistics, errands, pets, "
        "small talk, scheduling -- not just weather and office memos."
    ),
}

SYSTEM_TEMPLATE = """You are writing seed sentences for a training dataset about RU<->EN
translation nuance (formality register, sarcasm, idiom, emotional subtext). Write ONLY in
{lang_name}. Category for this request: {category}

{brief}

Write exactly {n} sentences, one per line, no numbering, no quotation marks, no commentary --
just the sentences themselves. Each must be a complete, natural, standalone sentence a real
person might actually type. Do not repeat a concept across lines within this batch."""

LANG_NAMES = {"ru": "Russian", "en": "English"}


def parse_targets(spec):
    """'formality_shift:40,sarcasm:30' -> {'formality_shift': 40, 'sarcasm': 30}"""
    out = {}
    for part in spec.split(","):
        cat, n = part.split(":")
        out[cat.strip()] = int(n)
    return out


def resolve_avoid_paths(spec, out_path):
    """None -> auto-discover seeds_batch*.txt in the cwd (the common case: don't make the
    caller remember to list every prior batch by hand). A comma-separated spec can mix literal
    filenames and globs (e.g. "seeds_batch*.txt" or "seeds_batch1.txt,seeds_batch2.txt"). Either
    way, the file we're about to write to is excluded, in case --out matches the pattern (e.g.
    re-running a failed batch) -- avoiding a file's own old partial content while regenerating
    it would just be confusing."""
    if spec is None:
        paths = sorted(glob.glob("seeds_batch*.txt"))
    else:
        paths = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if any(ch in part for ch in "*?["):
                matched = sorted(glob.glob(part))
                if not matched:
                    print(f"  (warning: --avoid pattern matched no files: {part})")
                paths.extend(matched)
            else:
                paths.append(part)

    out_abs = os.path.abspath(out_path)
    return [p for p in paths if os.path.abspath(p) != out_abs]


def load_existing_seeds(paths):
    existing = set()
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.add(line)
        except FileNotFoundError:
            print(f"  (warning: --avoid file not found, skipping: {p})")
    return existing


def call_model(category, lang, n, model_slug, api_key, max_tokens=1500):
    brief = CATEGORY_BRIEFS[category]
    system = SYSTEM_TEMPLATE.format(
        lang_name=LANG_NAMES[lang], category=category, brief=brief, n=n
    )
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_slug,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Write {n} {category} sentences in {LANG_NAMES[lang]} now."},
            ],
            "max_tokens": max_tokens,
            "temperature": 1.0,
        },
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    text = response.json()["choices"][0]["message"].get("content", "").strip()
    lines = [l.strip().strip('"').lstrip("0123456789.- ") for l in text.splitlines()]
    return [l for l in lines if l]


def call_with_retry(category, lang, n, model_slug, api_key):
    last_error = None
    for attempt in range(3):
        try:
            return call_model(category, lang, n, model_slug, api_key)
        except Exception as e:
            last_error = e
            wait = 5 * (attempt + 1)
            print(f"    -> failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise last_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", required=True,
                     help="e.g. formality_shift:40,sarcasm:30,idiom:15,emotional_subtext:25,none:50")
    ap.add_argument("--lang-split", type=float, default=0.55,
                     help="fraction of each category's target that should be Russian source "
                          "(rest is English source). Default 0.55, matching batch 1/2's ratio.")
    ap.add_argument("--avoid", default=None,
                     help="comma-separated seed files/globs to avoid exact-duplicating, e.g. "
                          "seeds_batch1.txt,seeds_batch2.txt or 'seeds_batch*.txt'. Default: "
                          "auto-discover every seeds_batch*.txt in the current directory, so "
                          "you don't have to remember to list prior batches by hand. Pass an "
                          "empty string (--avoid \"\") to disable avoidance entirely.")
    ap.add_argument("--chunk-size", type=int, default=12,
                     help="sentences requested per API call, to keep output quality up")
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment first.")

    model_slug = args.model
    if not model_slug:
        try:
            with open(args.config, encoding="utf-8") as f:
                cfg = json.load(f)
            model_slug = cfg.get("seed_generator_model", {}).get("slug", DEFAULT_MODEL)
        except FileNotFoundError:
            model_slug = DEFAULT_MODEL

    targets = parse_targets(args.targets)
    unknown = set(targets) - set(CATEGORY_BRIEFS)
    if unknown:
        raise SystemExit(f"Unknown categories in --targets: {unknown}. "
                          f"Known: {sorted(CATEGORY_BRIEFS)}")

    avoid_paths = resolve_avoid_paths(args.avoid, args.out)
    existing = load_existing_seeds(avoid_paths)
    shown = ", ".join(avoid_paths) if avoid_paths else "none found"
    print(f"Loaded {len(existing)} existing seeds to avoid duplicating (from {len(avoid_paths)} files: {shown}).")
    print(f"Seed generator model: {model_slug}")

    all_new = []
    seen_this_run = set()

    for category, total_n in targets.items():
        ru_n = round(total_n * args.lang_split)
        en_n = total_n - ru_n
        for lang, n in [("ru", ru_n), ("en", en_n)]:
            if n <= 0:
                continue
            collected = []
            while len(collected) < n:
                want = min(args.chunk_size, n - len(collected) + 3)  # ask a few extra to survive dedup
                print(f"  requesting {want} {category}/{lang} sentences...")
                lines = call_with_retry(category, lang, want, model_slug, api_key)
                for line in lines:
                    if line in existing or line in seen_this_run:
                        continue
                    collected.append(line)
                    seen_this_run.add(line)
                    if len(collected) >= n:
                        break
                time.sleep(args.sleep)
            all_new.extend(collected[:n])
            print(f"  -> got {len(collected[:n])}/{n} {category}/{lang}")

    with open(args.out, "w", encoding="utf-8") as f:
        for s in all_new:
            f.write(s + "\n")

    print(f"\nDone. Wrote {len(all_new)} seeds to {args.out}.")
    print("Next: python stage_a_generate.py --seeds", args.out, "--out <stage_a_output>.jsonl")


if __name__ == "__main__":
    main()
