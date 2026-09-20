"""
Stage B automated pre-filter (GLM-5.3), sitting between deterministic_checks.py
and the Cowork LLM judge.

Rationale (see docs/DESIGN.md): Cowork's usage is session-bounded and
semi-manual, so it's the bottleneck in the verification cycle. Most sentences
in a batch aren't actually contested -- every surviving generator already
agrees on has_subtext and category, and the only real work left is picking
the clearest phrasing among candidates that already agree. That's a cheap,
well-scoped task a fast model can do unattended, freeing Cowork sessions to
spend judgment only on the sentences generators actually disagree about.

This script does NOT re-decide has_subtext/category -- it only picks/polishes
a translation + nuance_note for groups where every surviving candidate
already agrees on both. Anything less than full agreement (the default; see
--min-agreement) is passed through untouched to the Cowork queue, unchanged
from today's process.

IMPORTANT: unanimous does not mean correct. All generators can share the
same blind spot. Auto-resolved rows still need to appear in the human
calibration sample (docs/DESIGN.md step 4) -- don't only calibration-check
Cowork's contested-row verdicts.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python stage_b_prefilter.py --in stage_a_batch2.clean.jsonl \
        --resolved stage_b_batch2_prefilter_resolved.jsonl \
        --needs-cowork stage_a_batch2_needs_cowork.jsonl

Input is expected to be the *clean* output of deterministic_checks.py (rows
that already passed schema/echo/dedup checks). Candidates flagged
_translation_suspect are still excluded from consideration here as a second
line of defense, in case an ungated file is passed in.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up OPENROUTER_API_KEY from a .env file in the cwd, if present
except ImportError:
    pass  # python-dotenv not installed -- fall back to a real exported env var

DEFAULT_MODEL = "z-ai/glm-5.3-flash"

PREFILTER_SYSTEM_PROMPT = """You are picking the best translation for training data. Several
independent translation models already agree on whether this sentence carries hidden nuance
(has_subtext) and what category it falls into -- that judgment is SETTLED, do not second-guess it.
Your only job: given the source sentence and a list of candidate translations + nuance notes that
all share the same has_subtext/category verdict, either pick the clearest one verbatim or lightly
merge them into one clean answer. Respond with ONLY a JSON object (no markdown fences, no preamble):

{
  "translation": "...",
  "nuance_note": "one short sentence, under 20 words, or empty string if has_subtext is false"
}
"""


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def group_by_source(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_text"]].append(r)
    return list(groups.items())


def eligible_for_autoresolve(valid_candidates, min_agreement):
    """Returns (eligible: bool, has_subtext, category, agreeing_candidates, fraction)."""
    if not valid_candidates:
        return False, None, None, [], 0.0
    subtext_counts = Counter(c.get("has_subtext") for c in valid_candidates)
    top_subtext, top_count = subtext_counts.most_common(1)[0]
    fraction = top_count / len(valid_candidates)
    if fraction < min_agreement:
        return False, None, None, [], fraction

    agreeing = [c for c in valid_candidates if c.get("has_subtext") == top_subtext]
    cats = set(c.get("category") for c in agreeing)
    if len(cats) != 1:
        return False, None, None, [], fraction

    return True, top_subtext, cats.pop(), agreeing, fraction


def call_glm(source_text, has_subtext, category, agreeing, model_slug, api_key):
    candidate_lines = "\n".join(
        f'- translation: "{c.get("translation")}" | nuance_note: "{c.get("nuance_note", "")}"'
        for c in agreeing
    )
    user_msg = (
        f"Source sentence: {source_text}\n"
        f"Settled verdict: has_subtext={has_subtext}, category={category}\n"
        f"Candidates (all agree on the verdict above):\n{candidate_lines}\n"
    )
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_slug,
            "messages": [
                {"role": "system", "content": PREFILTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 400,
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code} from OpenRouter: {response.text[:300]}")
    text = response.json()["choices"][0]["message"].get("content", "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"No JSON object in GLM response: {text[:300]}")
    return json.loads(text[start:end + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True,
                     help="deterministic_checks.py's clean output")
    ap.add_argument("--resolved", required=True, help="Auto-resolved verified rows, Stage B schema")
    ap.add_argument("--needs-cowork", required=True,
                     help="Contested groups' raw candidate rows, unchanged -- feed to Cowork as usual")
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--model", default=None,
                     help="Override the model slug (default: config's stage_b_prefilter_model, "
                          f"falling back to {DEFAULT_MODEL})")
    ap.add_argument("--min-agreement", type=float, default=1.0,
                     help="Fraction of candidates that must agree on has_subtext+category to "
                          "auto-resolve (default 1.0 = strict unanimity, no dissenters at all)")
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
            model_slug = cfg.get("stage_b_prefilter_model", {}).get("slug", DEFAULT_MODEL)
        except FileNotFoundError:
            model_slug = DEFAULT_MODEL

    rows = load_jsonl(args.infile)
    groups = group_by_source(rows)
    print(f"Loaded {len(rows)} candidate rows across {len(groups)} sentences.")
    print(f"Pre-filter model: {model_slug}  |  min agreement: {args.min_agreement}")

    resolved, needs_cowork = [], []
    auto_count, contested_count, error_count = 0, 0, 0

    for source_text, cands in groups:
        valid = [c for c in cands if "_translation_suspect" not in c]
        eligible, has_subtext, category, agreeing, fraction = eligible_for_autoresolve(
            valid, args.min_agreement
        )
        if not eligible:
            contested_count += 1
            needs_cowork.extend(cands)  # pass through unchanged, including suspect ones
            continue

        try:
            picked = call_glm(source_text, has_subtext, category, agreeing, model_slug, api_key)
            resolved.append({
                "source_lang": cands[0]["source_lang"],
                "source_text": source_text,
                "translation": picked["translation"],
                "has_subtext": has_subtext,
                "category": category,
                "nuance_note": picked.get("nuance_note", ""),
                "_stage_b_verification": "auto_resolved_prefilter",
                "_stage_b_prefilter_model": model_slug,
                "_stage_b_candidate_count": len(valid),
                "_stage_b_agreement_fraction": fraction,
            })
            auto_count += 1
            print(f"  auto-resolved ({fraction:.0%} agree): {source_text[:60]}")
        except Exception as e:
            # The verdict is already settled (every valid candidate agreed) -- only the
            # polish/pick call to the prefilter model failed (rate limit, bad JSON, etc).
            # Falling through to the Cowork queue here would waste a human verification
            # slot on something that was never actually contested, so instead fall back to
            # the first agreeing candidate's translation/nuance_note verbatim, unpolished.
            error_count += 1
            fallback = agreeing[0]
            resolved.append({
                "source_lang": cands[0]["source_lang"],
                "source_text": source_text,
                "translation": fallback["translation"],
                "has_subtext": has_subtext,
                "category": category,
                "nuance_note": fallback.get("nuance_note", ""),
                "_stage_b_verification": "auto_resolved_prefilter_fallback_unpolished",
                "_stage_b_prefilter_model": model_slug,
                "_stage_b_candidate_count": len(valid),
                "_stage_b_agreement_fraction": fraction,
                "_stage_b_prefilter_error": str(e)[:200],
            })
            auto_count += 1
            print(f"  auto-resolved via unpolished fallback (prefilter call failed: {e}): {source_text[:60]}")
        time.sleep(args.sleep)

    write_jsonl(args.resolved, resolved)
    write_jsonl(args.needs_cowork, needs_cowork)

    print(f"\nDone. {len(groups)} sentences: {auto_count} auto-resolved "
          f"({error_count} of those via unpolished fallback after a prefilter call error), "
          f"{contested_count} genuinely contested -> {args.needs_cowork}.")
    print(f"Cowork workload: {len({r['source_text'] for r in needs_cowork})} sentences "
          f"(down from {len(groups)}).")
    print("Reminder: auto-resolved rows still need to appear in the human calibration sample "
          "(docs/DESIGN.md step 4) -- unanimous doesn't mean correct.")


if __name__ == "__main__":
    main()
