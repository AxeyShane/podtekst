"""
Stage A: multi-model candidate generation for the Podtekst dataset.

For each seed sentence, calls every model listed in config/models.json's
stage_a_generators via OpenRouter, and asks each to produce a translation +
nuance annotation independently. Disagreement between models is a useful
signal on its own — sentences where generators disagree are often the
genuinely ambiguous ones worth extra scrutiny in Stage B (Cowork
verification, see docs/DESIGN.md).

This script only handles Stage A. Stage B (verification) runs in Claude
Cowork as a separate, semi-manual step — see docs/DESIGN.md for that flow.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python stage_a_generate.py --seeds seeds_example.txt --out stage_a_candidates.jsonl

Output: one JSON object per line, per (seed, model) pair. Multiple lines will
share the same source_text — that's expected; Stage B consumes all
candidates for a given sentence together.
"""

import argparse
import json
import os
import time
import requests

SYSTEM_PROMPT = """You are building training data for a small on-device model that
translates between Russian and English AND flags unstated intent that a literal
translation would lose (tone, formality register, sarcasm, idiom, emotional subtext).

For the given input sentence, respond with ONLY a JSON object (no markdown fences,
no preamble) with this exact shape:

{
  "source_lang": "ru" or "en",
  "source_text": "...",
  "translation": "...",
  "has_subtext": true or false,
  "category": "formality_shift" | "sarcasm" | "idiom" | "emotional_subtext" | "none",
  "nuance_note": "one short sentence explaining what a literal translation misses,
                   or empty string if has_subtext is false"
}

Rules:
- Only set has_subtext=true if a literal translation would genuinely mislead or
  flatten something important. Don't force a note onto plain, literal sentences.
- Cover the full range: many examples should have has_subtext=false. A dataset
  that always finds subtext teaches the model to over-annotate.
- For formality_shift, focus on ty/vy (informal/formal "you") and how that would
  be lost or misrepresented in English, or how English lacks a marker Russian has.
- Keep nuance_note under 20 words.
"""


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_seeds(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def call_model(sentence: str, model_slug: str, api_key: str) -> dict:
    """Calls one OpenRouter model to annotate one sentence."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_slug,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            "max_tokens": 300,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"].strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(text)
    parsed["_generator_model"] = model_slug
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="Text file, one sentence per line")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--config", default="config/models.json", help="Model roster config")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between calls (rate limiting)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment first.")

    config = load_config(args.config)
    generators = config["stage_a_generators"]
    seeds = load_seeds(args.seeds)
    print(f"Loaded {len(seeds)} seeds, {len(generators)} generator models.")
    print(f"Total calls to make: {len(seeds) * len(generators)}")

    results = []
    errors = 0
    call_count = 0
    total_calls = len(seeds) * len(generators)

    for sentence in seeds:
        for generator in generators:
            call_count += 1
            slug = generator["slug"]
            try:
                annotated = call_model(sentence, slug, api_key)
                results.append(annotated)
                print(f"[{call_count}/{total_calls}] ok   model={slug}  "
                      f"has_subtext={annotated.get('has_subtext')}")
            except Exception as e:
                errors += 1
                print(f"[{call_count}/{total_calls}] FAILED model={slug}: {e}")
            time.sleep(args.sleep)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(results)} candidate rows to {args.out}. {errors} failures.")
    print("Next: batch these by source_text and hand them to Claude Cowork for "
          "Stage B verification (see docs/DESIGN.md).")


if __name__ == "__main__":
    main()
