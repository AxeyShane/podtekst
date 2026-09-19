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

import roster_health as rh

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up OPENROUTER_API_KEY from a .env file in the cwd, if present
except ImportError:
    pass  # python-dotenv not installed -- fall back to a real exported env var

# In-run circuit breaker: after this many CONSECUTIVE failures from one model
# within a single run, stop calling it for the rest of this run's seeds
# instead of burning the remaining calls on a model that's clearly down.
# Skipped seeds are still recorded to --failures for later retry.
CIRCUIT_BREAKER_CONSECUTIVE = 4

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


def call_model(sentence: str, model_slug: str, api_key: str, max_tokens: int = 1200) -> dict:
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
            "max_tokens": max_tokens,
            "reasoning": {"effort": "low"},
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code} from OpenRouter for model={model_slug}: "
            f"{response.text[:500]}"
        )
    data = response.json()
    message = data["choices"][0]["message"]
    text = message.get("content")
    if not text:
        raise RuntimeError(
            f"EMPTY_CONTENT from {model_slug} (likely all budget spent on reasoning). "
            f"Full message: {json.dumps(message)[:500]}"
        )
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Some models prepend stray text before the JSON object -- extract the
    # outermost {...} block rather than assuming the response starts clean.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"No JSON object found in response from {model_slug}: {text[:300]}")
    text = text[start:end + 1]
    parsed = json.loads(text)
    parsed["_generator_model"] = model_slug
    if parsed.get("source_lang") == "en":
        src_norm = sentence.strip().lower()
        trans_norm = str(parsed.get("translation", "")).strip().lower()
        if src_norm == trans_norm:
            parsed["_translation_suspect"] = (
                "translation identical to source_text -- model likely failed "
                "to translate into Russian, just echoed the English back"
            )
    return parsed


def call_model_with_retry(sentence: str, model_slug: str, api_key: str) -> dict:
    """Wraps call_model with two targeted retry strategies:
    - HTTP 429 (upstream rate limit): backoff and retry, up to 3 attempts.
    - Empty content (reasoning ate the whole token budget): retry once with
      a much larger budget rather than raising the baseline for every call.
    """
    last_error = None
    for attempt in range(3):
        try:
            return call_model(sentence, model_slug, api_key)
        except RuntimeError as e:
            last_error = e
            msg = str(e)
            if "429" in msg:
                wait = 5 * (attempt + 1)
                print(f"    -> rate limited, retrying in {wait}s (attempt {attempt + 1}/3)")
                time.sleep(wait)
                continue
            if "EMPTY_CONTENT" in msg:
                print(f"    -> reasoning ate the budget, retrying once with more room")
                try:
                    return call_model(sentence, model_slug, api_key, max_tokens=2500)
                except RuntimeError as e2:
                    last_error = e2
                    break
            else:
                raise
    raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="Text file, one sentence per line")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--config", default="config/models.json", help="Model roster config")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between calls (rate limiting)")
    parser.add_argument("--failures", default=None,
                         help="Where to write failed (seed, model) pairs as JSONL, for "
                              "retry_failures.py. Defaults to <out> with .jsonl replaced by "
                              ".failures.jsonl. Only written if there's at least one failure.")
    parser.add_argument("--breaker-threshold", type=int, default=CIRCUIT_BREAKER_CONSECUTIVE,
                         help="Consecutive failures from one model within this run before it's "
                              "excluded from the rest of the run (default: "
                              f"{CIRCUIT_BREAKER_CONSECUTIVE}).")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment first.")

    config = load_config(args.config)
    health = rh.load_health()
    generators, skipped = rh.get_active_generators(config, health)
    if skipped:
        print("Auto-disabled generators, excluded from this run (see config/model_health.json):")
        for slug, reason in skipped:
            print(f"  - {slug}: {reason}")
    seeds = load_seeds(args.seeds)
    print(f"Loaded {len(seeds)} seeds, {len(generators)} active generator models.")
    print(f"Total calls to make: {len(seeds) * len(generators)}")

    results = []
    failures = []
    call_count = 0
    total_calls = len(seeds) * len(generators)
    consecutive_fail = {g["slug"]: 0 for g in generators}
    breaker_tripped = set()

    for sentence in seeds:
        for generator in generators:
            call_count += 1
            slug = generator["slug"]
            if slug in breaker_tripped:
                failures.append({
                    "source_text": sentence, "model": slug,
                    "error": f"skipped -- circuit breaker tripped after "
                             f"{args.breaker_threshold} consecutive failures this run",
                })
                print(f"[{call_count}/{total_calls}] SKIP model={slug}  (circuit breaker tripped)")
                continue
            try:
                annotated = call_model_with_retry(sentence, slug, api_key)
                results.append(annotated)
                consecutive_fail[slug] = 0
                print(f"[{call_count}/{total_calls}] ok   model={slug}  "
                      f"has_subtext={annotated.get('has_subtext')}")
            except Exception as e:
                failures.append({"source_text": sentence, "model": slug, "error": str(e)})
                consecutive_fail[slug] += 1
                print(f"[{call_count}/{total_calls}] FAILED model={slug}: {e}")
                if consecutive_fail[slug] >= args.breaker_threshold:
                    breaker_tripped.add(slug)
                    print(f"    !! {slug} hit {args.breaker_threshold} consecutive failures -- "
                          f"excluding it from the rest of this run (remaining seeds recorded "
                          f"as failures for retry, not attempted).")
            time.sleep(args.sleep)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(results)} candidate rows to {args.out}. {len(failures)} failures.")

    newly_disabled = []
    for generator in generators:
        slug = generator["slug"]
        attempted = call_count and slug in consecutive_fail
        if not attempted:
            continue
        tripped = slug in breaker_tripped
        attempts_for_slug = sum(1 for r in results if r.get("_generator_model") == slug) + \
            sum(1 for f in failures if f["model"] == slug and "circuit breaker" not in f["error"])
        fail_rate = (
            sum(1 for f in failures if f["model"] == slug and "circuit breaker" not in f["error"])
            / attempts_for_slug
        ) if attempts_for_slug else None
        if rh.record_run_result(health, slug, tripped_breaker=tripped,
                                 failure_rate=fail_rate, run_label=args.out):
            newly_disabled.append(slug)
    rh.save_health(health)
    if newly_disabled:
        print(f"\nAUTO-DISABLED (repeated failures across consecutive runs): {newly_disabled}")
        print("These will be skipped in every future run until manually reinstated:")
        print(f"  python check_roster.py --reinstate <slug>")

    if failures:
        failures_path = args.failures or (
            args.out[:-len(".jsonl")] + ".failures.jsonl" if args.out.endswith(".jsonl")
            else args.out + ".failures.jsonl"
        )
        with open(failures_path, "w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Failed (seed, model) pairs written to {failures_path} -- "
              f"retry with retry_failures.py before running deterministic_checks.py.")

    print("Next: batch these by source_text and hand them to Claude Cowork for "
          "Stage B verification (see docs/DESIGN.md).")


if __name__ == "__main__":
    main()
