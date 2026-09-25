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

# Balance guard: before every chunk of GUARD_CHUNK seeds, ask OpenRouter what's left and stop
# cleanly (unprocessed pairs -> --failures, exit 0) below MIN_BALANCE_USD, so a run never hits a
# 402 halfway through. Keep MIN_BALANCE_USD above the cost of one chunk.
GUARD_CHUNK = 25
MIN_BALANCE_USD = 0.75
CREDITS_URL = "https://openrouter.ai/api/v1/credits"

# 429s mean "slow down", not "this model is broken": back off longer, and never count them
# toward the circuit breaker or model_health.json (parallel batches share one rate limit).
RATE_LIMIT_WAITS = (5, 10, 20, 40, 60, 60)


class RateLimited(RuntimeError):
    """Still HTTP 429 after every backoff; the pair is worth retrying later."""


class OutOfCredits(RuntimeError):
    """HTTP 402: the balance ran out despite the guard."""


def remaining_balance(api_key: str) -> float | None:
    """total_credits - total_usage from OpenRouter, or None if the endpoint can't be read."""
    try:
        r = requests.get(CREDITS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        r.raise_for_status()
        d = r.json()["data"]
        return float(d["total_credits"]) - float(d["total_usage"])
    except (requests.RequestException, KeyError, TypeError, ValueError) as e:
        print(f"    (balance check failed: {e})")
        return None


def balance_ok(api_key: str, min_balance: float, label: str) -> bool:
    bal = remaining_balance(api_key)
    if bal is None:
        return True          # can't tell -- carry on; a real 402 still stops the run (OutOfCredits)
    print(f"  [balance ${bal:.2f} before {label}]")
    if bal < min_balance:
        print(f"  !! balance ${bal:.2f} is under the ${min_balance:.2f} guard -- stopping cleanly.")
        return False
    return True

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


def call_model(sentence: str, model_slug: str, api_key: str, max_tokens: int = 1200,
               provider: dict | None = None, api_model: str | None = None) -> dict:
    """Calls one OpenRouter model to annotate one sentence."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": api_model or model_slug,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            "max_tokens": max_tokens,
            "reasoning": {"effort": "low"},
            **({"provider": provider} if provider else {}),
        },
        timeout=30,
    )
    if response.status_code == 402:
        raise OutOfCredits(f"HTTP 402 from OpenRouter for model={model_slug}: {response.text[:300]}")
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


def call_model_with_retry(sentence: str, model_slug: str, api_key: str,
                           provider: dict | None = None, api_model: str | None = None) -> dict:
    """Wraps call_model with two targeted retry strategies:
    - HTTP 429 (rate limit): back off (RATE_LIMIT_WAITS) and retry; if it's still 429 after
      every wait, raise RateLimited so the caller can treat it as "retry later", not a failure.
    - Empty content (reasoning ate the whole token budget): retry once with
      a much larger budget rather than raising the baseline for every call.
    """
    for attempt, wait in enumerate(RATE_LIMIT_WAITS + (None,)):
        try:
            return call_model(sentence, model_slug, api_key, provider=provider, api_model=api_model)
        except OutOfCredits:
            raise
        except RuntimeError as e:
            msg = str(e)
            if "HTTP 429" in msg:
                if wait is None:
                    raise RateLimited(f"still rate limited after {attempt} backoffs: {msg[:200]}") from e
                print(f"    -> rate limited, retrying in {wait}s (attempt {attempt + 1}/{len(RATE_LIMIT_WAITS)})")
                time.sleep(wait)
                continue
            if "EMPTY_CONTENT" in msg:
                print(f"    -> reasoning ate the budget, retrying once with more room")
                return call_model(sentence, model_slug, api_key, max_tokens=2500,
                                  provider=provider, api_model=api_model)
            raise


# Failure rows whose error starts with one of these are "not attempted / retry later", not model
# failures: they never count toward the circuit breaker or model_health.json.
NOT_A_MODEL_FAILURE = ("skipped --", "rate limited --", "out of credits --")


def is_model_failure(row: dict) -> bool:
    return not row["error"].startswith(NOT_A_MODEL_FAILURE)


def generate(seeds, generators, api_key, breaker_threshold=CIRCUIT_BREAKER_CONSECUTIVE, sleep=0.5,
             min_balance=MIN_BALANCE_USD, chunk=GUARD_CHUNK, call=None, check_balance=None,
             done=frozenset(), on_result=None, parallel=True):
    """Runs every (seed, generator) pair; per seed, the generators are called concurrently (one
    thread each -- they're separate models/providers, and 429s back off on their own). Pairs in
    `done` ({(source_text, slug)}, from --resume) are skipped. on_result(row) is called for every
    success as it arrives, so the output file grows incrementally. Returns (results, failures,
    call_count, consecutive_fail, breaker_tripped). call / check_balance are injectable for tests."""
    from concurrent.futures import ThreadPoolExecutor
    call = call or call_model_with_retry
    check_balance = check_balance or (lambda label: balance_ok(api_key, min_balance, label))
    results, failures = [], []
    call_count, total_calls = 0, len(seeds) * len(generators)
    consecutive_fail = {g["slug"]: 0 for g in generators}
    breaker_tripped = set()
    pool = ThreadPoolExecutor(max_workers=max(1, len(generators))) if parallel else None

    def skip_rest(from_seed: int, reason: str):
        for s in seeds[from_seed:]:
            for g in generators:
                if (s, g["slug"]) not in done:
                    failures.append({"source_text": s, "model": g["slug"], "error": reason})

    def attempt(sentence, generator):
        try:
            return call(sentence, generator["slug"], api_key,
                        provider=generator.get("provider"), api_model=generator.get("api_model")), None
        except Exception as e:                    # returned, not raised, so every thread reports back
            return None, e

    try:
        for i, sentence in enumerate(seeds):
            if i % chunk == 0 and not check_balance(f"seeds {i + 1}-{min(i + chunk, len(seeds))}"):
                skip_rest(i, f"skipped -- balance guard stopped the run before seed {i + 1}")
                break
            todo = []
            for generator in generators:
                slug = generator["slug"]
                call_count += 1
                if (sentence, slug) in done:
                    continue
                if slug in breaker_tripped:
                    failures.append({
                        "source_text": sentence, "model": slug,
                        "error": f"skipped -- circuit breaker tripped after "
                                 f"{breaker_threshold} consecutive failures this run",
                    })
                    print(f"[{call_count}/{total_calls}] SKIP model={slug}  (circuit breaker tripped)")
                    continue
                todo.append((call_count, generator))
            outcomes = (list(pool.map(lambda t: attempt(sentence, t[1]), todo)) if pool
                        else [attempt(sentence, g) for _, g in todo])
            # Bookkeeping in roster order, after all of this seed's calls are back.
            out_of_credits = None
            for (n, generator), (annotated, err) in zip(todo, outcomes):
                slug = generator["slug"]
                if err is None:
                    results.append(annotated)
                    if on_result:
                        on_result(annotated)
                    consecutive_fail[slug] = 0
                    print(f"[{n}/{total_calls}] ok   model={slug}  has_subtext={annotated.get('has_subtext')}")
                elif isinstance(err, RateLimited):
                    failures.append({"source_text": sentence, "model": slug, "error": f"rate limited -- {err}"})
                    print(f"[{n}/{total_calls}] RATE-LIMITED model={slug} (retry later, not a model failure)")
                elif isinstance(err, OutOfCredits):
                    failures.append({"source_text": sentence, "model": slug, "error": f"out of credits -- {err}"})
                    out_of_credits = err
                else:
                    failures.append({"source_text": sentence, "model": slug, "error": str(err)})
                    consecutive_fail[slug] += 1
                    print(f"[{n}/{total_calls}] FAILED model={slug}: {err}")
                    if consecutive_fail[slug] >= breaker_threshold:
                        breaker_tripped.add(slug)
                        print(f"    !! {slug} hit {breaker_threshold} consecutive failures -- "
                              f"excluding it from the rest of this run (remaining seeds recorded "
                              f"as failures for retry, not attempted).")
            if out_of_credits:
                print(f"OUT OF CREDITS -- stopping cleanly: {out_of_credits}")
                skip_rest(i + 1, "out of credits -- run stopped on HTTP 402")
                break
            time.sleep(sleep)
    finally:
        if pool:
            pool.shutdown()
    return results, failures, call_count, consecutive_fail, breaker_tripped


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
    parser.add_argument("--min-balance", type=float, default=MIN_BALANCE_USD,
                         help="Stop cleanly (unprocessed pairs -> --failures, exit 0) when the "
                              f"OpenRouter balance drops under this many USD (default {MIN_BALANCE_USD}); "
                              f"checked before every {GUARD_CHUNK} seeds.")
    parser.add_argument("--models", default=None,
                         help="Comma-separated slugs: only run these roster models (e.g. to backfill "
                              "models added after a batch was generated)")
    parser.add_argument("--resume", action="store_true",
                         help="Keep the existing --out file and skip (seed, model) pairs already in it")
    parser.add_argument("--sequential", action="store_true",
                         help="Call the generators one at a time instead of concurrently per seed")
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
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        unknown = wanted - {g["slug"] for g in generators}
        if unknown:
            raise SystemExit(f"--models not in the active roster: {sorted(unknown)}")
        generators = [g for g in generators if g["slug"] in wanted]
    seeds = load_seeds(args.seeds)
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done.add((row.get("source_text"), row.get("_generator_model")))
        print(f"Resuming: {len(done)} (seed, model) pairs already in {args.out} will be skipped.")
    print(f"Loaded {len(seeds)} seeds, {len(generators)} active generator models.")
    print(f"Total calls to make: {len(seeds) * len(generators) - len(done)}")

    # Rows are appended as they arrive, so an interrupted run keeps what it paid for (--resume).
    with open(args.out, "a" if args.resume else "w", encoding="utf-8") as out:
        def write_row(row):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
        results, failures, call_count, consecutive_fail, breaker_tripped = generate(
            seeds, generators, api_key, breaker_threshold=args.breaker_threshold, sleep=args.sleep,
            min_balance=args.min_balance, done=done, on_result=write_row, parallel=not args.sequential)

    print(f"\nDone. Wrote {len(results)} new candidate rows to {args.out}. {len(failures)} failures.")

    newly_disabled = []
    for generator in generators:
        slug = generator["slug"]
        attempted = call_count and slug in consecutive_fail
        if not attempted:
            continue
        tripped = slug in breaker_tripped
        model_fails = sum(1 for f in failures if f["model"] == slug and is_model_failure(f))
        attempts_for_slug = sum(1 for r in results if r.get("_generator_model") == slug) + model_fails
        fail_rate = model_fails / attempts_for_slug if attempts_for_slug else None
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
