"""
Retries failed (seed, model) pairs from a Stage A run, instead of manually
diffing seeds x models against the output after the fact (which is how
batch 1's 19 failures were originally reconstructed -- see
docs/stage-b-verification-log.md).

Closes a gap named in docs/DESIGN.md's feedback-loop step: failed
generation calls now get a real retry path, not just a one-off
hand-written script per batch.

Two input modes:
  --failures  a JSONL file written by stage_a_generate.py's --failures
              output (each line: {"source_text": ..., "model": ..., "error": ...}).
              This is the normal path for any batch run after this script existed.
  --seeds + --model   a plain seed list (one sentence per line) retried against
              one explicitly named model -- kept for the older files written
              before --failures existed, e.g.:
                retry_failures.py --seeds retry_seeds_deepseek.txt --model deepseek/deepseek-v4-flash

Either way, any pair whose model is no longer in config/models.json's
stage_a_generators is skipped rather than retried -- a roster change (a
model getting dropped) shouldn't cause pointless retries against a model
that's no longer part of the pipeline. This is also why retry_seeds_qwen.txt
doesn't need separate handling: pointing this script at it (with
--model qwen/qwen3.8-flash) will just report all 18 pairs skipped, since
qwen was dropped from the roster -- confirming there's nothing to do there
rather than requiring anyone to remember that by hand.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python retry_failures.py --failures stage_a_batch5.failures.jsonl \
        --out stage_a_batch5_retries.jsonl

Output is successfully-retried rows only, in the same schema stage_a_generate.py
produces -- concatenate it onto the batch's main stage_a_*.jsonl before running
deterministic_checks.py. Anything that fails again goes to
<out with .still_failing.jsonl instead of .jsonl> for another round.
"""

import argparse
import json
import os
import time

import stage_a_generate as sa


def load_failures_jsonl(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append((row["source_text"], row["model"]))
    return pairs


def load_seed_list_for_model(path, model_slug):
    with open(path, encoding="utf-8") as f:
        return [(line.strip(), model_slug) for line in f if line.strip()]


def retry_pairs(pairs, api_key, sleep=0.5, min_balance=sa.MIN_BALANCE_USD, chunk=sa.GUARD_CHUNK,
                call=None, check_balance=None, routing=None, on_result=None):
    """Returns (results, still_failing). Checks the balance before every `chunk` pairs and stops
    cleanly under min_balance; unprocessed, rate-limited and 402'd pairs land in still_failing
    for a later round. routing maps slug -> roster entry so retries use the same provider /
    api_model routing as Stage A. call / check_balance are injectable for tests."""
    routing = routing or {}
    call = call or sa.call_model_with_retry
    check_balance = check_balance or (lambda label: sa.balance_ok(api_key, min_balance, label))
    results, still_failing = [], []

    def rest(start, reason):
        still_failing.extend({"source_text": s, "model": m, "error": reason} for s, m in pairs[start:])

    for i, (sentence, model_slug) in enumerate(pairs):
        if i % chunk == 0 and not check_balance(f"pairs {i + 1}-{min(i + chunk, len(pairs))}"):
            rest(i, f"skipped -- balance guard stopped the run before pair {i + 1}")
            break
        try:
            g = routing.get(model_slug, {})
            row = call(sentence, model_slug, api_key, provider=g.get("provider"), api_model=g.get("api_model"))
            results.append(row)
            if on_result:
                on_result(row)
            print(f"[{i + 1}/{len(pairs)}] ok   model={model_slug}  {sentence[:50]}")
        except sa.OutOfCredits as e:
            print(f"[{i + 1}/{len(pairs)}] OUT OF CREDITS -- stopping cleanly: {e}")
            rest(i, f"out of credits -- {e}")
            break
        except sa.RateLimited as e:
            still_failing.append({"source_text": sentence, "model": model_slug, "error": f"rate limited -- {e}"})
            print(f"[{i + 1}/{len(pairs)}] RATE-LIMITED model={model_slug} (retry later)")
        except Exception as e:
            still_failing.append({"source_text": sentence, "model": model_slug, "error": str(e)})
            print(f"[{i + 1}/{len(pairs)}] FAILED model={model_slug}: {e}  {sentence[:50]}")
        time.sleep(sleep)
    return results, still_failing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", help="JSONL failures file from stage_a_generate.py's --failures flag")
    ap.add_argument("--seeds", help="Plain seed list, retried against one --model (older-file compat mode)")
    ap.add_argument("--model", help="Required with --seeds: the model slug to retry against")
    ap.add_argument("--out", required=True, help="Output JSONL of successfully retried rows")
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--min-balance", type=float, default=sa.MIN_BALANCE_USD,
                    help="Stop cleanly when the OpenRouter balance drops under this many USD "
                         f"(checked every {sa.GUARD_CHUNK} pairs); the rest go to .still_failing.jsonl")
    args = ap.parse_args()

    if not args.failures and not args.seeds:
        raise SystemExit("Pass either --failures, or --seeds together with --model.")
    if args.seeds and not args.model:
        raise SystemExit("--seeds requires --model.")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment first.")

    config = sa.load_config(args.config)
    active_slugs = {g["slug"] for g in config["stage_a_generators"]}

    pairs = (
        load_failures_jsonl(args.failures) if args.failures
        else load_seed_list_for_model(args.seeds, args.model)
    )

    skipped = [(s, m) for s, m in pairs if m not in active_slugs]
    pairs = [(s, m) for s, m in pairs if m in active_slugs]

    if skipped:
        dropped_models = sorted({m for _, m in skipped})
        print(f"Skipping {len(skipped)} pair(s) whose model is no longer in the active "
              f"roster ({dropped_models}) -- nothing to retry there.")

    if not pairs:
        print("Nothing left to retry.")
        return

    print(f"Retrying {len(pairs)} (seed, model) pair(s)...")
    routing = {g["slug"]: g for g in config["stage_a_generators"]}
    # Rows are written as they arrive, so an interrupted retry keeps what it paid for.
    with open(args.out, "w", encoding="utf-8") as f:
        def write_row(row):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        results, still_failing = retry_pairs(pairs, api_key, sleep=args.sleep, min_balance=args.min_balance,
                                             routing=routing, on_result=write_row)

    if still_failing:
        still_path = (
            args.out[:-len(".jsonl")] + ".still_failing.jsonl" if args.out.endswith(".jsonl")
            else args.out + ".still_failing.jsonl"
        )
        with open(still_path, "w", encoding="utf-8") as f:
            for r in still_failing:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(still_failing)} pair(s) failed again -- wrote to {still_path} for another round.")

    print(f"\nDone. {len(results)}/{len(pairs)} succeeded -> {args.out}.")
    print("Concatenate this onto the batch's main stage_a_*.jsonl before running deterministic_checks.py.")


if __name__ == "__main__":
    main()
