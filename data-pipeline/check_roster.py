"""
Pre-flight roster check -- run this before every batch's Stage A generation
(catches a roster problem before a whole batch is spent on it, not after --
that's how qwen's reliability problem went unnoticed through all of
batch 1).

Two things, in order:
1. Live availability/pricing check against OpenRouter's public models API
   (https://openrouter.ai/api/v1/models -- no API key needed for this GET).
   Flags any configured slug no longer found there (renamed/deprecated) and
   shows current pricing/context length for everything so drift is visible
   at a glance, before you find out the hard way mid-batch.
2. Optional smoke test (--smoke-test; costs a few cents, needs
   OPENROUTER_API_KEY): sends one real seed to each currently-active
   generator and feeds the result into the same health tracking
   stage_a_generate.py's circuit breaker uses (roster_health.py) -- a model
   that's already unwell gets caught and, if it's already on thin ice, can
   get auto-disabled here instead of partway through the real batch.

This is a cheap health/availability gate, not a quality re-benchmark --
re-litigating which model translates best doesn't need to happen before
every batch (see the roster-comparison notes in the project log for that
kind of periodic, deliberate review). This script is about catching "did
something break" fast and for free (or for a few cents with --smoke-test).

Usage:
    python check_roster.py                       # pricing/availability check only, free
    python check_roster.py --smoke-test           # + one real call per active model
    python check_roster.py --reinstate <slug>     # manually un-disable a model
"""

import argparse
import os

import requests

import roster_health as rh
import stage_a_generate as sa

MODELS_API = "https://openrouter.ai/api/v1/models"


def fetch_live_models():
    resp = requests.get(MODELS_API, timeout=20)
    resp.raise_for_status()
    return {m["id"]: m for m in resp.json().get("data", [])}


def fmt_price_per_million(per_token_str):
    try:
        return f"${float(per_token_str) * 1_000_000:.3f}/M"
    except (TypeError, ValueError):
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--smoke-test", action="store_true",
                     help="Send one real seed to each active generator (costs a few cents, "
                          "needs OPENROUTER_API_KEY)")
    ap.add_argument("--test-sentence", default="Не могли бы вы уделить мне пару минут?")
    ap.add_argument("--reinstate", default=None,
                     help="Manually clear an auto-disabled model's health state and exit "
                          "(use after confirming a flagged failure was transient, e.g. an "
                          "OpenRouter outage rather than a real reliability problem)")
    args = ap.parse_args()

    health = rh.load_health()

    if args.reinstate:
        if rh.reinstate(health, args.reinstate):
            rh.save_health(health)
            print(f"Reinstated {args.reinstate} -- it will be called again starting next run.")
        else:
            print(f"{args.reinstate} wasn't disabled (nothing to do).")
        return

    config = sa.load_config(args.config)
    active, skipped = rh.get_active_generators(config, health)

    print(f"Configured generators: {len(config['stage_a_generators'])}")
    if skipped:
        print("Auto-disabled (excluded from every run until reinstated):")
        for slug, reason in skipped:
            print(f"  - {slug}: {reason}")
    print(f"Active this batch: {[g['slug'] for g in active]}\n")

    print("Checking live pricing/availability against OpenRouter...")
    try:
        live = fetch_live_models()
    except Exception as e:
        live = None
        print(f"  (couldn't reach OpenRouter's models API: {e} -- skipping this check)")

    if live is not None:
        watched = [g["slug"] for g in config["stage_a_generators"]]
        for key in ("stage_b_prefilter_model", "seed_generator_model", "stage_c_coverage_model"):
            if key in config:
                watched.append(config[key]["slug"])
        for slug in watched:
            entry = live.get(slug)
            if entry is None:
                print(f"  ! {slug}: NOT FOUND in OpenRouter's current model list -- "
                      f"likely renamed or deprecated. Verify on openrouter.ai before this batch.")
                continue
            pricing = entry.get("pricing", {})
            print(f"  {slug}: prompt {fmt_price_per_million(pricing.get('prompt'))}, "
                  f"completion {fmt_price_per_million(pricing.get('completion'))}, "
                  f"context {entry.get('context_length')}")

    if args.smoke_test:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENROUTER_API_KEY in your environment first.")
        print(f"\nSmoke-testing {len(active)} active model(s) with one seed each...")
        newly_disabled = []
        for g in active:
            slug = g["slug"]
            try:
                sa.call_model(args.test_sentence, slug, api_key)
                print(f"  ok   {slug}")
                rh.record_run_result(health, slug, tripped_breaker=False)
            except Exception as e:
                print(f"  FAIL {slug}: {e}")
                if rh.record_run_result(health, slug, tripped_breaker=True,
                                         failure_rate=1.0, run_label="check_roster smoke-test"):
                    newly_disabled.append(slug)
        rh.save_health(health)
        if newly_disabled:
            print(f"\nAUTO-DISABLED from this smoke test: {newly_disabled}")
            print("These won't be called in the batch that follows -- "
                  "python check_roster.py --reinstate <slug> to override.")

    print("\nDone. Safe to start the batch with the 'Active this batch' list above.")


if __name__ == "__main__":
    main()
