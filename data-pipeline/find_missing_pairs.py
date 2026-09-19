"""
Finds (seed, model) pairs missing from a Stage A output file, by diffing the
full seeds x active-roster matrix against what's actually present.

This is the same reconciliation originally done by hand for batch 1's 19
failures, formalized as a reusable script -- needed for any Stage A run that
started before stage_a_generate.py's --failures flag existed (batch 2's run
is the first case: it was already in progress when that patch landed, so it
won't write its own failures file), or any run where you suspect failures
went unreported for another reason (a hard crash instead of a caught
exception, for example).

Apostrophe variants (' vs the curly ') are normalized before comparing --
model output has been observed using a different apostrophe than the seed
file without that being a real generation failure (this bit us once
computing batch 1's failure list).

Only checks against the models currently listed in config/models.json's
stage_a_generators -- if the roster changed between when a run started and
now, pairs for a since-dropped model won't show up as "missing" here (there
was never going to be a retry for those anyway; see retry_failures.py's
same roster-aware skip logic).

Usage:
    python find_missing_pairs.py --seeds seeds_batch2.txt --stage-a stage_a_batch2.jsonl \
        --out stage_a_batch2.failures.jsonl

Only run this AFTER the Stage A run has fully finished -- against a
still-running/partial output file, every pair not generated yet will show up
as "missing" even though it's simply not done, not actually failed.

Output matches stage_a_generate.py's --failures schema, so it feeds straight
into retry_failures.py.
"""

import argparse
import json


def normalize(s):
    return (s or "").strip().replace("’", "'").replace("‘", "'")


def load_seeds(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_present_pairs(path):
    present = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            present.add((normalize(row.get("source_text")), row.get("_generator_model")))
    return present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="The seed file this Stage A run was given")
    ap.add_argument("--stage-a", required=True, dest="stage_a", help="The (finished) Stage A output JSONL")
    ap.add_argument("--out", required=True, help="Where to write the reconstructed failures JSONL")
    ap.add_argument("--config", default="config/models.json")
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    config = load_config(args.config)
    models = [g["slug"] for g in config["stage_a_generators"]]
    present = load_present_pairs(args.stage_a)

    missing = []
    for seed in seeds:
        for model in models:
            if (normalize(seed), model) not in present:
                missing.append({
                    "source_text": seed,
                    "model": model,
                    "error": "missing from stage_a output (run predates --failures tracking)",
                })

    with open(args.out, "w", encoding="utf-8") as f:
        for row in missing:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    expected = len(seeds) * len(models)
    print(f"Checked against active roster: {models}")
    print(f"Expected {expected} pairs ({len(seeds)} seeds x {len(models)} models), "
          f"found {expected - len(missing)} present, {len(missing)} missing.")
    print(f"Wrote missing pairs to {args.out} -- retry with:")
    print(f"  python retry_failures.py --failures {args.out} --out <retries_output>.jsonl")


if __name__ == "__main__":
    main()
