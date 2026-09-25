"""
End-to-end batch runner -- chains the six manual commands (check_roster,
stage_a_generate, retry_failures, deterministic_checks, stage_b_prefilter,
and the merge step in between) into one, so a batch goes from a seed file
to a Cowork-ready needs_cowork file without babysitting each script by hand.

Normal path (a fresh batch, seeds already exist -- run generate_seeds.py
first if they don't):
    export OPENROUTER_API_KEY=sk-or-...
    python run_batch.py --batch 3 --seeds seeds_batch3.txt

Steps, in order:
  1. check_roster.py           -- free pre-flight (pass --smoke-test to spend
                                   a few cents confirming each model actually
                                   responds before the real run)
  2. stage_a_generate.py       -- the expensive step, real API cost across
                                   every active generator x every seed
  3. retry_failures.py         -- only if step 2 wrote a --failures file
  4. merge retried rows into the main output (plain file concat)
  5. deterministic_checks.py   -- free, no API calls
  6. stage_b_prefilter.py      -- cheap GLM-5.3 calls, auto-resolves the
                                   unanimous subset

Ends with <out>_needs_cowork.jsonl ready to hand to Claude Cowork for Stage
B, <out>_prefilter_resolved.jsonl from the auto-resolve pass, and a
carryover seed file for next batch's rejects.

Resuming an already-finished Stage A run (e.g. one that predates
--failures tracking, like batch 2's did) -- skips steps 1-2 entirely:
    python run_batch.py --batch 2 --seeds seeds_batch2.txt \
        --resume-from stage_a_batch2.jsonl
If <resume-from> has no matching .failures.jsonl, this reconstructs one with
find_missing_pairs.py first (needs --seeds for that diff), same as if the
run had just finished with --failures support.

Any step exiting non-zero stops the chain immediately with a clear message
-- this does not try to paper over a real failure by skipping ahead.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def run(cmd, label):
    print(f"\n=== {label} ===")
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"\n'{label}' failed (exit {result.returncode}) -- stopping here. "
                          f"Fix the issue above and re-run; nothing after this step ran.")


def merge_jsonl(paths, out_path):
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            if not Path(p).exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.strip():
                        out.write(line + "\n")


def count_lines(path):
    if not Path(path).exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=int, help="Batch number, e.g. 3")
    ap.add_argument("--seeds", required=True, help="Seed file for this batch")
    ap.add_argument("--resume-from", default=None,
                     help="Skip check_roster + stage_a_generate; start from this existing "
                          "Stage A output JSONL instead (e.g. for a run that already finished)")
    ap.add_argument("--smoke-test", action="store_true",
                     help="Pass --smoke-test to check_roster.py (a few cents, skipped by default)")
    ap.add_argument("--min-agreement", type=float, default=1.0,
                     help="Passed through to stage_b_prefilter.py")
    args = ap.parse_args()

    n = args.batch
    stage_a_out = f"stage_a_batch{n}.jsonl"
    failures_path = f"stage_a_batch{n}.failures.jsonl"
    retries_out = f"stage_a_batch{n}_retries.jsonl"
    full_path = f"stage_a_batch{n}_full.jsonl"
    clean_path = f"stage_a_batch{n}.clean.jsonl"
    rejects_path = f"stage_a_batch{n}.rejects.jsonl"
    carryover_path = f"seeds_batch{n + 1}_carryover.txt"
    resolved_path = f"stage_b_batch{n}_prefilter_resolved.jsonl"
    needs_cowork_path = f"stage_a_batch{n}_needs_cowork.jsonl"

    if args.resume_from:
        stage_a_out = args.resume_from
        if not Path(failures_path).exists():
            # try the resumed file's own natural failures-file name too
            guess = args.resume_from[:-len(".jsonl")] + ".failures.jsonl" if args.resume_from.endswith(".jsonl") else None
            if guess and Path(guess).exists():
                failures_path = guess
            else:
                run([PY, "find_missing_pairs.py", "--seeds", args.seeds, "--stage-a", stage_a_out,
                     "--out", failures_path], "find_missing_pairs.py (reconstructing failures for a pre-existing run)")
    else:
        roster_cmd = [PY, "check_roster.py"]
        if args.smoke_test:
            roster_cmd.append("--smoke-test")
        run(roster_cmd, "check_roster.py")

        run([PY, "stage_a_generate.py", "--seeds", args.seeds, "--out", stage_a_out,
             "--failures", failures_path], "stage_a_generate.py")

    if Path(failures_path).exists() and count_lines(failures_path) > 0:
        run([PY, "retry_failures.py", "--failures", failures_path, "--out", retries_out],
            "retry_failures.py")
        merge_jsonl([stage_a_out, retries_out], full_path)
        print(f"Merged {stage_a_out} + {retries_out} -> {full_path}")
    else:
        merge_jsonl([stage_a_out], full_path)
        print("No failures to retry -- proceeding directly with the original Stage A output.")

    # One sentence = one prefilter group: point rows whose model echoed the seed with different
    # punctuation (’ -> ') back at the seed they were generated for.
    from stage_a_generate import rekey_to_seeds
    with open(args.seeds, encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]
    with open(full_path, encoding="utf-8") as f:
        full_rows = [json.loads(line) for line in f if line.strip()]
    changed = rekey_to_seeds(full_rows, seeds)
    if changed:
        with open(full_path, "w", encoding="utf-8") as f:
            for r in full_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Re-keyed {changed} row(s) whose model echoed the seed with different punctuation.")

    run([PY, "deterministic_checks.py", "--in", full_path, "--clean", clean_path,
         "--rejects", rejects_path, "--next-seeds", carryover_path], "deterministic_checks.py")

    run([PY, "stage_b_prefilter.py", "--in", clean_path, "--resolved", resolved_path,
         "--needs-cowork", needs_cowork_path, "--min-agreement", str(args.min_agreement)],
        "stage_b_prefilter.py")

    print("\n=== Batch pipeline complete ===")
    print(f"Auto-resolved by prefilter : {resolved_path} ({count_lines(resolved_path)} rows)")
    print(f"Needs Cowork (hand this off): {needs_cowork_path} "
          f"({count_lines(needs_cowork_path)} candidate rows)")
    print(f"Deterministic rejects      : {rejects_path} ({count_lines(rejects_path)} rows)")
    print(f"Carryover seeds for batch {n + 1}: {carryover_path} ({count_lines(carryover_path)} seeds)")


if __name__ == "__main__":
    main()
