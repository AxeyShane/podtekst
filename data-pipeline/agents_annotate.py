"""
CLI entrypoint chaining Detection -> Annotator Agents -> Adjudicator Agent for
a seed file -- the agent-based counterpart to stage_a_generate.py +
stage_b_prefilter.py, built on CrewAI (see agents/ package, AGENTS.md).

EXPERIMENTAL PARALLEL PATH: this does not replace run_batch.py. It has not
been live-tested (OpenRouter is unreachable from the sandboxed device_bash
environment this was built in -- same limitation documented throughout this
pipeline for every OpenRouter-calling script). Run it on a small --limit
first, from a real terminal with OPENROUTER_API_KEY set, and compare its
output against a run_batch.py run on the same seeds before trusting it at
volume.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python agents_annotate.py --seeds seeds_batch6.txt --limit 5 \
        --out-adjudicated stage_agents_batch6.jsonl \
        --out-candidates stage_agents_batch6_candidates.jsonl \
        --out-skipped stage_agents_batch6_skipped.jsonl

Per sentence:
  1. Detection Agent decides worth_annotating (rule-grounded, cheap model).
     False -> logged to --out-skipped, nothing else runs for this sentence.
  2. All active Annotator Agents judge it independently and concurrently
     (asyncio, real parallelism -- not sequential calls).
  3. Adjudicator Agent reads all Annotator outputs + calibration_guidelines.md
     and produces one verdict: unanimous / majority_vote / escalated_needs_human.

Output schemas: agents/schemas.py's AnnotationResult (candidates) and
AdjudicationResult (adjudicated). escalated_needs_human rows still get
written to --out-adjudicated with that resolution value -- filter on it
before treating a batch as Cowork-free.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.agent_defs import (  # noqa: E402
    build_adjudicator_agent,
    build_annotator_agents,
    build_detection_agent,
)
from agents.schemas import AdjudicationResult, AnnotationResult, DetectionVerdict  # noqa: E402


def load_seeds(path: str, limit: int | None) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]
    return seeds[:limit] if limit else seeds


def detect(detection_agent, sentence: str) -> DetectionVerdict:
    output = detection_agent.kickoff(
        messages=(
            f"Candidate source sentence: {sentence!r}\n\n"
            "Call flag_ru_address_pronoun and lookup_idiom on this sentence (or on the "
            "phrase, for lookup_idiom) before deciding. Then decide worth_annotating: "
            "false only if neither tool found anything AND you see no plausible sarcasm/"
            "emotional-subtext cue either -- when genuinely unsure, prefer true, since a "
            "false negative here silently drops a sentence from the dataset while a false "
            "positive just costs a few cheap Annotator calls."
        ),
        response_format=DetectionVerdict,
    )
    return output.pydantic


async def annotate_all(annotator_agents, sentence: str) -> list[AnnotationResult]:
    prompt = (
        f"Sentence: {sentence!r}\n\n"
        "Call read_calibration_guidelines first. If this looks like it might contain an "
        "idiom, call lookup_idiom to check whether it's a known entry (and whether its "
        "'literal_survives' flag means has_subtext should actually be false). Then decide: "
        "does this carry has_subtext, which category, what's the best translation, and a "
        "nuance_note under ~20 words (empty string if has_subtext is false)."
    )
    tasks = [
        agent.kickoff_async(messages=prompt, response_format=AnnotationResult)
        for agent in annotator_agents
    ]
    outputs = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for agent, output in zip(annotator_agents, outputs):
        if isinstance(output, Exception):
            print(f"    ! Annotator {agent.role} failed: {output}", file=sys.stderr)
            continue
        results.append(output.pydantic)
    return results


def adjudicate(adjudicator_agent, sentence: str,
                candidates: list[AnnotationResult]) -> AdjudicationResult:
    candidates_json = json.dumps([c.model_dump() for c in candidates], ensure_ascii=False, indent=2)
    output = adjudicator_agent.kickoff(
        messages=(
            f"Sentence: {sentence!r}\n\n"
            f"Independent Annotator Agent outputs for this sentence:\n{candidates_json}\n\n"
            "Call read_calibration_guidelines first. If every candidate agrees on both "
            "has_subtext and category, resolution='unanimous' -- pick/polish the best "
            "translation among them. If they disagree, apply the calibration guidelines "
            "to decide (a guideline-backed minority can outweigh an unexplained majority); "
            "set resolution='majority_vote' if you can confidently resolve it this way, or "
            "resolution='escalated_needs_human' if the guidelines don't clearly settle it "
            "-- an honest escalation beats a forced guess. Set agreement_ratio to the "
            "fraction of candidates that agreed with your final has_subtext+category. List "
            "dissenting_agents by role."
        ),
        response_format=AdjudicationResult,
    )
    return output.pydantic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="Process only the first N seeds -- use this for the first "
                          "smoke-test run, this has never been run at volume.")
    ap.add_argument("--out-adjudicated", required=True)
    ap.add_argument("--out-candidates", required=True)
    ap.add_argument("--out-skipped", required=True)
    args = ap.parse_args()

    seeds = load_seeds(args.seeds, args.limit)
    print(f"Loaded {len(seeds)} seed(s) from {args.seeds}")

    detection_agent = build_detection_agent()
    annotator_agents = build_annotator_agents()
    adjudicator_agent = build_adjudicator_agent()
    print(f"Built 1 Detection Agent, {len(annotator_agents)} Annotator Agents, "
          f"1 Adjudicator Agent.")

    adjudicated, candidates, skipped = [], [], []

    for i, sentence in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] {sentence[:70]}")
        try:
            verdict = detect(detection_agent, sentence)
        except Exception as e:
            print(f"  ! Detection Agent failed: {e} -- skipping this sentence", file=sys.stderr)
            skipped.append({"source_text": sentence, "reason": f"detection_agent_error: {e}"})
            continue

        if not verdict.worth_annotating:
            print(f"  Detection: not worth annotating ({verdict.tool_evidence})")
            skipped.append(verdict.model_dump())
            continue

        print(f"  Detection: worth annotating (candidates: {verdict.candidate_categories})")
        results = asyncio.run(annotate_all(annotator_agents, sentence))
        if not results:
            print("  ! All Annotator Agents failed for this sentence -- skipping adjudication",
                  file=sys.stderr)
            skipped.append({"source_text": sentence, "reason": "all_annotators_failed"})
            continue
        candidates.extend(r.model_dump() for r in results)
        print(f"  {len(results)} Annotator outputs collected")

        try:
            verdict_result = adjudicate(adjudicator_agent, sentence, results)
        except Exception as e:
            print(f"  ! Adjudicator Agent failed: {e}", file=sys.stderr)
            skipped.append({"source_text": sentence, "reason": f"adjudicator_error: {e}"})
            continue
        adjudicated.append(verdict_result.model_dump())
        print(f"  Adjudicated: has_subtext={verdict_result.has_subtext} "
              f"category={verdict_result.category} resolution={verdict_result.resolution}")

    with open(args.out_adjudicated, "w", encoding="utf-8") as f:
        for row in adjudicated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out_candidates, "w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out_skipped, "w", encoding="utf-8") as f:
        for row in skipped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    escalated = sum(1 for r in adjudicated if r["resolution"] == "escalated_needs_human")
    print(f"\n=== Done ===")
    print(f"Adjudicated: {len(adjudicated)} ({escalated} escalated_needs_human) -> {args.out_adjudicated}")
    print(f"Raw candidates: {len(candidates)} -> {args.out_candidates}")
    print(f"Skipped by Detection Agent or errors: {len(skipped)} -> {args.out_skipped}")


if __name__ == "__main__":
    main()
