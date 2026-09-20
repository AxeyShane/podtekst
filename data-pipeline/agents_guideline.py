"""
Guideline Agent CLI -- reads an agents_annotate.py --out-adjudicated file,
clusters the escalated_needs_human / dissented rows by category, and asks the
Guideline Agent to propose a calibration rule for whichever cluster looks
like a recurring pattern rather than a one-off (see AGENTS.md,
config/calibration_guidelines.md's "Pending review" section).

Can also run against Cowork verification's own disagreement record if one is
kept in that format -- anywhere adjudication produced dissenting_agents /
escalated rows works as input.

EXPERIMENTAL, UNTESTED LIVE (OpenRouter unreachable from device_bash, see
agents_annotate.py's header for the full explanation).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python agents_guideline.py --in stage_agents_batch6.jsonl \
        --min-cluster-size 3

Proposals are appended to config/calibration_guidelines.md's "Pending
review" section (via propose_guideline_update) -- nothing is auto-applied
to the numbered rules. Review and move approved ones up by hand.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.agent_defs import build_guideline_agent  # noqa: E402
from agents.schemas import GuidelineProposal  # noqa: E402


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cluster_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    """Only rows with real disagreement signal are worth the Guideline Agent's
    attention -- unanimous rows tell it nothing new."""
    clusters = defaultdict(list)
    for row in rows:
        if row.get("resolution") in ("majority_vote", "escalated_needs_human") or row.get("dissenting_agents"):
            clusters[row.get("category", "unknown")].append(row)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--min-cluster-size", type=int, default=3,
                     help="Don't bother the Guideline Agent with a pattern seen fewer "
                          "than this many times -- avoids proposing a rule from one "
                          "genuinely ambiguous sentence.")
    args = ap.parse_args()

    rows = load_rows(args.infile)
    clusters = cluster_by_category(rows)
    print(f"Loaded {len(rows)} row(s), {sum(len(v) for v in clusters.values())} with "
          f"disagreement signal, across {len(clusters)} categories.")

    guideline_agent = build_guideline_agent()
    proposals = []

    for category, cluster_rows in clusters.items():
        if len(cluster_rows) < args.min_cluster_size:
            print(f"  {category}: {len(cluster_rows)} row(s) -- below min-cluster-size, skipping")
            continue
        print(f"  {category}: {len(cluster_rows)} row(s) with disagreement -- analyzing")
        examples = json.dumps(
            [{"source_text": r["source_text"], "translation": r["translation"],
              "has_subtext": r["has_subtext"], "dissenting_agents": r.get("dissenting_agents", [])}
             for r in cluster_rows],
            ensure_ascii=False, indent=2,
        )
        try:
            output = guideline_agent.kickoff(
                messages=(
                    f"Category: {category}\n\n{len(cluster_rows)} sentences where the "
                    f"Annotator Agents disagreed or the Adjudicator escalated:\n{examples}\n\n"
                    "Call read_calibration_guidelines first -- don't propose a rule that "
                    "duplicates an existing one. If these examples share a real pattern "
                    "(not just 'these are hard'), propose a specific, generalizable rule in "
                    "the same style as the existing numbered rules. If they don't share a "
                    "clear pattern, say so honestly instead of forcing a rule -- return "
                    "direction='clarify_only' with proposed_rule explaining why no rule "
                    "fits yet, rather than inventing one."
                ),
                response_format=GuidelineProposal,
            )
            proposal = output.pydantic
            if proposal.direction != "clarify_only":
                from agents.tools import propose_guideline_update
                propose_guideline_update.run(
                    pattern_description=proposal.pattern_description,
                    proposed_rule=proposal.proposed_rule,
                    supporting_examples_json=json.dumps(proposal.supporting_examples, ensure_ascii=False),
                )
                print(f"    -> proposed: {proposal.pattern_description}")
            else:
                print(f"    -> no clear pattern: {proposal.proposed_rule}")
            proposals.append(proposal.model_dump())
        except Exception as e:
            print(f"    ! failed: {e}", file=sys.stderr)

    print(f"\n{len(proposals)} proposal(s) processed. Review "
          f"config/calibration_guidelines.md's 'Pending review' section.")


if __name__ == "__main__":
    main()
