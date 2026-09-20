"""
Auditor Agent CLI -- scans the accumulated stage_b_batchN.jsonl files for
category imbalance and near-duplicate redundancy, and recommends what the
next batch's seeds should target (see AGENTS.md; this is the role
config/models.json's stage_c_coverage_model was reserved for).

EXPERIMENTAL, UNTESTED LIVE (OpenRouter unreachable from device_bash, see
agents_annotate.py's header). The underlying dataset_category_stats tool
itself is local-only and IS verified -- tested against the real batch 1-3
data during this build (487 rows, category counts matched the known totals
exactly). Only the LLM recommendation step needs a live OpenRouter call.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python agents_audit.py --in stage_b_batch1.jsonl stage_b_batch2.jsonl \
        stage_b_batch3.jsonl --out coverage_report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.agent_defs import build_auditor_agent  # noqa: E402
from agents.schemas import CoverageReport  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infiles", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    auditor_agent = build_auditor_agent()
    paths_json = json.dumps(args.infiles)

    print(f"Auditing {len(args.infiles)} file(s): {', '.join(args.infiles)}")
    output = auditor_agent.kickoff(
        messages=(
            f"Call dataset_category_stats with stage_b_jsonl_paths_json={paths_json!r}. "
            "Using the result, fill in total_rows, category_counts, and "
            "has_subtext_true_ratio directly from the tool's numbers (don't recompute or "
            "estimate). Pass near_duplicate_clusters through as-is. Then use your own "
            "judgment to name underrepresented_patterns (categories or sub-patterns that "
            "look thin relative to the others) and write a specific recommendation for what "
            "the next batch's seed generation should target."
        ),
        response_format=CoverageReport,
    )
    report = output.pydantic

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"\nTotal rows: {report.total_rows}")
    print(f"Category counts: {report.category_counts}")
    print(f"has_subtext=true ratio: {report.has_subtext_true_ratio:.3f}")
    print(f"Near-duplicate clusters: {len(report.near_duplicate_clusters)}")
    print(f"Underrepresented: {report.underrepresented_patterns}")
    print(f"\nRecommendation: {report.recommendation}")
    print(f"\nWrote full report -> {args.out}")


if __name__ == "__main__":
    main()
