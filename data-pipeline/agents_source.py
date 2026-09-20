"""
Sourcing Agent CLI -- mines the Russian Language StackExchange for real RU-EN
pairs with human-written nuance commentary, instead of generating sentences
from scratch (see AGENTS.md, agents/tools.py's search_ru_en_forums).

EXPERIMENTAL, UNTESTED LIVE: api.stackexchange.com is unreachable from the
sandboxed device_bash environment this was built in (confirmed 2026-09-20,
same org-egress-proxy block affecting openrouter.ai) -- run from a real
terminal with normal internet access.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python agents_source.py --queries "ты вы разница" "формальное обращение" \
        --out sourced_candidates.jsonl --max-per-query 5

Output: agents/schemas.py's SourcedCandidate, one per line. This is raw
material, not a verified dataset row -- run it through agents_annotate.py (or
hand-review) before treating source_text/human_commentary as ground truth.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.agent_defs import build_sourcing_agent  # noqa: E402
from agents.schemas import SourcedCandidate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", required=True,
                     help="Search queries, e.g. 'ты вы разница' 'формальное обращение'")
    ap.add_argument("--max-per-query", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sourcing_agent = build_sourcing_agent()
    candidates: list[dict] = []

    for query in args.queries:
        print(f"Searching: {query!r}")
        try:
            output = sourcing_agent.kickoff(
                messages=(
                    f"Search for: {query!r} (use search_ru_en_forums with "
                    f"max_results={args.max_per_query}). From the results, extract up to "
                    f"{args.max_per_query} genuine RU-EN sentence pairs where a bilingual "
                    "human explained a real nuance (register, idiom, tone) -- skip results "
                    "that are just definitions or unrelated. For each, give the Russian "
                    "source_text, the human's own explanation as human_commentary (don't "
                    "paraphrase it into your own words), the source_url, and source_kind="
                    "'forum_qa'. Return ONE of the best candidates from this query as your "
                    "answer -- call this again per query for more."
                ),
                response_format=SourcedCandidate,
            )
            result = output.pydantic
            print(f"  -> {result.source_text[:60]!r} ({result.source_url})")
            candidates.append(result.model_dump())
        except Exception as e:
            print(f"  ! failed: {e}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(candidates)} candidate(s) -> {args.out}")


if __name__ == "__main__":
    main()
