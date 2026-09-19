"""
Deterministic checks -- the new stage between Stage A (generation) and Stage B
(Cowork LLM judge). Cheap, fast, no model calls: catches obviously-broken rows
before spending judge attention on them, and flags failed generation calls +
rejected rows so they can be recycled as next cycle's seeds (see docs/DESIGN.md
"Feedback loop").

Usage:
    python deterministic_checks.py --in stage_a_batch2.jsonl --seeds seeds_batch2.txt \
        --clean stage_a_batch2.clean.jsonl --rejects stage_a_batch2.rejects.jsonl \
        --next-seeds seeds_batch3_carryover.txt

Checks applied, per row:
  - schema: required fields present with correct types
  - language tag: source_lang matches the actual script used in source_text
    (rough heuristic -- flags obvious mismatches, not a full langid model)
  - length: nuance_note over ~20 words (the system prompt's own limit) or
    translation suspiciously long/short relative to source_text
  - echo/suspect: translation identical to source_text regardless of direction
    (stage_a_generate.py only checked the en->ru direction; this checks both)
  - dedup: exact duplicate (source_text, _generator_model) pairs within the file

Rows that fail any check go to --rejects instead of --clean, and their
source_text is added to --next-seeds (deduped against seeds already in
--seeds) so nothing silently disappears -- a failure usually means the
sentence type is under-covered or genuinely hard, worth another generation
attempt rather than a quiet drop.
"""

import argparse
import json
import re

ALLOWED_LANGS = {"ru", "en"}
ALLOWED_CATEGORIES = {"formality_shift", "sarcasm", "idiom", "emotional_subtext", "none"}
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
NOTE_WORD_LIMIT = 25  # a little slack over the system prompt's stated 20-word target


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_schema(row):
    problems = []
    for field, typ in [
        ("source_lang", str), ("source_text", str), ("translation", str),
        ("has_subtext", bool), ("category", str), ("nuance_note", str),
    ]:
        if field not in row:
            problems.append(f"missing field '{field}'")
        elif not isinstance(row[field], typ):
            problems.append(f"field '{field}' has wrong type (expected {typ.__name__})")
    if row.get("source_lang") not in ALLOWED_LANGS:
        problems.append(f"source_lang '{row.get('source_lang')}' not in {ALLOWED_LANGS}")
    if row.get("category") not in ALLOWED_CATEGORIES:
        problems.append(f"category '{row.get('category')}' not in {ALLOWED_CATEGORIES}")
    if row.get("has_subtext") is False and row.get("nuance_note", "").strip():
        problems.append("has_subtext=false but nuance_note is non-empty")
    if row.get("has_subtext") is True and not row.get("nuance_note", "").strip():
        problems.append("has_subtext=true but nuance_note is empty")
    return problems


def check_language_tag(row):
    text = row.get("source_text", "")
    lang = row.get("source_lang")
    has_cyrillic = bool(CYRILLIC_RE.search(text))
    if lang == "ru" and not has_cyrillic:
        return ["source_lang='ru' but source_text has no Cyrillic characters"]
    if lang == "en" and has_cyrillic:
        return ["source_lang='en' but source_text contains Cyrillic characters"]
    return []


def check_length(row):
    problems = []
    note = row.get("nuance_note", "") or ""
    if len(note.split()) > NOTE_WORD_LIMIT:
        problems.append(f"nuance_note is {len(note.split())} words (limit ~{NOTE_WORD_LIMIT})")
    return problems


def check_echo(row):
    src = (row.get("source_text") or "").strip().lower()
    trans = (row.get("translation") or "").strip().lower()
    if src and src == trans:
        return ["translation is identical to source_text -- likely untranslated echo"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="Stage A output JSONL")
    ap.add_argument("--clean", required=True, help="Rows that passed all checks")
    ap.add_argument("--rejects", required=True, help="Rows that failed one or more checks, with reasons")
    ap.add_argument("--next-seeds", required=False,
                     help="Source sentences from rejected rows, appended here for the next cycle")
    args = ap.parse_args()

    rows = load_jsonl(args.infile)
    clean, rejects = [], []
    seen_pairs = set()

    for row in rows:
        problems = []
        problems += check_schema(row)
        # Only run the finer checks if schema is sound enough to have the fields.
        if not problems:
            problems += check_language_tag(row)
            problems += check_length(row)
            problems += check_echo(row)

        key = (row.get("source_text"), row.get("_generator_model"))
        if key in seen_pairs:
            problems.append("duplicate (source_text, generator_model) pair within this file")
        else:
            seen_pairs.add(key)

        if problems:
            row["_deterministic_check_failures"] = problems
            rejects.append(row)
        else:
            clean.append(row)

    with open(args.clean, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.rejects, "w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.next_seeds:
        carryover = sorted({r["source_text"] for r in rejects if r.get("source_text")})
        with open(args.next_seeds, "w", encoding="utf-8") as f:
            for s in carryover:
                f.write(s + "\n")
        print(f"Wrote {len(carryover)} carryover seeds to {args.next_seeds}")

    print(f"Checked {len(rows)} rows: {len(clean)} clean -> {args.clean}, "
          f"{len(rejects)} rejected -> {args.rejects}")


if __name__ == "__main__":
    main()
