"""
Callable tools for the six agents. Wraps the rule-based/local pieces (Russian
morphology, the idiom lexicon, the calibration guidelines doc, dataset stats)
as CrewAI @tool functions so an agent can call them mid-reasoning instead of
guessing from parameters alone -- this is the actual "agent vs. one-shot LLM
call" upgrade discussed when this architecture was designed. Network-backed
tools (StackExchange search) are also here but, like every other OpenRouter
call in this pipeline, need a terminal with normal internet access -- the
sandboxed build environment blocked outbound requests to *both* openrouter.ai
and api.stackexchange.com (confirmed 2026-09-20; see AGENTS.md).
"""

import json
import re
from pathlib import Path

import requests

from crewai.tools import tool

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
IDIOM_LEXICON_PATH = CONFIG_DIR / "idiom_lexicon_ru_en.json"
GUIDELINES_PATH = CONFIG_DIR / "calibration_guidelines.md"

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# Address pronouns / verb-form cues worth flagging as formality_shift candidates.
# Deliberately narrow (rule #1 in calibration_guidelines.md: real social stakes,
# not just any second-person word) -- pymorphy2 gives the grammatical tag,
# this just decides which tags are worth a look.
_ADDRESS_LEMMAS = {"ты", "вы", "тебя", "вас", "тебе", "вам", "тобой", "вами"}


def _morph_analyzer():
    # Imported lazily so the rest of this module still loads even if
    # pymorphy2 isn't installed yet (e.g. before `pip install -r requirements.txt`).
    import pymorphy2
    return pymorphy2.MorphAnalyzer()


_MORPH = None


@tool("flag_ru_address_pronoun")
def flag_ru_address_pronoun(source_text: str) -> str:
    """Scans a Russian sentence for ты/вы-family address pronouns using pymorphy2
    morphological analysis (not a keyword search -- catches inflected forms like
    'тебе', 'вас', 'тобой'). Returns a JSON string: {"found": bool, "matches": [
    {"token": str, "lemma": str, "tag": str}]}. Use this on RU source_text before
    deciding whether formality_shift is even a live candidate category --
    per calibration_guidelines.md rule 1, only flag formality_shift when a real
    address pronoun or informal/formal imperative is present with actual social
    stakes, not just because the sentence has a verb."""

    global _MORPH
    if _MORPH is None:
        _MORPH = _morph_analyzer()

    tokens = re.findall(r"[А-Яа-яЁё]+", source_text)
    matches = []
    for tok in tokens:
        parsed = _MORPH.parse(tok)[0]
        lemma = parsed.normal_form
        if lemma in _ADDRESS_LEMMAS or (
            "NPRO" in parsed.tag and ("2per" in parsed.tag or lemma in {"ты", "вы"})
        ):
            matches.append({"token": tok, "lemma": lemma, "tag": str(parsed.tag)})
    return json.dumps({"found": bool(matches), "matches": matches}, ensure_ascii=False)


@tool("lookup_idiom")
def lookup_idiom(phrase_or_source_text: str) -> str:
    """Looks up a Russian phrase against config/idiom_lexicon_ru_en.json's curated
    idiom entries (substring match against each entry's 'ru' field). Returns a
    JSON string: {"matched": bool, "entries": [...]} where each entry includes
    literal_en, idiomatic_en, literal_survives (whether the idiom's charge
    survives a direct translation -- per calibration_guidelines.md, several do
    and should be has_subtext=false even though they're idioms), and note. Use
    this before claiming a sentence contains an untranslatable idiom -- grounds
    the claim in the curated lexicon instead of the model inventing one (see
    calibration rule 10: an ungrounded cultural-reference claim was rejected in
    batch 1)."""

    with open(IDIOM_LEXICON_PATH, "r", encoding="utf-8") as f:
        lexicon = json.load(f)

    hits = [
        entry for entry in lexicon["entries"]
        if entry["ru"].lower() in phrase_or_source_text.lower()
    ]
    return json.dumps({"matched": bool(hits), "entries": hits}, ensure_ascii=False)


@tool("read_calibration_guidelines")
def read_calibration_guidelines() -> str:
    """Returns the full current text of config/calibration_guidelines.md -- the
    numbered rules an Annotator or Adjudicator Agent should apply before flagging
    has_subtext=true, plus any not-yet-approved proposals under "Pending review".
    Call this once per batch (not per sentence) and hold the rules in context
    rather than re-reading for every candidate."""

    return GUIDELINES_PATH.read_text(encoding="utf-8")


@tool("propose_guideline_update")
def propose_guideline_update(pattern_description: str, proposed_rule: str,
                              supporting_examples_json: str) -> str:
    """Appends a new entry under config/calibration_guidelines.md's "Pending
    review" section -- does NOT touch the numbered rules, which are only
    edited by hand after a maintainer reviews a proposal. supporting_examples_json is a
    JSON array of 2-4 source_text strings the pattern was drawn from. Returns
    the text that was appended. This is the Guideline Agent's only write path
    into the guidelines file -- every other agent only reads it."""

    examples = json.loads(supporting_examples_json)
    entry = (
        f"\n### Proposed: {pattern_description}\n\n"
        f"{proposed_rule}\n\n"
        f"Examples: {'; '.join(examples)}\n"
    )
    current = GUIDELINES_PATH.read_text(encoding="utf-8")
    marker = "(none yet)"
    if marker in current:
        updated = current.replace(marker, entry.strip() + "\n\n" + marker, 1)
    else:
        updated = current.rstrip("\n") + "\n" + entry
    GUIDELINES_PATH.write_text(updated, encoding="utf-8")
    return entry


@tool("search_ru_en_forums")
def search_ru_en_forums(query: str, max_results: int = 5) -> str:
    """Searches the Russian Language StackExchange (via the public
    api.stackexchange.com API, no key required for light use) for Q&A threads
    matching `query`. Returns a JSON string: {"results": [{"title": str,
    "url": str, "excerpt": str}]}. Intended for the Sourcing Agent to find
    bilingual-expert explanations of register/idiom/nuance questions that
    already exist as human-written commentary, rather than generating sentences
    from scratch. NETWORK NOTE: api.stackexchange.com is unreachable from the
    sandboxed device_bash environment this tool was developed in (confirmed
    2026-09-20, same org-egress-proxy block that affects openrouter.ai) -- run
    this from a real terminal with normal internet access."""

    resp = requests.get(
        "https://api.stackexchange.com/2.3/search/advanced",
        params={
            "site": "russian.stackexchange.com",
            "q": query,
            "sort": "relevance",
            "order": "desc",
            "pagesize": max_results,
            "filter": "withbody",
        },
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])[:max_results]

    def strip_html(html: str, limit: int = 400) -> str:
        text = re.sub(r"<[^>]+>", " ", html or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit]

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "excerpt": strip_html(item.get("body", "")),
        }
        for item in items
    ]
    return json.dumps({"results": results}, ensure_ascii=False)


@tool("dataset_category_stats")
def dataset_category_stats(stage_b_jsonl_paths_json: str) -> str:
    """Computes category/has_subtext distribution and simple near-duplicate
    detection across one or more stage_b_batchN.jsonl files. Argument is a JSON
    array of file paths, e.g. '["stage_b_batch1.jsonl", "stage_b_batch2.jsonl",
    "stage_b_batch3.jsonl"]'. Returns a JSON string matching the CoverageReport
    schema's fields (total_rows, category_counts, has_subtext_true_ratio,
    near_duplicate_clusters -- grouped by a crude shared-first-3-words key, a
    real similarity model is future work). Use this before writing a
    CoverageReport rather than estimating from memory."""

    paths = json.loads(stage_b_jsonl_paths_json)
    rows = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    category_counts: dict[str, int] = {}
    true_count = 0
    clusters: dict[str, list[str]] = {}
    for row in rows:
        cat = row.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if row.get("has_subtext"):
            true_count += 1
        text = (row.get("source_text") or "").strip()
        key = " ".join(text.split()[:3]).lower()
        if key:
            clusters.setdefault(key, []).append(text)

    near_dupes = [texts for texts in clusters.values() if len(texts) > 1]

    return json.dumps({
        "total_rows": len(rows),
        "category_counts": category_counts,
        "has_subtext_true_ratio": (true_count / len(rows)) if rows else 0.0,
        "near_duplicate_clusters": near_dupes,
    }, ensure_ascii=False)
