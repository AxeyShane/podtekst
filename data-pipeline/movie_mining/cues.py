"""Lightweight nuance cues used to boost ranking (not to label anything).

- ты/вы address detection via pymorphy2 (same lemma logic as agents/tools.py,
  reimplemented here so this module doesn't import crewai).
- Idiom lexicon hits from config/idiom_lexicon_ru_en.json.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_ADDRESS_LEMMAS = {"ты", "вы", "твой", "ваш"}
_RU_TOKEN = re.compile(r"[А-Яа-яЁё]+")
LEXICON_PATH = Path(__file__).resolve().parent.parent / "config" / "idiom_lexicon_ru_en.json"


@lru_cache(maxsize=1)
def _morph():
    # pymorphy3 is the maintained fork (pymorphy2 breaks on Python 3.11+).
    for mod in ("pymorphy3", "pymorphy2"):
        try:
            return __import__(mod).MorphAnalyzer()
        except Exception:  # ImportError, or pymorphy2 failing on newer Pythons
            continue
    return None


def address_register(ru: str) -> str | None:
    """Returns 'ty', 'vy', 'both' or None. Falls back to a plain token match if
    pymorphy2 isn't installed (misses nothing common, just less precise)."""
    morph = _morph()
    found = set()
    for tok in _RU_TOKEN.findall(ru):
        lemma = morph.parse(tok)[0].normal_form if morph else tok.lower()
        if lemma in {"ты", "твой"} or (not morph and lemma in {"тебя", "тебе", "тобой"}):
            found.add("ty")
        elif lemma in {"вы", "ваш"} or (not morph and lemma in {"вас", "вам", "вами"}):
            found.add("vy")
    if not found:
        return None
    return "both" if len(found) == 2 else found.pop()


@lru_cache(maxsize=200_000)
def _known(tok: str) -> bool:
    return _morph().word_is_known(tok)


def _doubled_capital(tok: str) -> bool:
    """«Мможет»: capital + same letter again, unknown as written but a word without the
    first letter. Real words starting with a doubled letter (ссора, ввод) are known, so pass."""
    return (len(tok) > 2 and tok[0].isupper() and tok[1] == tok[0].lower()
            and not _known(tok.lower()) and _known(tok[1:]))


def ru_fluency_issue(ru: str, max_unknown: int = 1) -> str | None:
    """Cheap check for broken Russian (fan-sub typos, OCR junk, machine output).
    Returns 'doubled_capital', 'unknown_words' (more than max_unknown Cyrillic tokens
    missing from the pymorphy dictionary) or None. Needs pymorphy3 (returns None without it)."""
    if _morph() is None:
        return None
    toks = _RU_TOKEN.findall(ru)
    if any(_doubled_capital(t) for t in toks):
        return "doubled_capital"
    unknown = sum(1 for t in toks if not _known(t.lower()))
    return "unknown_words" if unknown > max_unknown else None


@lru_cache(maxsize=1)
def _lexicon() -> list[str]:
    try:
        data = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e["ru"].lower().replace("ё", "е") for e in data.get("entries", []) if e.get("ru")]


def idiom_hits(ru: str) -> list[str]:
    low = ru.lower().replace("ё", "е")
    return [phrase for phrase in _lexicon() if phrase in low]
