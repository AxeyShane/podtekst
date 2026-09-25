"""Pure-Python helpers for subtitle-pair cleaning and filtering.

No model dependencies here, so this module is cheap to import and unit-test.
"""
from __future__ import annotations

import re
import unicodedata

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")                 # <i>, {\an8}
_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)")        # [music], (sighs)
_SPEAKER_DASH = re.compile(r"^\s*[-–—]\s*")
_MUSIC = re.compile(r"[♪♫#]")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[\wЁё']+", re.UNICODE)

# Lines that are subtitle credits/ads rather than dialogue.
_JUNK = re.compile(
    r"(subtitle|субтитр|перевод[а-я]*\s*:|translated by|synced|www\.|http|"
    r"opensubtitles|addic7ed|downloaded from|релиз|озвуч)",
    re.IGNORECASE,
)


def clean_line(text: str) -> str:
    """Strip subtitle markup, sound cues, leading speaker dashes and extra spaces."""
    text = unicodedata.normalize("NFC", text or "")
    text = _TAGS.sub(" ", text)
    text = _BRACKETED.sub(" ", text)
    text = _MUSIC.sub(" ", text)
    text = _SPEAKER_DASH.sub("", text)
    text = text.replace("...", "…")
    return _WS.sub(" ", text).strip(" -–—")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def normalize_key(text: str) -> str:
    """Dedup key: lowercase, ё→е, letters/digits only."""
    t = text.lower().replace("ё", "е")
    return " ".join(_WORD.findall(t))


def is_multi_speaker(raw: str) -> bool:
    """A subtitle cue with two dash-led lines is two speakers -- skip those."""
    return len(re.findall(r"(^|\n|\s)[-–—]\s*\S", raw)) >= 2


# Letters that don't belong in the line's language, typical of mis-decoded (cp1251/cp1252) subtitles:
# «Ќа каком основании?», «ƒа, да.». RU lines may hold Russian Cyrillic and ASCII Latin (names, brands);
# EN lines ASCII plus Latin-1 accents (café, naïve).
_RU_FOREIGN = re.compile(r"[^\W\d_А-Яа-яЁёA-Za-z]")
_EN_FOREIGN = re.compile(r"[^\W\d_A-Za-zÀ-ÖØ-öø-ÿ]")

# Sounds and filler that make a line "divergent" from MT without carrying any meaning.
_INTERJECTIONS = {
    "о", "ох", "ах", "ай", "ой", "эй", "ух", "уф", "ха", "хе", "хи", "хм", "м", "мм", "ммм", "а", "э", "у", "ну",
    "ага", "угу", "эх", "фу", "тсс", "бр", "не", "нет", "да", "стоп",
    "oh", "ah", "aah", "ooh", "uh", "um", "hm", "hmm", "mm", "eh", "ha", "heh", "hey", "huh", "wow", "whoa",
    "no", "yes", "yeah", "stop", "o",
}


def has_mojibake(ru: str, en: str) -> bool:
    return bool(_RU_FOREIGN.search(ru) or _EN_FOREIGN.search(en))


def is_interjection_only(text: str) -> bool:
    """True when the line has no content words: only interjections, single letters or one repeated
    token («О, о-о-о.», «Хе-хе-хе.», «Не-не-не!», «Б, А»)."""
    toks = [t for t in re.split(r"[\W_]+", text.lower().replace("ё", "е")) if t and not t.isdigit()]
    content = {t for t in toks if len(t) > 1 and t not in _INTERJECTIONS}
    return not content or len(set(toks)) < 2


def latin_share(text: str) -> float:
    """Fraction of letters that are Latin (RU lines may carry a brand name or two)."""
    lat, cyr = len(LATIN.findall(text)), len(CYRILLIC.findall(text))
    return lat / max(1, lat + cyr)


def pair_passes(ru: str, en: str, min_words: int = 3, max_words: int = 25,
                max_len_ratio: float = 2.5) -> tuple[bool, str]:
    """Cheap filters before any model runs. Returns (ok, reason_if_rejected)."""
    if not ru or not en:
        return False, "empty"
    if not CYRILLIC.search(ru) or latin_share(ru) > 0.3:
        return False, "ru_script"
    if CYRILLIC.search(en) or not LATIN.search(en):
        return False, "en_script"
    if _JUNK.search(ru) or _JUNK.search(en):
        return False, "credits"
    wr, we = word_count(ru), word_count(en)
    if not (min_words <= wr <= max_words and min_words <= we <= max_words):
        return False, "length"
    if has_mojibake(ru, en):
        return False, "mojibake"
    if is_interjection_only(ru) or is_interjection_only(en):
        return False, "interjection"
    ratio = max(wr, we) / max(1, min(wr, we))
    if ratio > max_len_ratio:
        return False, "len_ratio"
    if normalize_key(ru) == normalize_key(en):
        return False, "identical"
    return True, ""


def film_id_from_ids_line(line: str) -> str | None:
    """OPUS .ids lines look like 'en/1999/12345/6789.xml.gz<TAB>ru/1999/12345/...'.
    Returns 'year/imdb' as a best-effort film key, or None if the format differs."""
    first = line.split("\t", 1)[0]
    parts = first.split("/")
    if len(parts) >= 3 and parts[1].isdigit():
        return f"{parts[1]}/{parts[2]}"
    return None
