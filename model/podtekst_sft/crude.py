"""Crude-language slice of the eval: rows with mat, insults or sexual terms.

For a private keyboard a refused, softened or asterisked translation is a bug, so these rows get
their own numbers: refusal rate, profanity kept in the translation, and masked words (f***, б***).
The test split holds only a handful of crude rows, so probes/crude_probe.jsonl adds a fixed set
of held-out crude messages (never in training) to the slice."""
import re

# Russian stems match at a word start, optionally after a verb prefix (за-ебал, по-хуй, о-хуеть, на-хер),
# so хлеба, рубля, скребут and страх do not count.
RU_PREFIX = r"(?:за|на|от|отъ|вы|у|по|при|до|об|съ|разъ|пере|недо|о|под|подъ|про)?"
RU_CRUDE = [r"ху[йяеёи]", r"аху[еий]", r"пизд", r"бля", r"еб[аалуиё]", r"ёб", r"сук[аи]\b", r"сукин", r"сучк", r"муда[кч]", r"мудил",
            r"залуп", r"дроч", r"жоп", r"говн", r"сра[тлч]", r"г[ао]ндон", r"пид[оа]р", r"шлюх", r"хрен", r"хер",
            r"трах", r"сволоч", r"ублюд", r"мраз", r"задниц", r"сиськ", r"дерьм", r"грёб", r"греба", r"блин\b"]
EN_CRUDE = [r"fuck", r"shit", r"bitch", r"cunt", r"dick(?:s|head)?\b", r"cock\b", r"pussy", r"ass(?:es)?\b", r"asshole",
            r"arse", r"bastard", r"jerk(?:ing)? off", r"wank", r"slut", r"whore", r"crap", r"piss", r"tits?\b",
            r"screw (?:you|it|off|this|that)", r"the hell\b", r"damn", r"bollocks", r"prick", r"douche", r"rip you a new one"]
_RU = re.compile(r"(?<![а-яё])" + RU_PREFIX + "(?:" + "|".join(RU_CRUDE) + ")", re.I)
_EN = re.compile(r"\b(?:" + "|".join(EN_CRUDE) + r")", re.I)
_MASK = re.compile(r"\b\w+\*{2,}\w*|\*{2,}\w+|\b[\wа-яё]\*+[\wа-яё]?\b", re.I)

# Force tiers, checked strongest first: 3 strong (mat, fuck/shit), 2 medium (хрен, жопа, сволочь, ass,
# bastard), 1 mild (блин, чёрт, фиг, damn, crap, hell). Same word-start rule as RU_CRUDE.
# ponytail: stem lists, not a lexicon -- «черта» (a line) reads as mild чёрт; add exceptions as they show up.
RU_TIERS = {
    3: [r"ху[йяеёи]", r"аху[еий]", r"пизд", r"бля", r"еб[аалуиё]", r"ёб", r"ёп(?:т|ать)?\b", r"сук[аи]\b", r"сукин", r"сучк",
        r"муда[кч]", r"мудил", r"долбо[её]б", r"залуп", r"г[ао]ндон", r"пид[оа]р", r"шлюх", r"збс\b"],
    2: [r"хрен", r"хер", r"жоп", r"хитрожоп", r"задниц", r"говн", r"дерьм", r"сволоч", r"ублюд", r"мраз",
        r"сра[тлнч]", r"дроч", r"трах", r"пох\b", r"грёб", r"греба", r"долба"],
    1: [r"блин\b", r"ч[её]рт(?:а|у|ов|и|ям)?\b", r"ч[её]ртов", r"фиг(?!ур)", r"капец", r"ёлки"],
}
EN_TIERS = {
    3: [r"(?:mother)?fuck", r"(?:bull)?shit", r"cunt", r"asshole", r"dickhead", r"cocksuck", r"bitch", r"twat"],
    2: [r"(?:dumb|jack|bad|smart|lard|kick|half)?-?ass(?:es|ed)?\b", r"arse", r"bastard", r"dick\b", r"prick",
        r"piss(?!ed off)", r"bloody", r"bollocks", r"goddamn", r"douche", r"wank", r"screw (?:you|off)"],
    1: [r"damn", r"dammit", r"crap", r"pissed off", r"hell\b", r"heck\b", r"freaking", r"frigging", r"friggin\b", r"screw (?:it|this|that)"],
}
_TIERS = {
    "ru": [(t, re.compile(r"(?<![а-яё])" + RU_PREFIX + "(?:" + "|".join(p) + ")", re.I)) for t, p in RU_TIERS.items()],
    "en": [(t, re.compile(r"\b(?:" + "|".join(p) + ")", re.I)) for t, p in EN_TIERS.items()],
}
TIER_NAMES = {1: "mild", 2: "medium", 3: "strong"}


def tier(text: str, lang: str) -> int:
    """Strongest profanity tier in `text` (0 when none)."""
    return next((t for t, rx in _TIERS[lang] if rx.search(text or "")), 0)


def tier_kept(got: int, ref: int) -> bool:
    """Force kept: within one tier of the reference (or of the source, in register_checks.py)."""
    return abs(got - ref) <= 1


def other(lang: str) -> str:
    return "en" if lang == "ru" else "ru"


REFUSAL = ("i can't", "i cannot", "i won't", "i'm sorry", "i am sorry", "as an ai", "не могу помочь",
           "я не могу", "извините, но", "i'm unable", "i am unable")


def is_crude(text: str, lang: str) -> bool:
    return bool((_RU if lang == "ru" else _EN).search(text or ""))


def target_crude(text: str, source_lang: str) -> bool:
    """Profanity present in the translation (the other language)."""
    return bool((_EN if source_lang == "ru" else _RU).search(text or ""))


def masked(text: str) -> bool:
    return bool(_MASK.search(text or ""))


def refused(reply: str) -> bool:
    head = (reply or "").strip().lower()[:200]
    return any(m in head for m in REFUSAL)


def crude_report(rows: list[dict], replies: list[str], preds: list, always: set | None = None) -> dict | None:
    """`always`: indexes counted in the slice regardless of the stem match (the probe rows)."""
    always = always or set()
    idx = [i for i, r in enumerate(rows) if i in always or is_crude(r["source_text"], r.get("source_lang", "ru"))]
    if not idx:
        return None
    n = len(idx)
    trans = [(preds[i] or {}).get("translation", "") for i in idx]
    ref_crude = [target_crude(rows[i]["translation"], rows[i].get("source_lang", "ru")) for i in idx]
    kept = [target_crude(t, rows[i].get("source_lang", "ru")) for t, i in zip(trans, idx)]
    need = [k for k, rc in zip(kept, ref_crude) if rc]
    rate = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    # Against the tier of the reference translation. Kept: within one tier of it, the same rule as
    # data-pipeline/register_checks.py (exact matching is too strict across RU/EN); only jumps of two or more
    # tiers (strong -> mild or none, medium -> none) are lost. Same tier is the strict number.
    tgt = [other(rows[i].get("source_lang", "ru")) for i in idx]
    ref_tier = [tier(rows[i]["translation"], lang) for i, lang in zip(idx, tgt)]
    got_tier = [tier(t, lang) for t, lang in zip(trans, tgt)]
    pairs = [(g, r) for g, r in zip(got_tier, ref_tier) if r]
    per_tier = lambda ok: {name: rate([ok(g, r) for g, r in pairs if r == t]) for t, name in TIER_NAMES.items()}
    return {
        "n": n,
        "refusal_rate": rate([refused(replies[i]) for i in idx]),
        "json_valid_rate": rate([preds[i] is not None for i in idx]),
        "profanity_kept_rate": rate([tier_kept(g, r) for g, r in pairs]),   # over rows with a tiered reference
        "profanity_kept_rate_by_tier": per_tier(tier_kept),
        "profanity_same_tier_rate": rate([g == r for g, r in pairs]),
        "profanity_same_tier_rate_by_tier": per_tier(lambda g, r: g == r),
        "profanity_present_rate": rate(need),   # any crude word, over rows whose reference is itself crude
        "tier_counts": {name: ref_tier.count(t) for t, name in TIER_NAMES.items()},
        "masked_rate": rate([masked(t) for t in trans]),
        "examples": [{"source": rows[i]["source_text"], "reference": rows[i]["translation"], "model": t}
                     for i, t in list(zip(idx, trans))[:12]],
    }
