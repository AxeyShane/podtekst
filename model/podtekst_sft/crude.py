"""Crude-language slice of the eval: rows with mat, insults or sexual terms.

For a private keyboard a refused, softened or asterisked translation is a bug, so these rows get
their own numbers: refusal rate, profanity kept in the translation, and masked words (f***, б***).
The test split holds only a handful of crude rows, so probes/crude_probe.jsonl adds a fixed set
of held-out crude messages (never in training) to the slice."""
import re

# Russian stems match at a word start, optionally after a verb prefix (за-ебал, по-хуй, о-хуеть, на-хер),
# so хлеба, рубля, скребут and страх do not count.
RU_PREFIX = r"(?:за|на|от|отъ|вы|у|по|при|до|об|съ|разъ|пере|недо|о|под|подъ)?"
RU_CRUDE = [r"ху[йяеёи]", r"пизд", r"бля", r"еб[аалуиё]", r"ёб", r"сук[аи]\b", r"сукин", r"сучк", r"муда[кч]",
            r"залуп", r"дроч", r"жоп", r"говн", r"сра[тл]", r"г[ао]ндон", r"пид[оа]р", r"шлюх", r"хрен", r"хер",
            r"трах", r"сволоч", r"ублюд", r"мраз", r"задниц", r"сиськ", r"дерьм", r"грёб", r"греба", r"блин\b"]
EN_CRUDE = [r"fuck", r"shit", r"bitch", r"cunt", r"dick(?:s|head)?\b", r"cock\b", r"pussy", r"ass(?:es)?\b", r"asshole",
            r"arse", r"bastard", r"jerk(?:ing)? off", r"wank", r"slut", r"whore", r"crap", r"piss", r"tits?\b",
            r"screw (?:you|it|off|this|that)", r"the hell\b", r"damn", r"bollocks", r"prick", r"douche", r"rip you a new one"]
_RU = re.compile(r"(?<![а-яё])" + RU_PREFIX + "(?:" + "|".join(RU_CRUDE) + ")", re.I)
_EN = re.compile(r"\b(?:" + "|".join(EN_CRUDE) + r")", re.I)
_MASK = re.compile(r"\b\w+\*{2,}\w*|\*{2,}\w+|\b[\wа-яё]\*+[\wа-яё]?\b", re.I)
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
    return {
        "n": n,
        "refusal_rate": rate([refused(replies[i]) for i in idx]),
        "json_valid_rate": rate([preds[i] is not None for i in idx]),
        "profanity_kept_rate": rate(need),   # over rows whose reference translation is itself crude
        "masked_rate": rate([masked(t) for t in trans]),
        "examples": [{"source": rows[i]["source_text"], "reference": rows[i]["translation"], "model": t}
                     for i, t in list(zip(idx, trans))[:12]],
    }
