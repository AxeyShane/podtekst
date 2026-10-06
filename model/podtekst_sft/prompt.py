"""One prompt format for training, evaluation and on-device inference.

The same SYSTEM_PROMPT and build_messages() must be used everywhere: a fine-tuned model
learns this exact framing, and the keyboard will send it at runtime. Keep it short -- every
token here is prefill latency on the phone.
"""
import json
import re

CATEGORIES = ("none", "formality_shift", "idiom", "sarcasm", "emotional_subtext")

SYSTEM_PROMPT = (
    "You translate chat messages between Russian and English for a phone keyboard. "
    "Translate the message into the other language, then say whether a plain translation "
    "loses nuance. Reply with JSON only: "
    '{"translation": str, "has_subtext": bool, '
    '"category": "none|formality_shift|idiom|sarcasm|emotional_subtext", "nuance_note": str}. '
    'Use category "none" and an empty nuance_note when nothing is lost.'
)

GENDER_WORDS = {"female": "female", "male": "male"}


def user_content(source_text: str, speaker_gender: str | None = None,
                 addressee_gender: str | None = None) -> str:
    """The user turn: the message itself, plus optional gender context the keyboard can know
    (the user's own gender from settings, the contact's from the chat)."""
    hints = []
    if speaker_gender in GENDER_WORDS:
        hints.append(f"speaker: {speaker_gender}")
    if addressee_gender in GENDER_WORDS:
        hints.append(f"addressee: {addressee_gender}")
    return source_text if not hints else f"{source_text}\n({', '.join(hints)})"


def target_json(row: dict) -> str:
    """The assistant turn for a dataset row, in a fixed key order."""
    has = bool(row["has_subtext"])
    return json.dumps({
        "translation": row["translation"],
        "has_subtext": has,
        "category": row["category"] if has else "none",
        "nuance_note": (row.get("nuance_note") or "") if has else "",
    }, ensure_ascii=False)


def row_genders(row: dict) -> tuple:
    g = lambda k: row.get(k) or row.get("_" + k)
    return g("speaker_gender"), g("addressee_gender")


def build_messages(row: dict, with_answer: bool = True) -> list[dict]:
    """Chat messages for one dataset row (system, user[, assistant])."""
    sg, ag = row_genders(row)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content(row["source_text"], sg, ag)}]
    if with_answer:
        msgs.append({"role": "assistant", "content": target_json(row)})
    return msgs


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_reply(text: str) -> dict | None:
    """Model reply -> normalised dict, or None when there is no usable JSON.
    Tolerates code fences, leading chatter and a stray <think></think> block."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("translation"), str):
        return None
    has = d.get("has_subtext")
    if isinstance(has, str):
        has = has.strip().lower() == "true"
    cat = d.get("category") if d.get("category") in CATEGORIES else "none"
    has = bool(has) and cat != "none"
    return {"translation": d["translation"], "has_subtext": has,
            "category": cat if has else "none", "nuance_note": str(d.get("nuance_note") or "")}
