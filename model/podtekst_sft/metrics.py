"""Eval metrics for the structured output: JSON validity, subtext detection, per-category
scores and translation chrF. Pure Python except chrF (sacrebleu, optional)."""
from collections import Counter

from .prompt import CATEGORIES


def detection_scores(gold: list[bool], pred: list[bool]) -> dict:
    """has_subtext as a binary detector. false_positive_rate = flagged nuance on rows with none
    (the PRODUCT.md metric users notice most: a keyboard that cries wolf)."""
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    div = lambda a, b: round(a / b, 4) if b else None
    return {"precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "false_positive_rate": div(fp, fp + tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def category_scores(gold: list[str], pred: list[str]) -> dict:
    out = {}
    for c in CATEGORIES:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if (tp + fn) else None)
        out[c] = {"support": tp + fn, "precision": None if prec is None else round(prec, 4),
                  "recall": None if rec is None else round(rec, 4), "f1": None if f1 is None else round(f1, 4)}
    out["accuracy"] = round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4) if gold else None
    out["confusion"] = {f"{g}->{p}": n for (g, p), n in Counter(zip(gold, pred)).most_common() if g != p}
    return out


def chrf(hyps: list[str], refs: list[str]) -> float | None:
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        return None
    return round(CHRF().corpus_score(hyps, [refs]).score, 2) if hyps else None


def summarize(rows: list[dict], preds: list[dict | None]) -> dict:
    """rows: gold dataset rows; preds: parse_reply() output per row (None = unparseable).
    Unparseable replies count as 'no subtext' and an empty translation, so bad JSON is
    punished in every metric rather than silently dropped."""
    ok = [p is not None for p in preds]
    pp = [p or {"translation": "", "has_subtext": False, "category": "none"} for p in preds]
    gold_cat = [r["category"] if r["has_subtext"] else "none" for r in rows]
    report = {
        "n": len(rows),
        "json_valid_rate": round(sum(ok) / len(rows), 4) if rows else None,
        "detection": detection_scores([bool(r["has_subtext"]) for r in rows], [p["has_subtext"] for p in pp]),
        "category": category_scores(gold_cat, [p["category"] for p in pp]),
        "chrf": chrf([p["translation"] for p in pp], [r["translation"] for r in rows]),
    }
    for lang in ("ru", "en"):
        idx = [i for i, r in enumerate(rows) if r.get("source_lang") == lang]
        if idx:
            report[f"chrf_{lang}_source"] = chrf([pp[i]["translation"] for i in idx], [rows[i]["translation"] for i in idx])
    return report
