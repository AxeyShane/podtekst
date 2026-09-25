"""
Stage B automated pre-filter (GLM-5.3), sitting between deterministic_checks.py
and the Cowork LLM judge.

Rationale (see docs/DESIGN.md): Cowork's usage is session-bounded and
semi-manual, so it's the bottleneck in the verification cycle. Most sentences
in a batch aren't actually contested -- every surviving generator already
agrees on has_subtext and category, and the only real work left is picking
the clearest phrasing among candidates that already agree. That's a cheap,
well-scoped task a fast model can do unattended, freeing Cowork sessions to
spend judgment only on the sentences generators actually disagree about.

This script does NOT re-decide has_subtext/category -- it only picks/polishes
a translation + nuance_note for groups where every surviving candidate
already agrees on both. Anything less than full agreement (the default; see
--min-agreement) is passed through untouched to the Cowork queue, unchanged
from today's process.

IMPORTANT: unanimous does not mean correct. All generators can share the
same blind spot. Auto-resolved rows still need to appear in the human
calibration sample (docs/DESIGN.md step 4) -- don't only calibration-check
Cowork's contested-row verdicts.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python stage_b_prefilter.py --in stage_a_batch2.clean.jsonl \
        --resolved stage_b_batch2_prefilter_resolved.jsonl \
        --needs-cowork stage_a_batch2_needs_cowork.jsonl

Input is expected to be the *clean* output of deterministic_checks.py (rows
that already passed schema/echo/dedup checks). Candidates flagged
_translation_suspect are still excluded from consideration here as a second
line of defense, in case an ungated file is passed in.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up OPENROUTER_API_KEY from a .env file in the cwd, if present
except ImportError:
    pass  # python-dotenv not installed -- fall back to a real exported env var

DEFAULT_MODEL = "z-ai/glm-5.3-flash"

PREFILTER_SYSTEM_PROMPT = """You are picking the best translation for training data. Several
independent translation models already agree on whether this sentence carries hidden nuance
(has_subtext) and what category it falls into -- that judgment is SETTLED, do not second-guess it.
Your only job: given the source sentence and a list of candidate translations + nuance notes that
all share the same has_subtext/category verdict, either pick the clearest one verbatim or lightly
merge them into one clean answer. Respond with ONLY a JSON object (no markdown fences, no preamble):

{
  "translation": "...",
  "nuance_note": "one short sentence, under 20 words, or empty string if has_subtext is false"
}
"""


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def group_by_source(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_text"]].append(r)
    return list(groups.items())


def eligible_for_autoresolve(valid_candidates, min_agreement):
    """Returns (eligible: bool, has_subtext, category, agreeing_candidates, fraction)."""
    if not valid_candidates:
        return False, None, None, [], 0.0
    subtext_counts = Counter(c.get("has_subtext") for c in valid_candidates)
    top_subtext, top_count = subtext_counts.most_common(1)[0]
    fraction = top_count / len(valid_candidates)
    if fraction < min_agreement:
        return False, None, None, [], fraction

    agreeing = [c for c in valid_candidates if c.get("has_subtext") == top_subtext]
    cats = set(c.get("category") for c in agreeing)
    if len(cats) != 1:
        return False, None, None, [], fraction

    return True, top_subtext, cats.pop(), agreeing, fraction


# Polish-call defaults, overridable per model in config/models.json. Diagnosed 2026-09-25 on batch 4:
# with max_tokens=400 and no reasoning setting, fallback providers spent up to 472 reasoning tokens
# and returned finish_reason=length with empty content (25 of 27 failures) or cut-off JSON (2).
# glm-5.3-flash can't turn reasoning off ("mandatory for this endpoint"), but effort=low keeps it
# to a few dozen tokens; 1500 leaves ample room for a two-field JSON answer.
POLISH_DEFAULTS = {"max_tokens": 1500, "reasoning": {"effort": "low"},
                   "response_format": {"type": "json_object"}}
DEFAULT_FALLBACK_MODEL = "deepseek/deepseek-v4-flash"


class PolishError(RuntimeError):
    """The model answered, but not with usable JSON (empty, cut off by length, or unparseable)."""


def build_user_msg(source_text, has_subtext, category, agreeing):
    candidate_lines = "\n".join(
        f'- translation: "{c.get("translation")}" | nuance_note: "{c.get("nuance_note", "")}"'
        for c in agreeing
    )
    return (
        f"Source sentence: {source_text}\n"
        f"Settled verdict: has_subtext={has_subtext}, category={category}\n"
        f"Candidates (all agree on the verdict above):\n{candidate_lines}\n"
    )


def call_polish(user_msg, model_cfg, api_key, post=None):
    """One polish call. Returns (parsed_json, log) where log has model, finish_reason, provider and
    token usage; raises PolishError (log attached as .log) on empty / cut-off / unparseable output."""
    post = post or requests.post
    params = {k: model_cfg.get(k, v) for k, v in POLISH_DEFAULTS.items()}
    provider = model_cfg.get("provider")
    if provider and params.get("response_format"):
        provider = {**provider, "require_parameters": True}   # skip routes that would ignore JSON mode
    response = post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_cfg.get("api_model") or model_cfg["slug"],
            "messages": [
                {"role": "system", "content": PREFILTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            **{k: v for k, v in params.items() if v is not None},
            **({"provider": provider} if provider else {}),
        },
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code} from OpenRouter: {response.text[:300]}")
    data = response.json()
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    log = {"model": model_cfg["slug"], "provider": data.get("provider"),
           "finish_reason": choice.get("finish_reason"),
           "completion_tokens": usage.get("completion_tokens"),
           "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
    text = (choice["message"].get("content") or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        if not text:
            raise ValueError("empty content")
        if start == -1 or end < start:
            raise ValueError(f"no JSON object: {text[:120]}")
        parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict) or not parsed.get("translation"):
            raise ValueError(f"JSON without a translation: {text[:120]}")
    except ValueError as e:
        err = PolishError(f"{e} (finish_reason={log['finish_reason']}, "
                          f"reasoning_tokens={log['reasoning_tokens']})")
        err.log = {**log, "error": str(e)[:200]}
        raise err
    return parsed, log


def polish(source_text, has_subtext, category, agreeing, primary, fallback, api_key, post=None,
           sleep=time.sleep):
    """primary, primary again, then the fallback model. Returns (picked, path, call_logs) with path
    one of primary / primary_retry / fallback_model, or (None, "unpolished", call_logs)."""
    user_msg = build_user_msg(source_text, has_subtext, category, agreeing)
    logs = []
    attempts = [("primary", primary), ("primary_retry", primary)]
    if fallback:
        attempts.append(("fallback_model", fallback))
    for i, (path, model_cfg) in enumerate(attempts):
        if i:
            sleep(1.0)
        try:
            picked, log = call_polish(user_msg, model_cfg, api_key, post=post)
            logs.append({**log, "path": path})
            return picked, path, logs
        except PolishError as e:
            logs.append({**e.log, "path": path})
        except Exception as e:       # HTTP error / timeout: same recovery chain
            logs.append({"model": model_cfg["slug"], "path": path, "error": str(e)[:200]})
    return None, "unpolished", logs


def load_polish_models(config_path, model_override=None):
    """(primary_cfg, fallback_cfg or None) from config/models.json."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    primary = dict(cfg.get("stage_b_prefilter_model") or {"slug": DEFAULT_MODEL})
    if model_override:
        primary = {"slug": model_override}
    fallback = cfg.get("stage_b_prefilter_fallback_model")
    return primary, (dict(fallback) if fallback else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True,
                     help="deterministic_checks.py's clean output")
    ap.add_argument("--resolved", required=True, help="Auto-resolved verified rows, Stage B schema")
    ap.add_argument("--needs-cowork", required=True,
                     help="Contested groups' raw candidate rows, unchanged -- feed to Cowork as usual")
    ap.add_argument("--config", default="config/models.json")
    ap.add_argument("--model", default=None,
                     help="Override the model slug (default: config's stage_b_prefilter_model, "
                          f"falling back to {DEFAULT_MODEL})")
    ap.add_argument("--min-agreement", type=float, default=1.0,
                     help="Fraction of candidates that must agree on has_subtext+category to "
                          "auto-resolve (default 1.0 = strict unanimity, no dissenters at all)")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--repolish-from", default=None,
                     help="Re-polish only the rows in this earlier --resolved file that fell back to "
                          "unpolished text (candidates come from --in); writes the full, updated "
                          "file to --resolved and leaves --needs-cowork untouched")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment first.")

    primary, fallback = load_polish_models(args.config, args.model)
    rows = load_jsonl(args.infile)
    groups = group_by_source(rows)
    print(f"Loaded {len(rows)} candidate rows across {len(groups)} sentences.")
    print(f"Polish model: {primary['slug']}  |  fallback: {(fallback or {}).get('slug')}  |  "
          f"min agreement: {args.min_agreement}")

    paths, finish = Counter(), Counter()

    def resolve(source_text, cands, valid, has_subtext, category, agreeing, fraction):
        picked, path, logs = polish(source_text, has_subtext, category, agreeing, primary, fallback, api_key)
        paths[path] += 1
        finish.update(f"{l['model']}:{l.get('finish_reason') or 'error'}" for l in logs)
        last = logs[-1] if logs else {}
        tokens = f"{last.get('completion_tokens')} tok / {last.get('reasoning_tokens')} reasoning"
        row = {
            "source_lang": cands[0]["source_lang"],
            "source_text": source_text,
            "has_subtext": has_subtext,
            "category": category,
            "_stage_b_polish_path": path,
            "_stage_b_polish_calls": logs,
            "_stage_b_candidate_count": len(valid),
            "_stage_b_agreement_fraction": fraction,
        }
        if picked:
            row.update(translation=picked["translation"], nuance_note=picked.get("nuance_note", ""),
                       _stage_b_verification="auto_resolved_prefilter",
                       _stage_b_prefilter_model=last["model"])
            print(f"  auto-resolved [{path}, {last.get('finish_reason')}, {tokens}]: {source_text[:60]}")
        else:
            # The verdict is already settled (every valid candidate agreed) -- only polishing
            # failed on every path. Sending it to Cowork would waste a human slot on something
            # that was never contested, so keep the first agreeing candidate verbatim, unpolished.
            first = agreeing[0]
            row.update(translation=first["translation"], nuance_note=first.get("nuance_note", ""),
                       _stage_b_verification="auto_resolved_prefilter_fallback_unpolished",
                       _stage_b_prefilter_model=primary["slug"],
                       _stage_b_prefilter_error="; ".join(l.get("error", "") for l in logs)[:300])
            print(f"  auto-resolved UNPOLISHED (every polish path failed): {source_text[:60]}")
        time.sleep(args.sleep)
        return row

    by_source = dict(groups)
    if args.repolish_from:
        resolved = load_jsonl(args.repolish_from)
        todo = [i for i, r in enumerate(resolved)
                if r.get("_stage_b_verification") == "auto_resolved_prefilter_fallback_unpolished"]
        print(f"Re-polishing {len(todo)} unpolished rows from {args.repolish_from}")
        for i in todo:
            src = resolved[i]["source_text"]
            valid = [c for c in by_source.get(src, []) if "_translation_suspect" not in c]
            eligible, has_subtext, category, agreeing, fraction = eligible_for_autoresolve(valid, args.min_agreement)
            if not eligible:
                print(f"  skipped (no longer unanimous in --in): {src[:60]}")
                continue
            resolved[i] = resolve(src, by_source[src], valid, has_subtext, category, agreeing, fraction)
        write_jsonl(args.resolved, resolved)
        print(f"\nDone. Re-polished {len(todo)} rows -> {args.resolved}.")
        print(f"Paths: {dict(paths)}")
        print(f"finish_reason per call: {dict(finish)}")
        return

    resolved, needs_cowork = [], []
    contested_count = 0
    for source_text, cands in groups:
        valid = [c for c in cands if "_translation_suspect" not in c]
        eligible, has_subtext, category, agreeing, fraction = eligible_for_autoresolve(
            valid, args.min_agreement
        )
        if not eligible:
            contested_count += 1
            needs_cowork.extend(cands)  # pass through unchanged, including suspect ones
            continue
        resolved.append(resolve(source_text, cands, valid, has_subtext, category, agreeing, fraction))

    write_jsonl(args.resolved, resolved)
    write_jsonl(args.needs_cowork, needs_cowork)

    print(f"\nDone. {len(groups)} sentences: {len(resolved)} auto-resolved "
          f"({paths['unpolished']} of those unpolished after every polish path failed), "
          f"{contested_count} genuinely contested -> {args.needs_cowork}.")
    print(f"Polish paths: primary {paths['primary']}, primary_retry {paths['primary_retry']}, "
          f"fallback_model {paths['fallback_model']}, unpolished {paths['unpolished']}")
    print(f"finish_reason per call: {dict(finish)}")
    print(f"Cowork workload: {len({r['source_text'] for r in needs_cowork})} sentences "
          f"(down from {len(groups)}).")
    print("Reminder: auto-resolved rows still need to appear in the human calibration sample "
          "(docs/DESIGN.md step 4) -- unanimous doesn't mean correct.")


if __name__ == "__main__":
    main()
