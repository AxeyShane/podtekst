"""English speech recognition on Indian-accented speech: score an ASR model on AI4Bharat's Svarah.

Svarah (ai4bharat/Svarah on Hugging Face, gated: accept the terms on the dataset page, then
`hf auth login`) is 9.6 h of English from 117 Indian speakers, tagged with each speaker's first
language. That lets us score a model on the accents closest to the target user (e.g. Hindi and
Kannada first-language speakers) as well as overall.

    python -m voice_eval.svarah --model openai/whisper-large-v3 --languages hindi,kannada
    python -m voice_eval.svarah --model openai/whisper-small --limit 300

Transcription uses a transformers ASR pipeline (GPU if available). WER is computed after a
simple normalisation: lowercase, digits kept, punctuation dropped, hyphens split, common
contractions expanded. Results go to <out-dir>/svarah_<model>.tsv (one row per utterance) and a
per-language summary is printed.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
from collections import defaultdict

CONTRACTIONS = {"i'm": "i am", "it's": "it is", "don't": "do not", "can't": "can not", "won't": "will not",
                "didn't": "did not", "isn't": "is not", "that's": "that is", "there's": "there is",
                "we're": "we are", "they're": "they are", "you're": "you are", "i've": "i have",
                "i'll": "i will", "let's": "let us", "doesn't": "does not", "wasn't": "was not"}
_PUNCT = re.compile(r"[^\w\s']")


def normalize(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("-", " ")
    text = _PUNCT.sub(" ", text)
    out = []
    for w in text.split():
        w = CONTRACTIONS.get(w, w).strip("'")
        out.extend(w.split())
    return [w for w in out if w]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def find_column(columns: list[str], *keys: str) -> str | None:
    for k in keys:
        for c in columns:
            if k in c.lower():
                return c
    return None


def summarize(rows: list[dict], group_key: str = "language") -> dict[str, tuple[int, int, int]]:
    """{group: (utterances, reference words, errors)} plus an 'ALL' entry."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for g in (r.get(group_key) or "unknown", "ALL"):
            acc[g][0] += 1; acc[g][1] += r["ref_words"]; acc[g][2] += r["errors"]
    return {g: tuple(v) for g, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--languages", default=None,
                    help="Comma-separated first languages to keep (substring match, e.g. hindi,kannada)")
    ap.add_argument("--limit", type=int, default=None, help="Score at most this many utterances")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out-dir", default=os.environ.get("PODTEKST_MEDIA_ROOT", "."))
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from transformers import pipeline

    ds = load_dataset("ai4bharat/Svarah", split="test")
    cols = ds.column_names
    audio_col = find_column(cols, "audio")
    text_col = find_column(cols, "text", "transcript", "sentence")
    lang_col = find_column(cols, "primary_language", "language", "lang")
    print(f"{len(ds)} utterances; columns: {cols}")
    print(f"using audio={audio_col!r} text={text_col!r} language={lang_col!r}")
    ds = ds.cast_column(audio_col, Audio(decode=False))
    if args.languages and lang_col:
        wanted = [w.strip().lower() for w in args.languages.split(",") if w.strip()]
        ds = ds.filter(lambda r: any(w in str(r[lang_col]).lower() for w in wanted))
        print(f"{len(ds)} utterances from first languages {wanted}")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device,
                   torch_dtype=torch.float16 if device == 0 else torch.float32)
    kw = {"generate_kwargs": {"language": "en", "task": "transcribe"}} if "whisper" in args.model.lower() else {}

    rows, batch = [], []
    def flush():
        inputs = [{"raw": a, "sampling_rate": sr} for _, a, sr in batch]
        outs = asr(inputs, batch_size=args.batch, **kw)
        for (r, _, _), o in zip(batch, outs):
            ref, hyp = normalize(r[text_col]), normalize(o["text"])
            rows.append({"language": str(r[lang_col]) if lang_col else "", "ref_words": len(ref),
                         "errors": edit_distance(ref, hyp), "reference": r[text_col], "hypothesis": o["text"].strip()})
        batch.clear()
    for k, r in enumerate(ds):
        a = r[audio_col]
        data, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        batch.append((r, data, sr))
        if len(batch) >= args.batch * 4:
            flush(); print(f"  {len(rows)}/{len(ds)}")
    if batch:
        flush()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"svarah_{args.model.replace('/', '_')}.tsv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["language", "ref_words", "errors", "reference", "hypothesis"], delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")
    for g, (n, words, errs) in sorted(summarize(rows).items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
        print(f"{g:<14} {n:>5} utts  WER {errs / words:.1%}" if words else f"{g:<14} {n:>5} utts")


if __name__ == "__main__":
    main()
