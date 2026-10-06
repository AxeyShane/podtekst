"""Score an ASR model on your own recordings (spontaneous speech, verbatim transcripts).

    python -m voice_eval.own_voice <folder> [--model openai/whisper-large-v3]

<folder> holds clips with a transcript beside each one: talk01.m4a + talk01.txt, and so on.
Any audio ffmpeg can read works (phone .m4a, .wav, .mp3, .ogg). Each .txt is what was
actually said, written as spoken: keep "um", "uh", "hmm", restarts and repeated words, and
write Hindi or other non-English words in Latin script (matlab, haan, yaar, accha).

Three WERs are reported:
  * raw: everything counts, fillers included;
  * no-fillers: hesitation sounds (um, uh, hmm, aah, ...) removed from both sides;
  * English-only: also drops the non-English words listed in --other-words (default: common
    Hindi fillers and discourse words), to separate accent errors from code-switching errors.

Per-clip results go to <folder>/own_voice_<model>.tsv. Recordings and transcripts stay on the
local media drive; they are never committed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess

import numpy as np

from .svarah import edit_distance, normalize

AUDIO_EXT = (".wav", ".m4a", ".mp3", ".ogg", ".opus", ".flac", ".aac", ".webm")
HESITATIONS = {"um", "umm", "ummm", "uh", "uhh", "uhm", "hmm", "hm", "mm", "mmm", "ah", "aah", "ahh", "er", "erm", "eh"}
HINDI_WORDS = {"matlab", "haan", "han", "nahi", "nahin", "yaar", "accha", "acha", "achha", "arre", "arey", "bas",
               "na", "toh", "to", "theek", "thik", "hai", "kya", "ki", "ke", "ko", "bhai", "chalo", "haina", "ji"}


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Decode any audio file to mono float32 at sr via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def find_pairs(folder: str) -> list[tuple[str, str]]:
    pairs = []
    for txt in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        stem = txt[:-4]
        audio = next((stem + e for e in AUDIO_EXT if os.path.exists(stem + e)), None)
        if audio:
            pairs.append((audio, txt))
    return pairs


def score(ref_text: str, hyp_text: str, other_words: set[str]) -> dict:
    ref, hyp = normalize(ref_text), normalize(hyp_text)
    ref_nf = [w for w in ref if w not in HESITATIONS]; hyp_nf = [w for w in hyp if w not in HESITATIONS]
    ref_en = [w for w in ref_nf if w not in other_words]; hyp_en = [w for w in hyp_nf if w not in other_words]
    return {"words": len(ref), "errors": edit_distance(ref, hyp),
            "words_nf": len(ref_nf), "errors_nf": edit_distance(ref_nf, hyp_nf),
            "words_en": len(ref_en), "errors_en": edit_distance(ref_en, hyp_en)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--other-words", default=",".join(sorted(HINDI_WORDS)),
                    help="Comma-separated non-English words to drop for the English-only WER")
    args = ap.parse_args()
    other = {w.strip().lower() for w in args.other_words.split(",") if w.strip()}

    import torch
    from transformers import pipeline
    pairs = find_pairs(args.folder)
    if not pairs:
        raise SystemExit(f"No audio+.txt pairs in {args.folder}")
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device,
                   torch_dtype=torch.float16 if device == 0 else torch.float32)
    kw = {"generate_kwargs": {"language": "en", "task": "transcribe"}} if "whisper" in args.model.lower() else {}

    rows = []
    for audio, txt in pairs:
        with open(txt, encoding="utf-8") as f:
            ref = f.read().strip()
        hyp = asr({"raw": load_audio(audio), "sampling_rate": 16000}, return_timestamps=True, **kw)["text"].strip()
        rows.append({"clip": os.path.basename(audio), **score(ref, hyp, other), "reference": ref, "hypothesis": hyp})
        print(f"  {rows[-1]['clip']}: WER {rows[-1]['errors'] / max(1, rows[-1]['words']):.1%}")

    out = os.path.join(args.folder, f"own_voice_{args.model.replace('/', '_')}.tsv")
    cols = ["clip", "words", "errors", "words_nf", "errors_nf", "words_en", "errors_en", "reference", "hypothesis"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(rows)
    tot = {k: sum(r[k] for r in rows) for k in cols[1:7]}
    print(f"Wrote {out}")
    print(f"{len(rows)} clips, {tot['words']} words: WER {tot['errors'] / tot['words']:.1%}, "
          f"no fillers {tot['errors_nf'] / max(1, tot['words_nf']):.1%}, "
          f"English only {tot['errors_en'] / max(1, tot['words_en']):.1%}")


if __name__ == "__main__":
    main()
