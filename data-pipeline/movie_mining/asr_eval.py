"""Score GigaAM against a video's human Russian subtitles (word error rate).

    python -m movie_mining.asr_eval <video>.wav <video>.ru.srt [--minutes 20] [--start-minute 5]

The audio must be 16 kHz mono WAV (yt-dlp ... --audio-format wav --postprocessor-args
"ffmpeg:-ar 16000 -ac 1"). Subtitle cues are cleaned with align_subs.read_srt, then
consecutive cues less than MAX_GAP apart are merged into segments of at most MAX_SEG
seconds (GigaAM's transcribe() limit is 25 s). Each segment is cut, transcribed and
compared with its subtitle text.

Two WERs are reported:
  * raw: subtitle text vs GigaAM, after lowercasing, ё->е and dropping punctuation;
  * no-fillers: the same with hesitation words (э, ээ, м, мм, угу, ...) removed from both.
Creator subtitles are usually tidied (fillers, false starts and repeats dropped), so both
numbers are upper bounds on GigaAM's real error rate. A short hand-corrected verbatim
sample is what gives the true number. Segments where the subtitle skips lines, or where
two people talk at once, show up as very high per-segment WER in the TSV; check those
before trusting the average.

Writes asr_eval.tsv (segment, start, end, wer, wer_no_fillers, subtitle, gigaam) next
to the audio. Keep it in the media root: it holds verbatim dialogue.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile

from .align_subs import read_srt

MAX_GAP = 0.6     # seconds between cues that still counts as one stretch of speech
MAX_SEG = 15.0    # seconds per segment, well inside GigaAM's 25 s limit
MIN_SEG = 1.0     # shorter segments are skipped (too little context, often interjections)
FILLERS = {"э", "ээ", "эээ", "э-э", "м", "мм", "ммм", "м-м", "хм", "угу", "ага", "а-а", "аа", "ну-у"}
_NON_WORD = re.compile(r"[^\w\s-]")


def words(text: str, drop_fillers: bool = False) -> list[str]:
    text = _NON_WORD.sub(" ", text.lower().replace("ё", "е"))
    out = [w.strip("-") for w in text.split()]
    out = [w for w in out if w]
    return [w for w in out if w not in FILLERS] if drop_fillers else out


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def segments(cues: list[dict], start: float = 0.0, end: float = float("inf")) -> list[dict]:
    """Merge consecutive cues into speech segments of at most MAX_SEG seconds."""
    segs, cur = [], None
    for c in cues:
        if c["end"] <= start or c["start"] >= end:
            continue
        if cur and c["start"] - cur["end"] <= MAX_GAP and c["end"] - cur["start"] <= MAX_SEG:
            cur["end"] = c["end"]; cur["text"] += " " + c["text"]
        else:
            if cur: segs.append(cur)
            cur = {"start": c["start"], "end": c["end"], "text": c["text"]}
    if cur: segs.append(cur)
    return [s for s in segs if s["end"] - s["start"] >= MIN_SEG]


def evaluate(segs: list[dict], audio, sr: int, asr, tmp_dir: str) -> list[dict]:
    import soundfile as sf
    rows = []
    for k, s in enumerate(segs):
        piece = audio[int(s["start"] * sr):int(s["end"] * sr)]
        path = os.path.join(tmp_dir, f"seg{k}.wav")
        sf.write(path, piece, sr)
        hyp = asr.transcribe(path)
        ref_w, hyp_w = words(s["text"]), words(hyp)
        ref_f, hyp_f = words(s["text"], True), words(hyp, True)
        rows.append({"segment": k, "start": round(s["start"], 2), "end": round(s["end"], 2),
                     "ref_words": len(ref_w), "errors": edit_distance(ref_w, hyp_w),
                     "ref_words_nf": len(ref_f), "errors_nf": edit_distance(ref_f, hyp_f),
                     "subtitle": s["text"], "gigaam": hyp})
    return rows


def summarize(rows: list[dict]) -> dict:
    n = sum(r["ref_words"] for r in rows); e = sum(r["errors"] for r in rows)
    nf = sum(r["ref_words_nf"] for r in rows); ef = sum(r["errors_nf"] for r in rows)
    return {"segments": len(rows), "ref_words": n, "wer": e / n if n else 0.0,
            "wer_no_fillers": ef / nf if nf else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav"); ap.add_argument("ru_srt")
    ap.add_argument("--start-minute", type=float, default=0.0)
    ap.add_argument("--minutes", type=float, default=None, help="Only score this many minutes")
    ap.add_argument("--gigaam-model", default=None)
    args = ap.parse_args()
    import soundfile as sf
    from .transcribe import GigaAM, GIGAAM_DEFAULT
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1 or sr != 16000:
        raise SystemExit("Expected 16 kHz mono WAV.")
    start = args.start_minute * 60
    end = start + args.minutes * 60 if args.minutes else float("inf")
    segs = segments(read_srt(args.ru_srt), start, end)
    print(f"{len(segs)} segments to score")
    asr = GigaAM(args.gigaam_model or GIGAAM_DEFAULT)
    with tempfile.TemporaryDirectory() as tmp:
        rows = evaluate(segs, audio, sr, asr, tmp)
    for r in rows:
        r["wer"] = round(r["errors"] / r["ref_words"], 3) if r["ref_words"] else ""
        r["wer_no_fillers"] = round(r["errors_nf"] / r["ref_words_nf"], 3) if r["ref_words_nf"] else ""
    out = os.path.join(os.path.dirname(os.path.abspath(args.wav)), "asr_eval.tsv")
    cols = ["segment", "start", "end", "wer", "wer_no_fillers", "subtitle", "gigaam"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    s = summarize(rows)
    print(f"Wrote {out}")
    print(f"{s['segments']} segments, {s['ref_words']} subtitle words: "
          f"WER {s['wer']:.1%}, without fillers {s['wer_no_fillers']:.1%}")


if __name__ == "__main__":
    main()
