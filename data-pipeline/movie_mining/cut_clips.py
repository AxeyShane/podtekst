"""Step 3 of the audio track: clean single-speaker clips aligned to subtitles.

    python -m movie_mining.cut_clips <media root>/work/<film>

Reads dialogue.wav, background.wav, segments.json (+ ru.srt / en.srt if
present, else the Whisper transcript ru.whisper.srt from transcribe.py). Same-speaker
segments less than --merge-gap apart are joined first. For each segment:
  1. cut out any time where 2+ speakers overlap,
  2. keep pieces between --min-dur and --max-dur seconds (longer pieces are
     split),
  3. score speech-to-background ratio (dialogue vs background stem RMS, dB)
     and drop noisy pieces (< --min-sbr) or near-silent ones,
  4. attach the RU/EN subtitle text that falls inside the piece.

Writes clips/*.wav and manifest.jsonl in the work folder. Everything stays
under the media root (see paths.py) -- never commit or share the clips.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .srt import load_srt, text_in_window

EPS = 1e-10


def merge_same_speaker(segs: list[dict], max_gap: float) -> list[dict]:
    """Join consecutive segments of one speaker separated by less than max_gap seconds.
    Diarization splits sentences at short pauses; unmerged, most pieces fall under --min-dur.
    If another speaker talks inside a bridged gap, overlap_regions() cuts that span out again."""
    merged: list[dict] = []
    last: dict[str, dict] = {}
    for s in sorted(segs, key=lambda x: x["start"]):
        prev = last.get(s["speaker"])
        if prev is not None and s["start"] - prev["end"] < max_gap:
            prev["end"] = max(prev["end"], s["end"])
            continue
        s = dict(s)
        merged.append(s)
        last[s["speaker"]] = s
    return merged


def overlap_regions(segs: list[dict]) -> list[tuple[float, float]]:
    """Time spans where 2+ different speakers are active (sweep line)."""
    events = []
    for s in segs:
        events.append((s["start"], 1, s["speaker"]))
        events.append((s["end"], -1, s["speaker"]))
    events.sort(key=lambda e: (e[0], e[1]))       # ends before starts at the same instant
    active: Counter = Counter()
    regions, open_at = [], None
    for t, kind, spk in events:
        before = sum(1 for v in active.values() if v > 0)
        active[spk] += kind
        after = sum(1 for v in active.values() if v > 0)
        if before < 2 <= after:
            open_at = t
        elif before >= 2 > after and open_at is not None:
            if t > open_at:
                regions.append((open_at, t))
            open_at = None
    return regions


def subtract(span: tuple[float, float], holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [span]
    for hs, he in holes:
        nxt = []
        for s, e in pieces:
            if he <= s or hs >= e:
                nxt.append((s, e))
                continue
            if hs > s:
                nxt.append((s, hs))
            if he < e:
                nxt.append((he, e))
        pieces = nxt
    return pieces


def split_long(s: float, e: float, max_dur: float) -> list[tuple[float, float]]:
    n = max(1, int(np.ceil((e - s) / max_dur)))
    step = (e - s) / n
    return [(s + i * step, s + (i + 1) * step) for i in range(n)]


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + EPS)


def cut(work_dir: Path, min_dur: float = 1.0, max_dur: float = 12.0, min_sbr: float = 8.0,
        min_level: float = -45.0, require_subs: bool = True, pad: float = 0.05, merge_gap: float = 0.5) -> dict:
    import soundfile as sf
    work_dir = Path(work_dir)
    dia, sr = sf.read(work_dir / "dialogue.wav", dtype="float32")
    bg_path = work_dir / "background.wav"
    bg = sf.read(bg_path, dtype="float32")[0] if bg_path.exists() else np.zeros_like(dia)
    n = min(len(dia), len(bg))
    dia, bg = dia[:n], bg[:n]
    segs = json.loads((work_dir / "segments.json").read_text(encoding="utf-8"))
    ru, en = load_srt(work_dir / "ru.srt"), load_srt(work_dir / "en.srt")
    ru_source = "human_subs" if ru else None
    if not ru:
        ru = load_srt(work_dir / "ru.whisper.srt")      # machine transcript (transcribe.py)
        ru_source = "whisper" if ru else None
    if require_subs and not ru:
        print(f"{work_dir.name}: no ru.srt or ru.whisper.srt -- keeping clips without text")
        require_subs = False
    meta_path = work_dir / "meta.json"
    method = json.loads(meta_path.read_text(encoding="utf-8")).get("method") if meta_path.exists() else None

    n_raw = len(segs)
    segs = merge_same_speaker(segs, merge_gap)
    holes = overlap_regions(segs)
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    stats: Counter = Counter(segments=n_raw, merged_segments=len(segs), overlap_regions=len(holes))
    rows = []
    for seg in sorted(segs, key=lambda s: s["start"]):
        for ps, pe in subtract((seg["start"], seg["end"]), holes):
            if pe - ps < min_dur:
                stats["drop_short"] += 1
                continue
            for cs, ce in split_long(ps, pe, max_dur):
                a, b = int(max(0, cs - pad) * sr), int(min(n / sr, ce + pad) * sr)
                if b - a < int(min_dur * sr * 0.9):
                    stats["drop_short"] += 1
                    continue
                level = rms_db(dia[a:b])
                sbr = level - rms_db(bg[a:b])
                if level < min_level:
                    stats["drop_silent"] += 1
                    continue
                if sbr < min_sbr:
                    stats["drop_noisy"] += 1
                    continue
                ru_text, en_text = text_in_window(ru, cs, ce), text_in_window(en, cs, ce)
                if require_subs and not ru_text:
                    stats["drop_no_subtitle"] += 1
                    continue
                name = f"{work_dir.name}_{len(rows):05d}_spk{seg['speaker']}.wav"
                sf.write(clips_dir / name, dia[a:b], sr, subtype="PCM_16")
                rows.append({"clip": f"clips/{name}", "film": work_dir.name, "start": round(cs, 3),
                             "end": round(ce, 3), "dur": round(ce - cs, 3), "speaker": seg["speaker"],
                             "level_db": round(level, 1), "sbr_db": round(sbr, 1),
                             "ru_text": ru_text, "ru_text_source": ru_source if ru_text else None,
                             "en_text": en_text, "method": method})
    with open(work_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["clips"] = len(rows)
    stats["clip_minutes"] = round(sum(r["dur"] for r in rows) / 60, 1)
    print(f"{work_dir.name}: {dict(stats)}")
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dirs", nargs="+", type=Path)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--min-sbr", type=float, default=8.0, help="Speech-to-background ratio floor, dB")
    ap.add_argument("--min-level", type=float, default=-45.0, help="Dialogue level floor, dBFS")
    ap.add_argument("--no-require-subs", action="store_true", help="Keep clips with no subtitle text")
    ap.add_argument("--merge-gap", type=float, default=0.5,
                    help="Join same-speaker segments separated by less than this many seconds")
    args = ap.parse_args()
    for wd in args.work_dirs:
        cut(wd, args.min_dur, args.max_dur, args.min_sbr, args.min_level, not args.no_require_subs,
            merge_gap=args.merge_gap)


if __name__ == "__main__":
    main()
