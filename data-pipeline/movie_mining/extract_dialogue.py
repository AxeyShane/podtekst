"""Step 1 of the audio track: pull a clean dialogue stem out of a film.

    python -m movie_mining.extract_dialogue path/to/film.mkv

Writes <media root>/work/<film>/ (see paths.py):
  dialogue.wav     16 kHz mono dialogue stem
  background.wav   16 kHz mono music/effects stem (used to score noise later)
  ru.srt / en.srt  embedded text subtitle tracks, when the file has them
  meta.json        which method/stream was used

Method, in order of preference:
  * 5.1/7.1 audio -> the center (FC) channel. Film mixes put dialogue there and
    most music/effects in the other channels, so this is nearly clean for free.
    Background = FL+FR.
  * Stereo/mono  -> Demucs (htdemucs, two-stem vocals). Heavier, needs the
    audio extras (see requirements-audio.txt). Runs on 20-minute chunks whose
    stems are concatenated, so multi-hour files fit in RAM.

The Russian audio track is picked by language tag (rus/ru) when the file has
several; override with --audio-stream.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .paths import WORK_ROOT
SR = 16000
TEXT_SUB_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}


def _run(cmd: list[str]) -> str:
    # Explicit UTF-8: Windows defaults to cp1252, and demucs/ffmpeg progress output isn't always decodable in it.
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr[-2000:]}")
    return res.stdout


def probe(path: Path) -> list[dict]:
    out = _run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
    return json.loads(out).get("streams", [])


def _lang(stream: dict) -> str:
    return (stream.get("tags") or {}).get("language", "").lower()


def pick_audio(streams: list[dict], index: int | None) -> tuple[int, dict]:
    """Returns (position among audio streams, stream dict)."""
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise SystemExit("No audio stream found")
    if index is not None:
        return index, audio[index]
    for i, s in enumerate(audio):
        if _lang(s) in ("rus", "ru"):
            return i, s
    return 0, audio[0]


def _has_center(stream: dict) -> bool:
    layout = (stream.get("channel_layout") or "").lower()
    return int(stream.get("channels", 0)) >= 6 and ("5.1" in layout or "7.1" in layout or "fc" in layout
                                                    or layout == "")


def extract_center(src: Path, a_idx: int, out_dir: Path) -> None:
    common = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", f"0:a:{a_idx}", "-ar", str(SR)]
    _run(common + ["-af", "pan=mono|c0=FC", str(out_dir / "dialogue.wav")])
    _run(common + ["-af", "pan=mono|c0=0.5*FL+0.5*FR", str(out_dir / "background.wav")])


def duration(path: Path) -> float:
    return float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                       "default=nw=1:nk=1", str(path)]).strip())


def chunk_spans(total: float, chunk_s: float) -> list[tuple[float, float]]:
    """[(start, length)] covering 0..total in chunk_s pieces; a short tail (under a minute, or a
    quarter chunk for tiny chunks) joins the last piece instead of becoming its own Demucs run."""
    spans, start = [], 0.0
    min_tail = min(60.0, chunk_s / 4)
    while start < total:
        length = min(chunk_s, total - start)
        if spans and length < min_tail:
            s, l = spans[-1]
            spans[-1] = (s, l + length)
            break
        spans.append((start, length))
        start += length
    return spans


def extract_demucs(src: Path, a_idx: int, out_dir: Path, device: str | None, chunk_minutes: float = 20) -> None:
    """Demucs on chunk_minutes pieces, stems joined afterwards: Demucs holds the whole input (and its
    outputs) in RAM, so a multi-hour file would exhaust memory in one go. Only one chunk's 44.1 kHz
    stereo audio sits in the temp dir at a time.
    ponytail: hard cuts, no overlap -- a word split at a chunk edge can lose a few ms of separation
    quality; add overlap + crossfade if clips at the ~20-min marks turn out damaged."""
    spans = chunk_spans(duration(src), chunk_minutes * 60)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts: dict[str, list[Path]] = {"dialogue": [], "background": []}
        for i, (start, length) in enumerate(spans):
            stereo = tmp / f"mix{i}.wav"
            _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
                  "-map", f"0:a:{a_idx}", "-ac", "2", "-ar", "44100", str(stereo)])
            cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(tmp / "sep")]
            if device:
                cmd += ["-d", device]
            _run(cmd + [str(stereo)])
            stem_dir = tmp / "sep" / "htdemucs" / stereo.stem
            for stem, name in (("vocals", "dialogue"), ("no_vocals", "background")):
                part = tmp / f"{name}{i}.wav"
                _run(["ffmpeg", "-v", "error", "-y", "-i", str(stem_dir / f"{stem}.wav"), "-ac", "1", "-ar", str(SR),
                      str(part)])
                parts[name].append(part)
            stereo.unlink(missing_ok=True)
            shutil.rmtree(stem_dir, ignore_errors=True)
            if len(spans) > 1:
                print(f"  demucs chunk {i + 1}/{len(spans)}")
        for name, files in parts.items():
            listing = tmp / f"{name}.txt"
            listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in files), encoding="utf-8")
            _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                  str(out_dir / f"{name}.wav")])


def extract_subs(src: Path, streams: list[dict], out_dir: Path) -> dict:
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    found = {}
    for pos, s in enumerate(subs):
        lang = _lang(s)
        key = "ru" if lang in ("rus", "ru") else "en" if lang in ("eng", "en") else None
        if not key or key in found:
            continue
        if s.get("codec_name") not in TEXT_SUB_CODECS:
            found[f"{key}_skipped"] = f"{s.get('codec_name')} is image-based (needs OCR)"
            continue
        dest = out_dir / f"{key}.srt"
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", f"0:s:{pos}", "-c:s", "srt", str(dest)])
        found[key] = dest.name
    return found


def extract(src: Path, out_dir: Path | None = None, audio_stream: int | None = None,
            force_demucs: bool = False, device: str | None = None, ru_srt: Path | None = None,
            en_srt: Path | None = None) -> Path:
    src = Path(src)
    out_dir = Path(out_dir) if out_dir else WORK_ROOT / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    streams = probe(src)
    a_idx, a = pick_audio(streams, audio_stream)
    method = "demucs" if force_demucs or not _has_center(a) else "center_channel"
    if method == "center_channel":
        extract_center(src, a_idx, out_dir)
    else:
        extract_demucs(src, a_idx, out_dir, device)
    subs = extract_subs(src, streams, out_dir)
    for key, given in (("ru", ru_srt), ("en", en_srt)):       # external .srt files win
        if given:
            shutil.copyfile(given, out_dir / f"{key}.srt")
            subs[key] = f"{key}.srt"
    meta = {"source": src.name, "method": method, "audio_stream": a_idx, "audio_lang": _lang(a),
            "channels": a.get("channels"), "channel_layout": a.get("channel_layout"), "subs": subs}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{src.name}: {method} (audio stream {a_idx}, {a.get('channels')} ch) -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--audio-stream", type=int, default=None, help="Audio stream position (0-based)")
    ap.add_argument("--demucs", action="store_true", help="Force Demucs even when 5.1 is available")
    ap.add_argument("--device", default=None, help="Demucs device (cuda/cpu)")
    ap.add_argument("--ru-srt", type=Path, default=None, help="External Russian .srt")
    ap.add_argument("--en-srt", type=Path, default=None, help="External English .srt")
    args = ap.parse_args()
    extract(args.film, args.out_dir, args.audio_stream, args.demucs, args.device, args.ru_srt, args.en_srt)


if __name__ == "__main__":
    main()
