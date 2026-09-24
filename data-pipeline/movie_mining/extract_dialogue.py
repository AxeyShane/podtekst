"""Step 1 of the audio track: pull a clean dialogue stem out of a film.

    python -m movie_mining.extract_dialogue path/to/film.mkv

Writes raw-media/work/<film>/:
  dialogue.wav     16 kHz mono dialogue stem
  background.wav   16 kHz mono music/effects stem (used to score noise later)
  ru.srt / en.srt  embedded text subtitle tracks, when the file has them
  meta.json        which method/stream was used

Method, in order of preference:
  * 5.1/7.1 audio -> the center (FC) channel. Film mixes put dialogue there and
    most music/effects in the other channels, so this is nearly clean for free.
    Background = FL+FR.
  * Stereo/mono  -> Demucs (htdemucs, two-stem vocals). Heavier, needs the
    audio extras (see requirements-audio.txt).

The Russian audio track is picked by language tag (rus/ru) when the file has
several; override with --audio-stream.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
WORK_ROOT = PIPELINE_DIR / "raw-media" / "work"
SR = 16000
TEXT_SUB_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}


def _run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
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


def extract_demucs(src: Path, a_idx: int, out_dir: Path, device: str | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stereo = tmp / "mix.wav"
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", f"0:a:{a_idx}", "-ac", "2", "-ar", "44100",
              str(stereo)])
        cmd = ["python", "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(tmp / "sep")]
        if device:
            cmd += ["-d", device]
        _run(cmd + [str(stereo)])
        stem_dir = tmp / "sep" / "htdemucs" / "mix"
        for stem, name in (("vocals", "dialogue"), ("no_vocals", "background")):
            _run(["ffmpeg", "-v", "error", "-y", "-i", str(stem_dir / f"{stem}.wav"), "-ac", "1", "-ar", str(SR),
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
