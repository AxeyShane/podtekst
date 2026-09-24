"""Run the whole audio track on one film or a folder of films.

    python -m movie_mining.run_film                             # every video in <media root>/films
    python -m movie_mining.run_film D:\\films\\                   # or any folder
    python -m movie_mining.run_film film.mkv --ru-srt film.ru.srt --en-srt film.en.srt

extract_dialogue -> diarize (Nemotron 3) -> cut_clips -> transcribe clips with
GigaAM v3 when there's no Russian subtitle (or --asr whisper: whisper.cpp on
the whole stem before cutting; --asr none to skip). Steps whose outputs
already exist are skipped, so re-running after a crash resumes. The diarization
model is loaded once for the whole batch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import cut_clips, diarize, extract_dialogue, transcribe
from .paths import FILMS_DIR

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".webm"}
# Audio-only files work too (stereo -> Demucs; no subtitles -> clips without text).
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".opus", ".ogg", ".flac", ".wav", ".mka"}
MEDIA_EXT = VIDEO_EXT | AUDIO_EXT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, nargs="?", default=FILMS_DIR,
                    help="A video/audio file or a folder of them (default: <media root>/films)")
    ap.add_argument("--ru-srt", type=Path, default=None, help="External RU .srt (single-film runs)")
    ap.add_argument("--en-srt", type=Path, default=None, help="External EN .srt (single-film runs)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--demucs", action="store_true", help="Force Demucs instead of the center channel")
    ap.add_argument("--backend", choices=["auto", "nemo", "transformers"], default="auto",
                    help="Diarization backend (auto: NeMo if installed, else transformers)")
    ap.add_argument("--chunk-minutes", type=float, default=0)
    ap.add_argument("--min-sbr", type=float, default=8.0)
    ap.add_argument("--merge-gap", type=float, default=0.5, help="Join same-speaker segments closer than this (s)")
    ap.add_argument("--asr", choices=["gigaam", "whisper", "none"], default="gigaam",
                    help="Transcriber for films without a Russian subtitle")
    ap.add_argument("--whisper-model", default="large-v3", help="large-v3, large-v3-turbo, ... or a .bin path")
    ap.add_argument("--no-require-subs", action="store_true")
    args = ap.parse_args()

    films = sorted(p for p in args.path.iterdir() if p.suffix.lower() in MEDIA_EXT) if args.path.is_dir() \
        else [args.path]
    if not films:
        raise SystemExit(f"No video or audio files in {args.path}")
    work_dirs = []
    for film in films:
        wd = extract_dialogue.WORK_ROOT / film.stem
        # Sidecar subtitles next to the film (film.ru.srt / film.en.srt) are picked up automatically.
        ru = args.ru_srt or next((p for p in (film.with_suffix(".ru.srt"), film.with_suffix(".rus.srt")) if p.exists()), None)
        en = args.en_srt or next((p for p in (film.with_suffix(".en.srt"), film.with_suffix(".eng.srt")) if p.exists()), None)
        if not (wd / "dialogue.wav").exists():
            extract_dialogue.extract(film, wd, force_demucs=args.demucs, device=args.device, ru_srt=ru, en_srt=en)
        work_dirs.append(wd)

    todo = [wd for wd in work_dirs if not (wd / "segments.json").exists()]
    if todo:
        diarizer = diarize.make_diarizer(args.backend, args.device, args.chunk_minutes)
        for wd in todo:
            segs = diarizer.diarize(wd / "dialogue.wav")
            (wd / "segments.json").write_text(__import__("json").dumps(segs, indent=1), encoding="utf-8")
            print(f"{wd.name}: {len(segs)} segments")
    needs_asr = [wd for wd in work_dirs if not (wd / "ru.srt").exists()]
    if args.asr == "whisper":
        for wd in needs_asr:
            transcribe.transcribe(wd, args.whisper_model)
    for wd in work_dirs:
        cut_clips.cut(wd, min_sbr=args.min_sbr, require_subs=not args.no_require_subs, merge_gap=args.merge_gap)
    if args.asr == "gigaam" and needs_asr:
        asr = transcribe.GigaAM()
        for wd in needs_asr:
            transcribe.transcribe_clips(wd, asr)


if __name__ == "__main__":
    main()
