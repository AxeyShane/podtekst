"""Run the whole audio track on one film or a folder of films.

    python -m movie_mining.run_film                             # every video in <media root>/films
    python -m movie_mining.run_film D:\\films\\                   # or any folder
    python -m movie_mining.run_film film.mkv --ru-srt film.ru.srt --en-srt film.en.srt

extract_dialogue -> diarize (Nemotron 3) -> cut_clips. Steps whose outputs
already exist are skipped, so re-running after a crash resumes. The diarization
model is loaded once for the whole batch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import cut_clips, diarize, extract_dialogue
from .paths import FILMS_DIR

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".webm"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, nargs="?", default=FILMS_DIR,
                    help="A video file or a folder of them (default: <media root>/films)")
    ap.add_argument("--ru-srt", type=Path, default=None, help="External RU .srt (single-film runs)")
    ap.add_argument("--en-srt", type=Path, default=None, help="External EN .srt (single-film runs)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--demucs", action="store_true", help="Force Demucs instead of the center channel")
    ap.add_argument("--backend", choices=["auto", "nemo", "transformers"], default="auto",
                    help="Diarization backend (auto: NeMo if installed, else transformers)")
    ap.add_argument("--chunk-minutes", type=float, default=0)
    ap.add_argument("--min-sbr", type=float, default=8.0)
    ap.add_argument("--no-require-subs", action="store_true")
    args = ap.parse_args()

    films = sorted(p for p in args.path.iterdir() if p.suffix.lower() in VIDEO_EXT) if args.path.is_dir() \
        else [args.path]
    if not films:
        raise SystemExit(f"No video files in {args.path}")
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
    for wd in work_dirs:
        cut_clips.cut(wd, min_sbr=args.min_sbr, require_subs=not args.no_require_subs)


if __name__ == "__main__":
    main()
