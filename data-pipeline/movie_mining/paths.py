"""Where movie-mining data lives.

Big files (the OpenSubtitles zip, films, extracted audio, clips) go under
MEDIA_ROOT. It defaults to data-pipeline/raw-media/ (gitignored); set the
PODTEKST_MEDIA_ROOT environment variable to put it on another drive, e.g.
D:\\podtekst-media. setup_windows.ps1 sets this for you when C: is short on space.
Model caches follow the standard HF_HOME / TORCH_HOME variables.
"""
from __future__ import annotations

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
MEDIA_ROOT = Path(os.environ.get("PODTEKST_MEDIA_ROOT") or PIPELINE_DIR / "raw-media")
OPENSUBS_ZIP = MEDIA_ROOT / "opensubtitles" / "en-ru.txt.zip"
FILMS_DIR = MEDIA_ROOT / "films"
WORK_ROOT = MEDIA_ROOT / "work"
