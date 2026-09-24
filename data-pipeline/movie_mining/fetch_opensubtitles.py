"""Download the OPUS OpenSubtitles RU-EN sentence alignments (moses format).

    python -m movie_mining.fetch_opensubtitles            # from data-pipeline/

Writes to <media root>/opensubtitles/ (see paths.py; gitignored by default). Resumable: re-running continues
a partial download. The zip holds three line-aligned files: .en, .ru and .ids
(the .ids file identifies which film each line pair came from).

Data: OPUS OpenSubtitles (Lison & Tiedemann, 2016) -- research-use corpus built
from opensubtitles.org. Mined lines stay local and are never committed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

DEFAULT_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ru.txt.zip"
from .paths import OPENSUBS_ZIP as DEFAULT_OUT


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:            # server says we already have it all
            part.rename(dest)
            return dest
        r.raise_for_status()
        resumed = r.status_code == 206
        total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
        mode = "ab" if resumed else "wb"
        done = have if resumed else 0
        with open(part, mode) as f:
            for block in r.iter_content(chunk):
                f.write(block)
                done += len(block)
                if total:
                    sys.stdout.write(f"\r{done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                    sys.stdout.flush()
    print()
    part.rename(dest)
    print(f"Saved {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="OPUS moses zip URL (swap the version to try a newer release)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    download(args.url, args.out)


if __name__ == "__main__":
    main()
