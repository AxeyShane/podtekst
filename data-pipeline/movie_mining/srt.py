"""Minimal SRT reader (ffmpeg converts ASS/MOV_TEXT subtitle streams to SRT)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .text_utils import clean_line

_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip().lstrip("﻿")):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            m = _TIME.search(ln)
            if m:
                body = clean_line(" ".join(lines[i + 1:]))
                if body:
                    cues.append(Cue(_secs(*m.groups()[:4]), _secs(*m.groups()[4:]), body))
                break
    cues.sort(key=lambda c: c.start)
    return cues


def load_srt(path: Path | None) -> list[Cue]:
    if not path or not Path(path).exists():
        return []
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251", "latin-1"):     # Russian subs are often cp1251
        try:
            return parse_srt(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return []


def text_in_window(cues: list[Cue], start: float, end: float, min_cover: float = 0.5) -> str:
    """Joins cues that mostly fall inside [start, end] (>= min_cover of the cue)."""
    parts = []
    for c in cues:
        if c.end <= start or c.start >= end:
            continue
        overlap = min(end, c.end) - max(start, c.start)
        if overlap / max(1e-6, c.end - c.start) >= min_cover:
            parts.append(c.text)
    return " ".join(parts)
