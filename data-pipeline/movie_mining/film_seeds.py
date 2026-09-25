"""Stage A seeds from film dialogue clips: machine-transcribed lines both ASR engines agree on.

    python -m movie_mining.film_seeds --out seeds_batch6_film.txt --n 500 --per-film 30 \
        --avoid "seeds_batch*.txt,stage_a_batch*_next_seeds.txt,seeds_example.txt"

Takes ru_text from every <media root>/work/*/manifest.jsonl row with asr_agree == true (GigaAM and
Whisper within CER 0.10), 4-20 words, Russian only (Cyrillic, no Latin), passing the subtitle
miner's fluency / mojibake / interjection checks, and deduped (normalised) against itself and every
--avoid file. Samples round-robin across films, at most --per-film each, so no talky film dominates.

Writes --out (one seed per line) and a sidecar <out>.meta.jsonl with film, clip, start/end and the
ASR agreement for each seed. Both hold verbatim film dialogue: keep them out of git.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

from .cues import ru_fluency_issue
from .paths import WORK_ROOT
from .text_utils import is_interjection_only, normalize_key, word_count

_LATIN = re.compile(r"[A-Za-z]")
_CYR = re.compile(r"[А-Яа-яЁё]")


def usable(text: str, min_words: int = 4, max_words: int = 20) -> bool:
    text = (text or "").strip()
    return (min_words <= word_count(text) <= max_words and bool(_CYR.search(text)) and not _LATIN.search(text)
            and not is_interjection_only(text) and ru_fluency_issue(text) is None)


def load_avoid(patterns: str) -> set[str]:
    keys = set()
    for pat in (p.strip() for p in patterns.split(",") if p.strip()):
        for path in glob.glob(pat):
            with open(path, encoding="utf-8") as f:
                keys.update(normalize_key(l) for l in f if l.strip())
    return keys


def collect(work_root: Path, avoid: set[str], min_words: int = 4, max_words: int = 20) -> dict[str, list[dict]]:
    """{film: [candidate rows]} after filters and dedupe (first occurrence wins)."""
    seen = set(avoid)
    by_film: dict[str, list[dict]] = {}
    for manifest in sorted(Path(work_root).glob("*/manifest.jsonl")):
        film = manifest.parent.name
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            text = (r.get("ru_text") or "").strip()
            if r.get("asr_agree") is not True or not usable(text, min_words, max_words):
                continue
            key = normalize_key(text)
            if key in seen:
                continue
            seen.add(key)
            by_film.setdefault(film, []).append({"seed": text, "film": film, "clip": r["clip"],
                                                 "start": r.get("start"), "end": r.get("end"),
                                                 "asr_cer": r.get("asr_cer"), "speaker": r.get("speaker")})
    return by_film


def sample(by_film: dict[str, list[dict]], n: int, per_film: int, rng: random.Random) -> list[dict]:
    """Round-robin across films (shuffled within each), at most per_film each, until n."""
    pools = {f: rng.sample(rows, len(rows))[:per_film] for f, rows in by_film.items()}
    out = []
    while len(out) < n and any(pools.values()):
        for film in sorted(pools):
            if pools[film] and len(out) < n:
                out.append(pools[film].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--per-film", type=int, default=30)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=20)
    ap.add_argument("--avoid", default="", help="Comma-separated globs of earlier seed files to dedupe against")
    ap.add_argument("--work-root", type=Path, default=WORK_ROOT)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()

    by_film = collect(args.work_root, load_avoid(args.avoid), args.min_words, args.max_words)
    picked = sample(by_film, args.n, args.per_film, random.Random(args.seed))
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(r["seed"] + "\n" for r in picked)
    meta = Path(args.out).with_suffix(".meta.jsonl")
    with open(meta, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({**r, "source": "film"}, ensure_ascii=False) + "\n" for r in picked)
    per = {film: sum(r["film"] == film for r in picked) for film in by_film}
    print(f"{sum(map(len, by_film.values()))} eligible lines across {len(by_film)} films; "
          f"wrote {len(picked)} seeds -> {args.out} (+ {meta.name})")
    print("per film: " + ", ".join(f"{k} {v}/{len(by_film[k])}" for k, v in sorted(per.items())))


if __name__ == "__main__":
    main()
