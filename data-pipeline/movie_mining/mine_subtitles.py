"""Mine OpenSubtitles RU-EN pairs for real, human-made nuance handling.

Idea (docs: "Additional data source", use #1): a subtitler translating
"Какого чёрта ты тут делаешь?" as "What the hell are you doing here?" is doing
something a literal MT wouldn't. Where the human line and a literal MT of the
same source diverge on the surface while still meaning the same thing, the
pair is a *candidate* for idiom / register / sarcasm handling -- found in real
dialogue, not generated.

Pipeline (all local, GPU recommended):
  0. Optional (--origin ru): keep only films whose original language is Russian
     (IMDb id -> Wikidata P364, see origin.py).
  1. Stream the OPUS zip, clean subtitle markup, cheap filters, a Russian
     fluency check (pymorphy unknown words, doubled capitals), dedupe,
     reservoir-sample a pool spread across the whole corpus.
  2. LaBSE cosine(ru, en) >= --min-align  -> keeps only true translations
     (OpenSubtitles alignments are noisy).
  3. Literal MT with opus-mt (ru->en and/or en->ru).
  4. Divergence: chrF(literal, human) <= --max-chrf (whole-line rewrite), or a
     replaced span of >= --min-span words inside an otherwise literal line
     (local idiom-sized swap).
  5. Rank by alignment x divergence (small boost for idiom-lexicon hits) and
     split into two buckets: "address" (the Russian line has ты/вы -- English
     "you" makes these diverge trivially, so they'd flood the ranking) capped at
     --address-share of the selection, and "general" for everything else.
     Cap per film, write outputs.

Outputs go to <media root>/mining/ (see paths.py) -- verbatim subtitle dialogue,
kept off the repo:
  subs_candidates_<name>.jsonl   full records + scores + bucket
  subs_stats_<name>.json         filter/reject counts
  seeds_subs_<name>.txt          top source lines, ready for
                                 stage_a_generate.py / run_batch.py

Candidates are NOT labels. They still go through the normal Stage A -> checks ->
prefilter -> Cowork -> calibration cycle. The human subtitle is kept in the
candidates file as reference context for verification.

    python -m movie_mining.mine_subtitles --name subs1 --max-lines 3000000
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from .cues import address_register, idiom_hits, ru_fluency_issue
from .text_utils import clean_line, film_id_from_ids_line, is_multi_speaker, normalize_key, pair_passes

from .paths import FILM_LANG_CACHE, MINING_DIR, OPENSUBS_ZIP as DEFAULT_ZIP

LABSE = "sentence-transformers/LaBSE"
MT_MODELS = {"ru-en": "Helsinki-NLP/opus-mt-ru-en", "en-ru": "Helsinki-NLP/opus-mt-en-ru"}


# ---------------------------------------------------------------- reading
def _member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    for name in zf.namelist():
        if name.endswith(suffix):
            return name
    return None


def iter_pairs(zip_path: Path, max_lines: int | None = None) -> Iterator[tuple[str, str, str | None]]:
    """Yields (ru_raw, en_raw, film_id) from an OPUS moses zip, line-aligned."""
    with zipfile.ZipFile(zip_path) as zf:
        ru_name, en_name, ids_name = _member(zf, ".ru"), _member(zf, ".en"), _member(zf, ".ids")
        if not (ru_name and en_name):
            raise SystemExit(f"{zip_path} has no .ru/.en members: {zf.namelist()}")
        opened = [io.TextIOWrapper(zf.open(n), encoding="utf-8", errors="replace") for n in (ru_name, en_name)]
        ids = io.TextIOWrapper(zf.open(ids_name), encoding="utf-8", errors="replace") if ids_name else None
        try:
            for i, (ru, en) in enumerate(zip(*opened)):
                if max_lines is not None and i >= max_lines:
                    break
                film = film_id_from_ids_line(ids.readline()) if ids else None
                yield ru.rstrip("\n"), en.rstrip("\n"), film
        finally:
            for f in opened + ([ids] if ids else []):
                f.close()


def collect_pool(pairs, pool_size: int, rng: random.Random, min_words: int, max_words: int,
                 films: set[str] | None = None):
    """Filter + dedupe, then reservoir-sample so the pool spans the whole corpus
    instead of only the first films in the file. films: if given, only these film keys."""
    stats: Counter = Counter()
    seen: set[str] = set()
    pool: list[dict] = []
    kept = 0
    for ru_raw, en_raw, film in pairs:
        stats["read"] += 1
        if films is not None and film not in films:
            stats["reject_origin"] += 1
            continue
        if is_multi_speaker(ru_raw) or is_multi_speaker(en_raw):
            stats["reject_multi_speaker"] += 1
            continue
        ru, en = clean_line(ru_raw), clean_line(en_raw)
        ok, why = pair_passes(ru, en, min_words=min_words, max_words=max_words)
        if not ok:
            stats[f"reject_{why}"] += 1
            continue
        key = normalize_key(ru)
        if key in seen:
            stats["reject_duplicate"] += 1
            continue
        seen.add(key)
        issue = ru_fluency_issue(ru)            # after dedupe: each distinct line is checked once
        if issue:
            stats[f"reject_fluency_{issue}"] += 1
            continue
        rec = {"ru": ru, "en": en, "film": film}
        kept += 1
        if len(pool) < pool_size:
            pool.append(rec)
        else:
            j = rng.randrange(kept)
            if j < pool_size:
                pool[j] = rec
    stats["passed_filters"] = kept
    stats["pool"] = len(pool)
    return pool, stats


# ---------------------------------------------------------------- models
class Models:
    """Loads LaBSE + opus-mt lazily. Any object with .embed() and .translate()
    works in score_pool(), which keeps the ranking logic testable without models."""

    def __init__(self, device: str, batch_size: int, beams: int):
        self.device, self.bs, self.beams = device, batch_size, beams
        self._labse = None
        self._mt: dict = {}

    def embed(self, texts: list[str]):
        if self._labse is None:
            from sentence_transformers import SentenceTransformer
            self._labse = SentenceTransformer(LABSE, device=self.device)
        return self._labse.encode(texts, batch_size=self.bs, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)

    def translate(self, texts: list[str], direction: str) -> list[str]:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        if direction not in self._mt:
            name = MT_MODELS[direction]
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name).to(self.device).eval()
            if self.device.startswith("cuda"):
                model = model.half()
            self._mt[direction] = (tok, model)
        tok, model = self._mt[direction]
        out: list[str] = []
        for i in range(0, len(texts), self.bs):
            chunk = texts[i:i + self.bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.inference_mode():
                gen = model.generate(**enc, num_beams=self.beams, max_new_tokens=96)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out


_CHRF = None
_TOK = __import__("re").compile(r"[\w']+", __import__("re").UNICODE)


def _chrf(hyp: str, ref: str) -> float:
    global _CHRF
    if _CHRF is None:
        from sacrebleu.metrics import CHRF
        _CHRF = CHRF()
    return _CHRF.sentence_score(hyp, [ref]).score


def divergence(literal: str, human: str) -> dict:
    """Two views of how far the human line departs from the literal MT.

    - chrF catches whole-line rewrites ("Well, you give!" -> "Wow, unbelievable!").
    - The longest replaced/inserted word span catches *local* swaps inside an
      otherwise literal line ("What devil ..." -> "What the hell ..."), which
      chrF alone scores as close (~67). Idioms are usually multi-word, so a
      span of >= 2 words is the useful signal; single-word swaps are mostly
      synonym noise between MT and subtitler.
    """
    import difflib
    lit = [t.lower() for t in _TOK.findall(literal)]
    hum = [t.lower() for t in _TOK.findall(human)]
    longest = 0
    changed = 0
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=lit, b=hum, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            longest = max(longest, j2 - j1)
            changed += j2 - j1
    return {
        "chrf": _chrf(literal, human),
        "span": longest,
        "novelty": changed / max(1, len(hum)),
    }


# ---------------------------------------------------------------- scoring
def score_pool(pool: list[dict], models, directions: list[str], min_align: float,
               max_chrf: float, min_span: int = 2, max_chrf_local: float = 85.0, log=print) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    if not pool:
        return [], stats

    t = time.time()
    e_ru = models.embed([p["ru"] for p in pool])
    e_en = models.embed([p["en"] for p in pool])
    aligned = []
    for p, a, b in zip(pool, e_ru, e_en):
        p["align_cos"] = round(float((a * b).sum()), 4)
        if p["align_cos"] >= min_align:
            aligned.append(p)
        else:
            stats["reject_misaligned"] += 1
    log(f"  alignment: {len(aligned)}/{len(pool)} kept (LaBSE >= {min_align}) in {time.time() - t:.0f}s")

    candidates = []
    for direction in directions:
        t = time.time()
        src_key, ref_key = ("ru", "en") if direction == "ru-en" else ("en", "ru")
        literal = models.translate([p[src_key] for p in aligned], direction)
        for p, lit in zip(aligned, literal):
            div = divergence(lit, p[ref_key])
            chrf = div["chrf"]
            global_rewrite = chrf <= max_chrf
            local_swap = div["span"] >= min_span and chrf <= max_chrf_local
            if not (global_rewrite or local_swap):
                stats[f"reject_literal_close_{direction}"] += 1
                continue
            rec = dict(p)
            rec.update({
                "direction": direction,
                "source": p[src_key],
                "human_translation": p[ref_key],
                "literal_mt": lit,
                "chrf_literal_vs_human": round(chrf, 1),
                "diverged_span_words": div["span"],
                "novelty": round(div["novelty"], 3),
                "divergence_kind": "rewrite" if global_rewrite else "local_swap",
            })
            candidates.append(rec)
        log(f"  {direction}: literal MT + chrF over {len(aligned)} pairs in {time.time() - t:.0f}s")

    if candidates:
        # Guard against divergence that is really a meaning change / mistranslation:
        # the literal MT should still mean roughly what the human line means.
        e_lit = models.embed([c["literal_mt"] for c in candidates])
        e_hum = models.embed([c["human_translation"] for c in candidates])
        for c, a, b in zip(candidates, e_lit, e_hum):
            c["sem_literal_vs_human"] = round(float((a * b).sum()), 4)
    return candidates, stats


def rank(candidates: list[dict], min_sem: float) -> list[dict]:
    ranked = []
    for c in candidates:
        if c.get("sem_literal_vs_human", 1.0) < min_sem:
            continue
        c["address"] = address_register(c["ru"])
        c["idioms"] = idiom_hits(c["ru"])
        divergence = max((100.0 - c["chrf_literal_vs_human"]) / 100.0, c.get("novelty", 0.0))
        c["score"] = round(c["align_cos"] * divergence + 0.1 * min(2, len(c["idioms"])), 4)
        c["bucket"] = "address" if c["address"] else "general"
        ranked.append(c)
    ranked.sort(key=lambda c: c["score"], reverse=True)
    return ranked


def select(ranked: list[dict], target: int, max_per_film: int, address_share: float = 0.3) -> list[dict]:
    """Top-N with a per-film cap so one talky film can't dominate, one record per
    source line (a pair can qualify in both directions), and the ты/вы bucket held
    to address_share of the target. The rest is not back-filled with ты/вы lines."""
    per_film: dict = defaultdict(int)
    seen_src: set[str] = set()
    max_address = int(target * address_share)
    n_address = 0
    out = []
    for c in ranked:
        film = c.get("film") or "unknown"
        src = normalize_key(c["source"])
        if per_film[film] >= max_per_film or src in seen_src:
            continue
        if c.get("bucket") == "address":
            if n_address >= max_address:
                continue
            n_address += 1
        per_film[film] += 1
        seen_src.add(src)
        out.append(c)
        if len(out) >= target:
            break
    return out


def write_outputs(selected: list[dict], stats: Counter, name: str, out_dir: Path,
                  seeds_path: Path, seed_count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / f"subs_candidates_{name}.jsonl"
    with open(cand_path, "w", encoding="utf-8") as f:
        for c in selected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    stats = Counter(stats)
    stats["selected"] = len(selected)
    stats["by_direction"] = dict(Counter(c["direction"] for c in selected))
    stats["by_bucket"] = dict(Counter(c.get("bucket") for c in selected))
    stats["by_bucket_direction"] = dict(Counter(f"{c.get('bucket')}/{c['direction']}" for c in selected))
    stats["with_address_pronoun"] = sum(1 for c in selected if c.get("address"))
    stats["with_idiom_hit"] = sum(1 for c in selected if c.get("idioms"))
    (out_dir / f"subs_stats_{name}.json").write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    with open(seeds_path, "w", encoding="utf-8") as f:
        for c in selected[:seed_count]:
            f.write(c["source"] + "\n")
    print(f"Wrote {len(selected)} candidates -> {cand_path}")
    print(f"Wrote {min(seed_count, len(selected))} seeds -> {seeds_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="OPUS moses zip (fetch_opensubtitles.py)")
    ap.add_argument("--name", required=True, help="Run name, e.g. subs1")
    ap.add_argument("--max-lines", type=int, default=None, help="Stop reading after N lines (quick trials)")
    ap.add_argument("--pool-size", type=int, default=60000, help="Filtered pairs sampled for model scoring")
    ap.add_argument("--target", type=int, default=2000, help="Candidates to keep after ranking")
    ap.add_argument("--seed-count", type=int, default=500, help="Top candidates written as Stage A seeds")
    ap.add_argument("--directions", default="ru-en,en-ru", help="Comma list: ru-en, en-ru")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=25)
    ap.add_argument("--min-align", type=float, default=0.75, help="LaBSE ru/en cosine to trust the alignment")
    ap.add_argument("--max-chrf", type=float, default=45.0, help="chrF(literal, human) at or below = divergent")
    ap.add_argument("--min-span", type=int, default=2,
                    help="Words in the longest human-only span that count as a local swap (idiom-sized)")
    ap.add_argument("--max-chrf-local", type=float, default=85.0,
                    help="chrF ceiling for local swaps (above this the lines are near-identical)")
    ap.add_argument("--min-sem", type=float, default=0.70, help="LaBSE literal/human cosine floor (drops meaning changes)")
    ap.add_argument("--max-per-film", type=int, default=15)
    ap.add_argument("--address-share", type=float, default=0.3,
                    help="Max fraction of the selection from the ты/вы bucket")
    ap.add_argument("--origin", choices=["any", "ru"], default="any",
                    help="ru: only films whose original language is Russian (Wikidata, cached)")
    ap.add_argument("--out-dir", type=Path, default=MINING_DIR, help="Where candidates, stats and seeds go")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--beams", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"{args.zip} not found -- run: python -m movie_mining.fetch_opensubtitles")
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    for d in directions:
        if d not in MT_MODELS:
            raise SystemExit(f"Unknown direction {d!r}; use ru-en and/or en-ru")
    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    films, origin_stats = None, {}
    if args.origin == "ru":
        from . import origin
        t = time.time()
        films, origin_stats = origin.films_with_origin(origin.film_keys(args.zip, args.max_lines), FILM_LANG_CACHE)
        print(f"Origin ru: {origin_stats} ({time.time() - t:.0f}s)")

    t = time.time()
    pool, stats = collect_pool(iter_pairs(args.zip, args.max_lines), args.pool_size,
                               random.Random(args.seed), args.min_words, args.max_words, films=films)
    stats.update(origin_stats)
    print(f"Read {stats['read']:,} lines, {stats['passed_filters']:,} passed filters, "
          f"pool {len(pool):,} ({time.time() - t:.0f}s)")

    models = Models(device, args.batch_size, args.beams)
    candidates, s2 = score_pool(pool, models, directions, args.min_align, args.max_chrf,
                                args.min_span, args.max_chrf_local)
    stats.update(s2)
    ranked = rank(candidates, args.min_sem)
    selected = select(ranked, args.target, args.max_per_film, args.address_share)
    write_outputs(selected, stats, args.name, args.out_dir, args.out_dir / f"seeds_subs_{args.name}.txt",
                  args.seed_count)


if __name__ == "__main__":
    main()
