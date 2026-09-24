"""Original language of OpenSubtitles films, for mining Russian-origin dialogue only.

OPUS .ids paths are <lang>/<year>/<imdb number>/<sub id>.xml.gz, so the film key
'year/NNNNN' carries a bare IMDb number (tt + zero-padded to 7 digits). Wikidata
maps IMDb ids (P345) to original language of film or TV show (P364). Results are
cached in <media root>/opensubtitles/film_lang.json; ids Wikidata doesn't know, or
titles without P364, are cached as [] and treated as not Russian.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

from .text_utils import film_id_from_ids_line

RUSSIAN = "Q7737"
# QLever serves the same Wikidata dump and answers a 200-id batch in ~1.5 s; the
# official endpoint took ~60 s and often 502'd, so it's only the fallback.
SPARQL_URLS = ("https://qlever.cs.uni-freiburg.de/api/wikidata", "https://query.wikidata.org/sparql")
USER_AGENT = "podtekst-mining/0.1 (open-source research; https://github.com/AxeyShane/podtekst)"


def imdb_id(film_key: str) -> str | None:
    """'1979/79679' -> 'tt0079679'."""
    num = film_key.rsplit("/", 1)[-1]
    return f"tt{int(num):07d}" if num.isdigit() else None


def film_keys(zip_path: Path, max_lines: int | None = None) -> set[str]:
    """Distinct 'year/imdb' keys in the first max_lines lines of the .ids member."""
    keys: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        name = next((n for n in zf.namelist() if n.endswith(".ids")), None)
        if name is None:
            raise SystemExit(f"{zip_path} has no .ids member -- can't filter by origin")
        with zf.open(name) as f:
            last = None
            for i, raw in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                path = raw.split(b"\t", 1)[0]
                if path == last:                 # consecutive lines share a film; skip the decode
                    continue
                last = path
                key = film_id_from_ids_line(path.decode("utf-8", "replace"))
                if key:
                    keys.add(key)
    return keys


def query_wikidata(ids: list[str], retries: int = 3) -> dict[str, list[str]]:
    """{imdb id: [language QIDs]} for one batch; ids Wikidata doesn't know are absent."""
    import requests
    values = " ".join(f'"{i}"' for i in ids)
    q = "PREFIX wdt: <http://www.wikidata.org/prop/direct/> " \
        f"SELECT ?imdb ?lang WHERE {{ VALUES ?imdb {{ {values} }} ?item wdt:P345 ?imdb . " \
        f"OPTIONAL {{ ?item wdt:P364 ?lang }} }}"
    errors = []
    for url in SPARQL_URLS:
        for attempt in range(retries):
            try:
                r = requests.post(url, data={"query": q}, timeout=120,
                                  headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT})
            except requests.RequestException as e:
                errors.append(f"{url}: {type(e).__name__}")
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                errors.append(f"{url}: {r.status_code}")
                time.sleep(min(30.0, float(r.headers.get("Retry-After", 5 * (attempt + 1)))))
                continue
            r.raise_for_status()
            out: dict[str, list[str]] = {}
            for b in r.json()["results"]["bindings"]:
                langs = out.setdefault(b["imdb"]["value"], [])
                if "lang" in b:
                    qid = b["lang"]["value"].rsplit("/", 1)[-1]
                    if qid not in langs:
                        langs.append(qid)
            return out
    raise RuntimeError(f"Wikidata query failed on every endpoint: {errors}")


def resolve_languages(ids: set[str], cache_path: Path, fetch=query_wikidata, batch: int = 200,
                      log=print) -> dict[str, list[str]]:
    """Original-language QIDs per IMDb id, from cache or Wikidata (saved after every batch)."""
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = sorted(i for i in ids if i not in cache)
    if todo:
        log(f"  origin: resolving {len(todo)} IMDb ids via Wikidata ({len(ids) - len(todo)} cached)")
    for n in range(0, len(todo), batch):
        chunk = todo[n:n + batch]
        found = fetch(chunk)
        for i in chunk:
            cache[i] = found.get(i, [])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")
    return {i: cache[i] for i in ids}


def films_with_origin(keys: set[str], cache_path: Path, lang_qid: str = RUSSIAN, fetch=query_wikidata,
                      log=print) -> tuple[set[str], dict]:
    """Film keys whose original languages include lang_qid, plus coverage stats."""
    by_key = {k: imdb_id(k) for k in keys}
    langs = resolve_languages({i for i in by_key.values() if i}, cache_path, fetch=fetch, log=log)
    keep = {k for k, i in by_key.items() if i and lang_qid in langs.get(i, [])}
    stats = {"origin_films_seen": len(keys),
             "origin_films_with_language": sum(1 for i in by_key.values() if i and langs.get(i)),
             "origin_films_kept": len(keep)}
    return keep, stats
