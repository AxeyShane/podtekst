"""Align a video's human-made Russian and English subtitle tracks and score each pair with LaBSE.

For videos whose creator uploaded both RU and EN subtitles (interviews, podcasts). The English
track is often a condensed translation with its own cue timings, so cues don't map 1:1.
A banded monotonic DP (same idea as Bertalign) groups 1-3 English cues with 1-5 Russian cues,
picking the grouping with the best LaBSE similarity among groups that overlap in time.

Usage (from data-pipeline/, with the movie-mining venv active):
  python -m movie_mining.align_subs <video>.ru.srt <video>.en.srt
Writes aligned_pairs.tsv next to the subtitles. Keep the subtitles and the TSV in the media root;
they hold verbatim dialogue and are never committed.
"""
import re, sys, os, csv
import numpy as np

MAX_EN, MAX_RU = 3, 5
TIME_SLACK = 3.0      # seconds of allowed mismatch between group time spans
BAND = 40.0           # only consider Russian cues starting within this many seconds of the English cue
SKIP_PENALTY = 0.30
MERGE_PENALTY = 0.04
TIME_WEIGHT = 0.3

def t2s(t):
    h, m, r = t.split(":"); s, ms = r.split(","); return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def clean(x):
    x = re.sub(r"♪[^♪]*♪", " ", x)                       # song lyrics between note marks
    x = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", x)          # (хлопок), [music]
    x = re.sub(r"^\s*[—–-]\s*", "", x); x = re.sub(r"\s[—–]\s", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def read_srt(p):
    out = []
    for b in re.split(r"\n\s*\n", open(p, encoding="utf-8-sig").read().strip()):
        L = b.strip().split("\n")
        if len(L) < 3 or "-->" not in L[1]: continue
        a, z = [s.strip() for s in L[1].split("-->")]
        txt = clean(" ".join(L[2:]))
        if not txt or re.fullmatch(r"[\d.,:\s]+", txt): continue
        out.append({"start": t2s(a), "end": t2s(z), "text": txt})
    for k, u in enumerate(out): u["id"] = str(k + 1)
    return out

def gtext(units, i, k):
    return " ".join(u["text"] for u in units[i:i+k])

def group_texts(en, ru):
    """Every text a group can take, so all embeddings can be computed in one batch."""
    texts = set()
    for i in range(len(en)):
        for a in range(1, MAX_EN+1):
            if i+a <= len(en): texts.add(gtext(en, i, a))
    for j in range(len(ru)):
        for b in range(1, MAX_RU+1):
            if j+b <= len(ru): texts.add(gtext(ru, j, b))
    return sorted(texts)

def align(en, ru, sim):
    """Banded monotonic DP. sim(en_text, ru_text) -> similarity. Returns TSV rows in order."""
    n, m = len(en), len(ru)
    INF = float("-inf")
    best = {(0, 0): 0.0}; back = {}
    lo = [0]*(n+1); hi = [m]*(n+1)          # Russian cue window per English cue, from timing
    for i in range(n+1):
        t = en[i]["start"] if i < n else en[-1]["end"]
        lo[i] = next((j for j in range(m) if ru[j]["start"] >= t - BAND), m)
        hi[i] = next((j for j in range(m) if ru[j]["start"] > t + BAND), m)
    lo[0] = 0; hi[n] = m
    def relax(key, v, prev):
        if v > best.get(key, INF): best[key] = v; back[key] = prev
    for i in range(n+1):
        for j in range(max(0, lo[i]-MAX_RU), min(m, hi[i]) + 1):
            if (i, j) not in best: continue
            cur = best[(i, j)]
            if i < n: relax((i+1, j), cur - SKIP_PENALTY, (i, j, "en_only"))
            if j < m: relax((i, j+1), cur - SKIP_PENALTY, (i, j, "ru_only"))
            for a in range(1, MAX_EN+1):
                if i+a > n: break
                es = (en[i]["start"], en[i+a-1]["end"]); et = gtext(en, i, a)
                for b in range(1, MAX_RU+1):
                    if j+b > m: break
                    rs = (ru[j]["start"], ru[j+b-1]["end"])
                    if min(es[1], rs[1]) + TIME_SLACK < max(es[0], rs[0]): continue
                    s = sim(et, gtext(ru, j, b))
                    ov = max(0.0, min(es[1], rs[1]) - max(es[0], rs[0])) / max(1e-6, max(es[1], rs[1]) - min(es[0], rs[0]))
                    relax((i+a, j+b), cur + s + TIME_WEIGHT*ov - MERGE_PENALTY*(a+b-2), (i, j, (a, b, s)))
    if (n, m) not in best:
        raise RuntimeError("Alignment did not reach the end; widen BAND.")
    rows, key = [], (n, m)
    while key != (0, 0):
        pi, pj, mv = back[key]
        if mv == "en_only": rows.append(("en_only", en[pi]["id"], "", en[pi]["start"], "", en[pi]["text"], ""))
        elif mv == "ru_only": rows.append(("ru_only", "", ru[pj]["id"], ru[pj]["start"], "", "", ru[pj]["text"]))
        else:
            a, b, s = mv
            rows.append((f"{a}:{b}", "+".join(u["id"] for u in en[pi:pi+a]), "+".join(u["id"] for u in ru[pj:pj+b]),
                         en[pi]["start"], round(s, 3), gtext(en, pi, a), gtext(ru, pj, b)))
        key = (pi, pj)
    return rows[::-1]

def main():
    ru_p, en_p = sys.argv[1], sys.argv[2]
    ru, en = read_srt(ru_p), read_srt(en_p)
    print(f"{len(en)} English cues, {len(ru)} Russian cues")
    texts = group_texts(en, ru)
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding {len(texts)} group texts...")
    E = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    idx = {t: k for k, t in enumerate(texts)}
    rows = align(en, ru, lambda a, b: float(np.dot(E[idx[a]], E[idx[b]])))
    out = os.path.join(os.path.dirname(os.path.abspath(ru_p)), "aligned_pairs.tsv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["kind", "en_ids", "ru_ids", "start", "score", "en", "ru"]); w.writerows(rows)
    paired = [r for r in rows if r[0][0].isdigit()]
    print(f"Wrote {out}: {len(paired)} pairs, {sum(1 for r in rows if r[0]=='en_only')} English-only, "
          f"{sum(1 for r in rows if r[0]=='ru_only')} Russian-only")
    sc = sorted(r[4] for r in paired)
    if sc: print("score quartiles:", [sc[int(q*(len(sc)-1))] for q in (0.1, 0.25, 0.5, 0.75)])

if __name__ == "__main__":
    main()
