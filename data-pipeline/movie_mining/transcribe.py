"""Russian transcripts for audio that has no subtitles: GigaAM v3 (default) or whisper.cpp.

    python -m movie_mining.transcribe <media root>/work/<film>                  # GigaAM, after cut_clips
    python -m movie_mining.transcribe <media root>/work/<film> --asr whisper    # whisper.cpp, before cut_clips

GigaAM v3 (default, MIT, Russian-specialised; beats Whisper-large-v3 ~70:30 in
the authors' side-by-side evals). The v3_e2e_rnnt variant outputs punctuated,
normalised text. Its .transcribe() handles audio up to 25 s, and our clips are
at most 12 s, so it runs per clip *after* cut_clips and fills the empty ru_text
fields in manifest.jsonl. That avoids GigaAM's long-form mode, which needs
pyannote plus a Hugging Face token. Caveat: GigaAM is also the ASR planned
for the app, so its own transcripts can't be used to *evaluate* it; they're
fine for clip text, seeds and emotion2vec checks.

--cross-check whisper (after GigaAM) re-transcribes each clip with whisper.cpp and
stores ru_text_whisper, asr_cer (character error rate vs ru_text, after lowercasing,
ё->е and dropping punctuation) and asr_agree (CER <= 0.10). Two independent ASRs
agreeing is a cheap confidence signal for pseudo-labels -- still not ground truth.

whisper.cpp (alternative) transcribes the whole dialogue stem *before*
cut_clips into ru.whisper.srt, as described below.

Runs whisper-cli on the cleaned dialogue.wav (already 16 kHz mono) and writes
ru.whisper.srt next to it. cut_clips.py uses a human ru.srt when one exists,
and only falls back to this machine transcript otherwise. The manifest marks
which one each clip's text came from.

Treat Whisper text as *pseudo-labels*: good for giving clips readable text,
spot-checking emotion2vec, and finding natural spoken lines to use as seeds.
For ASR evaluation you still need human transcripts.

whisper.cpp:
  * binary: whisper-cli(.exe). Found via WHISPER_CPP_BIN, then PATH.
    setup_windows.ps1 downloads the CUDA build and sets WHISPER_CPP_BIN.
  * model: ggml-large-v3 by default (best Russian accuracy; ~3 GB, fits the
    RTX 4060's 8 GB). --model large-v3-turbo is ~2x faster, slightly less accurate.
  * Silero VAD (--vad) skips music/silence, and -mc 0 (no carried-over text
    context) curbs Whisper's repetition loops on long films.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

WHISPER_REPO = "ggerganov/whisper.cpp"
VAD_REPO, VAD_FILE = "ggml-org/whisper-vad", "ggml-silero-v6.2.0.bin"
MODELS = {"large-v3": "ggml-large-v3.bin", "large-v3-turbo": "ggml-large-v3-turbo.bin",
          "large-v3-turbo-q8": "ggml-large-v3-turbo-q8_0.bin", "medium": "ggml-medium.bin"}
OUT_NAME = "ru.whisper.srt"


GIGAAM_DEFAULT = "v3_e2e_rnnt"


class GigaAM:
    """Thin wrapper so tests can swap in a fake with the same .transcribe(path) method."""

    def __init__(self, model_name: str = GIGAAM_DEFAULT, device: str | None = None):
        import gigaam
        kwargs = {"device": device} if device else {}
        try:
            self.model = gigaam.load_model(model_name, **kwargs)
        except TypeError:
            self.model = gigaam.load_model(model_name)
        self.name = f"gigaam_{model_name}"

    def transcribe(self, path: str) -> str:
        out = self.model.transcribe(path)
        return (out if isinstance(out, str) else getattr(out, "text", str(out))).strip()


def write_manifest(manifest: Path, rows: list[dict]) -> None:
    """Write to a temp file, then os.replace: a crash mid-write can't leave a half-written manifest."""
    import json
    tmp = manifest.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, manifest)


def transcribe_clips(work_dir: Path, asr, force: bool = False, checkpoint_every: int = 100) -> int:
    """Fills ru_text for clips that have no human subtitle text. Human text is
    never overwritten; --force redoes machine transcripts. The manifest is saved every
    checkpoint_every clips, and clips that already have ru_text are skipped, so an
    interrupted run resumes where it stopped."""
    import json
    work_dir = Path(work_dir)
    manifest = work_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} missing -- run cut_clips first")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = 0
    for r in rows:
        if r.get("ru_text_source") == "human_subs":
            continue
        if r.get("ru_text") and not force:
            continue
        r["ru_text"] = asr.transcribe(str(work_dir / r["clip"]))
        r["ru_text_source"] = asr.name if r["ru_text"] else None
        done += 1
        if done % checkpoint_every == 0:
            write_manifest(manifest, rows)
    write_manifest(manifest, rows)
    print(f"{work_dir.name}: transcribed {done} clips with {asr.name}")
    return done


# ------------------------------------------------------------ cross-check (--cross-check whisper)
_NON_WORD = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalize_for_cer(text: str) -> str:
    """Lowercase, ё->е, punctuation dropped, whitespace collapsed."""
    return " ".join(_NON_WORD.sub(" ", text.lower().replace("ё", "е")).split())


def cer(hyp: str, ref: str) -> float:
    """Character error rate of hyp against ref after normalize_for_cer (Levenshtein / len(ref))."""
    h, r = normalize_for_cer(hyp), normalize_for_cer(ref)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def whisper_clip_texts(clips_dir: Path, names: list[str], model: str = "large-v3", threads: int = 8,
                       binary: str | None = None, batch: int = 150) -> dict[str, str]:
    """Transcribe clip files with whisper-cli, many per call so the model loads once per batch
    (batches keep the Windows command line under its 32k limit). No VAD: clips are speech already."""
    binary = binary or find_binary()
    if not binary:
        raise SystemExit("whisper-cli not found (set WHISPER_CPP_BIN or run setup_windows.ps1)")
    model = model_path(model)
    out: dict[str, str] = {}
    for n in range(0, len(names), batch):
        chunk = names[n:n + batch]
        cmd = [binary, "-m", model, "-l", "ru", "-t", str(threads), "-nt", "-np", "-otxt"] + chunk
        res = subprocess.run(cmd, cwd=clips_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"whisper-cli failed ({res.returncode}):\n{res.stderr[-2000:]}")
        for name in chunk:
            txt = clips_dir / f"{name}.txt"                  # whisper-cli writes <input>.txt
            out[name] = " ".join(txt.read_text(encoding="utf-8").split()) if txt.exists() else ""
            txt.unlink(missing_ok=True)
        print(f"  whisper cross-check: {min(n + batch, len(names))}/{len(names)} clips")
    return out


def cross_check(work_dir: Path, transcribe_fn=None, max_cer: float = 0.10) -> dict:
    """Second ASR opinion on machine-transcribed clips: adds ru_text_whisper, asr_cer (Whisper
    vs ru_text) and asr_agree (asr_cer <= max_cer) to manifest.jsonl. Human-subtitle rows are
    skipped. transcribe_fn(clips_dir, names) -> {name: text}; defaults to whisper.cpp."""
    import json
    work_dir = Path(work_dir)
    manifest = work_dir / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [r for r in rows if r.get("ru_text_source") != "human_subs"]
    texts = (transcribe_fn or whisper_clip_texts)(work_dir / "clips", [Path(r["clip"]).name for r in todo])
    for r in todo:
        r["ru_text_whisper"] = texts.get(Path(r["clip"]).name, "")
        r["asr_cer"] = round(cer(r["ru_text_whisper"], r.get("ru_text") or ""), 3)
        r["asr_agree"] = r["asr_cer"] <= max_cer
    write_manifest(manifest, rows)
    agree = sum(r["asr_agree"] for r in todo)
    stats = {"cross_checked": len(todo), "asr_agree": agree,
             "asr_agree_pct": round(100 * agree / max(1, len(todo)), 1)}
    print(f"{work_dir.name}: {stats}")
    return stats


def find_binary() -> str | None:
    env = os.environ.get("WHISPER_CPP_BIN")
    if env and Path(env).exists():
        return env
    for name in ("whisper-cli", "whisper-cli.exe"):
        hit = shutil.which(name)
        if hit:
            return hit
    return None


def model_path(model: str) -> str:
    """A local .bin path is used as-is; a short name is fetched into the HF cache."""
    if Path(model).exists():
        return str(model)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(WHISPER_REPO, MODELS.get(model, model))


def vad_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(VAD_REPO, VAD_FILE)


def build_command(binary: str, model: str, wav: Path, out_base: Path, language: str = "ru",
                  threads: int = 8, vad: str | None = None) -> list[str]:
    cmd = [binary, "-m", model, "-f", str(wav), "-l", language, "-osrt", "-of", str(out_base),
           "-t", str(threads), "-mc", "0", "-np"]
    if vad:
        cmd += ["--vad", "--vad-model", vad]
    return cmd


def transcribe(work_dir: Path, model: str = "large-v3", use_vad: bool = True, threads: int = 8,
               force: bool = False, binary: str | None = None) -> Path | None:
    work_dir = Path(work_dir)
    out = work_dir / OUT_NAME
    if out.exists() and not force:
        return out
    binary = binary or find_binary()
    if not binary:
        print("whisper-cli not found (set WHISPER_CPP_BIN or run setup_windows.ps1) -- skipping transcription")
        return None
    cmd = build_command(binary, model_path(model), work_dir / "dialogue.wav", out.with_suffix(""),
                        threads=threads, vad=vad_path() if use_vad else None)
    print(f"{work_dir.name}: transcribing with whisper.cpp ({model})...")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f"whisper-cli failed ({res.returncode}):\n{res.stderr[-2000:]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dirs", nargs="+", type=Path, help="Folders made by extract_dialogue.py")
    ap.add_argument("--asr", choices=["gigaam", "whisper"], default="gigaam")
    ap.add_argument("--gigaam-model", default=GIGAAM_DEFAULT)
    ap.add_argument("--model", default="large-v3", help=f"whisper: {', '.join(MODELS)} or a path to a ggml .bin")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Re-transcribe even if ru.whisper.srt exists")
    ap.add_argument("--cross-check", choices=["none", "whisper"], default="none",
                    help="whisper: re-transcribe each clip with whisper.cpp and flag GigaAM/Whisper agreement")
    args = ap.parse_args()
    if args.asr == "gigaam":
        asr = GigaAM(args.gigaam_model)
        for wd in args.work_dirs:
            transcribe_clips(wd, asr, force=args.force)
            if args.cross_check == "whisper":
                cross_check(wd, lambda d, n: whisper_clip_texts(d, n, args.model, args.threads))
        return
    for wd in args.work_dirs:
        out = transcribe(wd, args.model, not args.no_vad, args.threads, args.force)
        if out:
            print(f"{wd.name}: -> {out.name}")


if __name__ == "__main__":
    main()
