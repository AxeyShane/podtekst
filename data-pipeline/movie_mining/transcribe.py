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


def transcribe_clips(work_dir: Path, asr, force: bool = False) -> int:
    """Fills ru_text for clips that have no human subtitle text. Human text is
    never overwritten; --force redoes machine transcripts."""
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
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{work_dir.name}: transcribed {done} clips with {asr.name}")
    return done


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
    args = ap.parse_args()
    if args.asr == "gigaam":
        asr = GigaAM(args.gigaam_model)
        for wd in args.work_dirs:
            transcribe_clips(wd, asr, force=args.force)
        return
    for wd in args.work_dirs:
        out = transcribe(wd, args.model, not args.no_vad, args.threads, args.force)
        if out:
            print(f"{wd.name}: -> {out.name}")


if __name__ == "__main__":
    main()
