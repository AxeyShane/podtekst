"""Step 2 of the audio track: who speaks when, with NVIDIA Nemotron 3 Diarization.

    python -m movie_mining.diarize <media root>/work/<film>

Reads dialogue.wav, writes segments.json: [{"start": s, "end": s, "speaker": k}, ...].

Model: nvidia/Nemotron-3-Diarization (Streaming Sortformer, 100M params, up to
8 speakers, 16 kHz mono, OpenMDW-1.1 -- commercial use permitted). Uses the
"very high latency" (30.4 s buffer) profile from the model card: this is
offline batch mining, so we take the most accurate setting.

Caveats (see docs/DESIGN.md): Russian isn't in the model's listed training
languages -- hand-check a few scenes before trusting a large run. Accuracy
drops with 5+ speakers in a scene.

Two backends (--backend, default auto = NeMo if installed, else transformers):
  * nemo          -- the model card's reference path (`nemo-toolkit[asr]`). Best on
                     Linux; on Windows run it inside WSL2 Ubuntu with CUDA.
  * transformers  -- runs natively on Windows (AutoModelForAudioFrameClassification).
                     Audio is processed in --window-seconds windows; per-frame
                     speaker probabilities are thresholded into segments here.
                     Newer integration -- sanity-check its segments against a few
                     minutes you've listened to before a big run.
Labels are local to one file ("speaker 2"), not actor identities. With
--chunk-minutes, labels are also local to each chunk (prefixed c0_, c1_...),
which is fine for clip extraction.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

MODEL = "nvidia/Nemotron-3-Diarization"
# Model card, "very high latency" profile (units: 80 ms frames).
VERY_HIGH_LATENCY = {"spkcache_len": 264, "fifo_len": 40, "chunk_len": 340,
                     "chunk_right_context": 40, "spkcache_update_period": 300}
_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_segment(seg) -> dict | None:
    """NeMo returns segments as strings like '12.34 15.60 speaker_1' (or with
    commas); be tolerant of both and of tuple/list forms."""
    if isinstance(seg, (list, tuple)) and len(seg) >= 3:
        start, end, spk = seg[0], seg[1], seg[2]
    else:
        parts = re.split(r"[,\s]+", str(seg).strip())
        if len(parts) < 3:
            return None
        start, end, spk = parts[0], parts[1], parts[2]
    try:
        start, end = float(start), float(end)
    except ValueError:
        return None
    m = _NUM.findall(str(spk))
    speaker = str(int(float(m[-1]))) if m else str(spk)
    if end <= start:
        return None
    return {"start": round(start, 3), "end": round(end, 3), "speaker": speaker}


def load_model(device: str | None):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained(MODEL)
    model.eval()
    for k, v in VERY_HIGH_LATENCY.items():
        setattr(model.sortformer_modules, k, v)
    model._check_streaming_parameters()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev)


def _duration(wav: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=nw=1:nk=1", str(wav)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# ------------------------------------------------------------ transformers backend
class TransformersDiarizer:
    def __init__(self, device: str | None, window_seconds: float = 120.0, threshold: float = 0.5):
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(MODEL).to(self.dev).eval()
        self.window, self.threshold = window_seconds, threshold

    def _probs(self, audio, sr: int):
        import numpy as np
        import torch
        try:
            inputs = self.proc(audio, sampling_rate=sr, return_tensors="pt")
        except TypeError:
            inputs = self.proc(audio=audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs)
        x = (out.logits if hasattr(out, "logits") else out[0])[0].float()
        if x.dim() == 2 and x.shape[0] < x.shape[1] and x.shape[0] <= 8:   # [spk, T] -> [T, spk]
            x = x.T
        if float(x.min()) < 0 or float(x.max()) > 1:
            x = torch.sigmoid(x)
        probs = x.cpu().numpy()
        return probs, (len(audio) / sr) / max(1, probs.shape[0])

    def diarize(self, wav: Path) -> list[dict]:
        import soundfile as sf
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        step = int(self.window * sr)
        segs: list[dict] = []
        for i, a in enumerate(range(0, len(audio), step)):
            chunk = audio[a:a + step]
            if len(chunk) < sr:              # skip a sub-second tail
                continue
            probs, frame_sec = self._probs(chunk, sr)
            segs += probs_to_segments(probs, frame_sec, offset=a / sr, threshold=self.threshold,
                                      prefix=f"w{i}_")
        return segs


def probs_to_segments(probs, frame_sec: float, offset: float = 0.0, threshold: float = 0.5,
                      min_on: float = 0.25, min_gap: float = 0.2, prefix: str = "") -> list[dict]:
    """[T, speakers] activity probabilities -> [{start, end, speaker}].
    Gaps shorter than min_gap are bridged; runs shorter than min_on are dropped."""
    import numpy as np
    probs = np.asarray(probs)
    segs = []
    for spk in range(probs.shape[1]):
        active = probs[:, spk] >= threshold
        runs, start = [], None
        for t, on in enumerate(np.append(active, False)):
            if on and start is None:
                start = t
            elif not on and start is not None:
                runs.append([start * frame_sec, t * frame_sec])
                start = None
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] < min_gap:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        for s, e in merged:
            if e - s >= min_on:
                segs.append({"start": round(s + offset, 3), "end": round(e + offset, 3),
                             "speaker": f"{prefix}{spk}"})
    return sorted(segs, key=lambda x: x["start"])


class NemoDiarizer:
    def __init__(self, device: str | None, chunk_minutes: float = 0):
        self.model, self.chunk_minutes = load_model(device), chunk_minutes

    def diarize(self, wav: Path) -> list[dict]:
        return diarize_file(self.model, wav, self.chunk_minutes)


def make_diarizer(backend: str = "auto", device: str | None = None, chunk_minutes: float = 0,
                  window_seconds: float = 120.0):
    if backend == "auto":
        try:
            import nemo.collections.asr  # noqa: F401
            backend = "nemo"
        except Exception:
            backend = "transformers"
    print(f"Diarization backend: {backend}")
    if backend == "nemo":
        return NemoDiarizer(device, chunk_minutes)
    return TransformersDiarizer(device, window_seconds)


def diarize_file(model, wav: Path, chunk_minutes: float = 0) -> list[dict]:
    if not chunk_minutes:
        raw = model.diarize(audio=[str(wav)], batch_size=1)[0]
        return [s for s in (parse_segment(x) for x in raw) if s]
    segs: list[dict] = []
    total, step = _duration(wav), chunk_minutes * 60
    with tempfile.TemporaryDirectory() as tmp:
        for i, off in enumerate(range(0, int(total) + 1, int(step))):
            part = Path(tmp) / f"c{i}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(off), "-t", str(step), "-i", str(wav),
                            str(part)], check=True)
            for s in (parse_segment(x) for x in model.diarize(audio=[str(part)], batch_size=1)[0]):
                if s:
                    segs.append({"start": round(s["start"] + off, 3), "end": round(s["end"] + off, 3),
                                 "speaker": f"c{i}_{s['speaker']}"})
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dirs", nargs="+", type=Path, help="Folders made by extract_dialogue.py")
    ap.add_argument("--backend", choices=["auto", "nemo", "transformers"], default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk-minutes", type=float, default=0,
                    help="NeMo: split long audio into chunks (use if a full film runs out of GPU memory)")
    ap.add_argument("--window-seconds", type=float, default=120.0, help="transformers: window length")
    args = ap.parse_args()
    diarizer = make_diarizer(args.backend, args.device, args.chunk_minutes, args.window_seconds)
    for wd in args.work_dirs:
        segs = diarizer.diarize(wd / "dialogue.wav")
        (wd / "segments.json").write_text(json.dumps(segs, indent=1), encoding="utf-8")
        print(f"{wd.name}: {len(segs)} segments, {len({s['speaker'] for s in segs})} speaker labels")


if __name__ == "__main__":
    main()
