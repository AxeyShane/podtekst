"""Step 2 of the audio track: who speaks when, with NVIDIA Nemotron 3 Diarization.

    python -m movie_mining.diarize raw-media/work/<film>

Reads dialogue.wav, writes segments.json: [{"start": s, "end": s, "speaker": k}, ...].

Model: nvidia/Nemotron-3-Diarization (Streaming Sortformer, 100M params, up to
8 speakers, 16 kHz mono, OpenMDW-1.1 -- commercial use permitted). Uses the
"very high latency" (30.4 s buffer) profile from the model card: this is
offline batch mining, so we take the most accurate setting.

Caveats (see docs/DESIGN.md): Russian isn't in the model's listed training
languages -- hand-check a few scenes before trusting a large run. Accuracy
drops with 5+ speakers in a scene.

Needs NeMo (`nemo-toolkit[asr]`, see requirements-audio.txt). NeMo is best
supported on Linux; on Windows run this step inside WSL2 (Ubuntu) with CUDA.
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
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk-minutes", type=float, default=0,
                    help="Split long audio into chunks (use if a full film runs out of GPU memory)")
    args = ap.parse_args()
    model = load_model(args.device)
    for wd in args.work_dirs:
        segs = diarize_file(model, wd / "dialogue.wav", args.chunk_minutes)
        (wd / "segments.json").write_text(json.dumps(segs, indent=1), encoding="utf-8")
        print(f"{wd.name}: {len(segs)} segments, {len({s['speaker'] for s in segs})} speaker labels")


if __name__ == "__main__":
    main()
