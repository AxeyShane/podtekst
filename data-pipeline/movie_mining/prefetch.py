"""Download every model and dataset the movie-mining tools need, up front.

    python -m movie_mining.prefetch            # models + OpenSubtitles zip
    python -m movie_mining.prefetch --no-data  # models only

Models land in the Hugging Face cache (HF_HOME) and torch hub cache (TORCH_HOME);
setup_windows.ps1 points those at D: when C: is short on space. After this runs,
mining works offline.
"""
from __future__ import annotations

import argparse

from .diarize import MODEL as DIAR_MODEL
from .mine_subtitles import LABSE, MT_MODELS

# Skip weight formats we never load (TF/ONNX/GGUF/etc.) to save disk.
SKIP = ["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.gguf", "*.ot", "tf_model*", "flax_model*", "rust_model*",
        "openvino/*", "coreml/*"]


def fetch_hf(repo: str, extra_skip: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo, ignore_patterns=SKIP + (extra_skip or []))
    print(f"  ok  {repo} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-data", action="store_true", help="Skip the OpenSubtitles zip")
    ap.add_argument("--nemo", action="store_true", help="Also fetch the .nemo checkpoint (for the NeMo backend)")
    args = ap.parse_args()

    print("Text models:")
    fetch_hf(LABSE)
    for repo in MT_MODELS.values():
        fetch_hf(repo)

    print("Diarization model:")
    fetch_hf(DIAR_MODEL, None if args.nemo else ["*.nemo"])

    print("Demucs (stereo fallback):")
    try:
        from demucs.pretrained import get_model
        get_model("htdemucs")
        print("  ok  htdemucs")
    except Exception as e:  # demucs is optional -- only needed for stereo films
        print(f"  skipped htdemucs ({e.__class__.__name__}: {e})")

    if not args.no_data:
        print("OpenSubtitles RU-EN:")
        from .fetch_opensubtitles import DEFAULT_OUT, DEFAULT_URL, download
        download(DEFAULT_URL, DEFAULT_OUT)


if __name__ == "__main__":
    main()
