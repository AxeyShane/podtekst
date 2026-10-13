"""Start a training run on Kaggle from the PC, no browser needed.

    python model/kaggle/launch.py --name v3 --epochs 2 --oversample "formality_shift=2,emotional_subtext=2"

Then check and download (same Kaggle CLI token as before):

    kaggle kernels status <owner>/podtekst-train
    kaggle kernels output <owner>/podtekst-train -p D:\\podtekst-mm\\kaggle-<name>

Each push creates a new saved version of the `podtekst-train` script kernel with a T4 GPU, internet
on and the private `podtekst-sft` dataset attached. Push the repo to GitHub first: the job clones it.
"""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_script(run: dict) -> str:
    src = (HERE / "train_kaggle.py").read_text(encoding="utf-8")
    line = f"RUN = {run!r}  # filled in by launch.py"  # Python literal: True/False, not JSON true/false
    out, n = re.subn(r"^RUN = \{.*?\}  # launch\.py replaces this line$", lambda m: line, src, flags=re.M | re.S)
    if n != 1:
        raise SystemExit("could not find the RUN line in train_kaggle.py")
    return out


def metadata(owner: str, slug: str, dataset, gpu: bool = True) -> dict:
    datasets = [dataset] if isinstance(dataset, str) else list(dataset)
    return {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "train_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{d}" for d in datasets],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="", help="Run name, e.g. v3 (adapter folder lora-<name>)")
    ap.add_argument("--eval-bases", default="",
                    help="Comma-separated HF model ids to score zero-shot instead of training")
    ap.add_argument("--base", default="Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning",
                    help="HF model id to fine-tune, e.g. RefalMachine/RuadaptQwen3-4B-Instruct")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--oversample", default="formality_shift=2,emotional_subtext=2")
    ap.add_argument("--batch", type=int, default=4,
                    help="Per-step batch; use 2 for ~4B models on a T4 (grad-accum keeps the effective batch)")
    ap.add_argument("--grad-accum", type=int, default=0, help="Default: 16 / batch (effective batch 16 per GPU)")
    ap.add_argument("--qlora", action="store_true", help="Load the base in 4-bit (last resort for memory)")
    ap.add_argument("--convert", default="",
                    help="Private Kaggle dataset holding a LoRA adapter (e.g. podtekst-lora-v2b): merge it into "
                         "--base and export a .litertlm for the phone instead of training (CPU job)")
    ap.add_argument("--quant", default="dynamic_wi8_afp32",
                    help="With --convert: dynamic_wi8_afp32 (INT8, safest) or dynamic_wi4b32_afp32 (INT4, smaller)")
    ap.add_argument("--expect", default="",
                    help="Fail before training unless the dataset matches, e.g. train=3451,test=380,formality_shift=296")
    ap.add_argument("--owner", default="akshaykharvi1")
    ap.add_argument("--slug", default="", help="Kaggle kernel slug (default podtekst-train / podtekst-eval)")
    ap.add_argument("--dataset", default="podtekst-sft")
    ap.add_argument("--kaggle", default="kaggle", help="Path to the kaggle CLI")
    ap.add_argument("--dry-run", action="store_true", help="Write the folder and print it, don't push")
    args = ap.parse_args()

    bases = [b.strip() for b in args.eval_bases.split(",") if b.strip()]
    if not bases and not args.name and not args.convert:
        raise SystemExit("pass --name (training), --eval-bases (zero-shot comparison) or --convert (phone export)")
    mode = "convert" if args.convert else "eval" if bases else "train"
    args.slug = args.slug or {"convert": "podtekst-convert", "eval": "podtekst-eval", "train": "podtekst-train"}[mode]
    if mode == "convert" and not args.name:
        args.name = args.convert.removeprefix("podtekst-lora-")
    run = {"mode": mode, "bases": bases, "adapter": args.convert, "quant": args.quant,
           "name": args.name or "zs", "epochs": args.epochs, "oversample": args.oversample,
           "repo": "https://github.com/AxeyShane/podtekst",
           "base": args.base, "max_formality_rows": 400,
           "batch": args.batch, "grad_accum": args.grad_accum or max(1, 16 // args.batch), "qlora": args.qlora,
           "expect": {k.strip(): int(v) for k, v in (p.split("=") for p in args.expect.split(",") if p.strip())}}
    folder = Path(tempfile.mkdtemp(prefix="podtekst-kaggle-"))
    (folder / "train_kaggle.py").write_text(build_script(run), encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata(args.owner, args.slug, [args.convert] if mode == "convert" else [args.dataset],
                            gpu=mode != "convert"), indent=2), encoding="utf-8")
    print(f"kernel folder: {folder}")
    if args.dry_run:
        return
    subprocess.run([args.kaggle, "kernels", "push", "-p", str(folder)], check=True)
    shutil.rmtree(folder, ignore_errors=True)
    print(f"\nStarted. Check:    {args.kaggle} kernels status {args.owner}/{args.slug}")
    print(f"Download after:   {args.kaggle} kernels output {args.owner}/{args.slug} -p <folder>")


if __name__ == "__main__":
    main()
