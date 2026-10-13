"""Kaggle background job: train one LoRA adapter and evaluate it, or (mode "eval") score several
untrained base models zero-shot on the test set. Only the useful output is kept.

Started from the PC with `python model/kaggle/launch.py --name v3 ...`, which fills in RUN below and
pushes this script as a Kaggle "script" kernel (GPU + internet on, the podtekst-sft dataset attached).
The run is a saved version: it keeps going with the browser closed, and
`kaggle kernels output <owner>/podtekst-train -p <folder>` downloads the adapter and eval files.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

RUN = {"name": "v3", "epochs": 2, "oversample": "formality_shift=2,emotional_subtext=2",
       "repo": "https://github.com/AxeyShane/podtekst", "base": "Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning",
       "max_formality_rows": 400}  # launch.py replaces this line

WORK = "/kaggle/working"
CODE = "/tmp/podtekst"
MODEL = f"{CODE}/model"
BASE = "/tmp/base"
ADAPTER = f"{WORK}/lora-{RUN['name']}"
# Fewer out-of-memory failures from fragmentation on the 15 GB T4 (inherited by the subprocesses).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def sh(cmd, cwd=None):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def main():
    print("run:", json.dumps(RUN), flush=True)
    sh(f"rm -rf {CODE} && git clone -q --depth 1 {RUN['repo']} {CODE}")
    if RUN.get("mode") == "convert":
        convert()
        return
    sh("pip install -q -r requirements-train.txt && pip uninstall -y -q torchao", cwd=MODEL)

    # Training data comes only from the attached dataset, never from an older run's output.
    files = sorted(glob.glob("/kaggle/input/**/podtekst-sft/sft_*.jsonl", recursive=True))
    if len(files) != 2:
        sys.exit(f"expected sft_train.jsonl and sft_test.jsonl from the podtekst-sft dataset, found {files}")
    os.makedirs(f"{MODEL}/data", exist_ok=True)
    for f in files:
        print("copying", f, flush=True)
        shutil.copy(f, f"{MODEL}/data/")
    counts = data_counts(f"{MODEL}/data")
    print("data counts:", counts, flush=True)
    if counts["formality_shift"] > RUN["max_formality_rows"]:
        sys.exit("training data looks like the pre-relabel version; upload the new files to the dataset")
    wrong = {k: (counts[k], v) for k, v in RUN.get("expect", {}).items() if counts.get(k) != v}
    if wrong:
        sys.exit(f"dataset does not match the expected counts (got, expected): {wrong}")

    if RUN.get("mode") == "eval":
        eval_bases()
        return

    sh(f"python -c \"from huggingface_hub import snapshot_download; "
       f"snapshot_download('{RUN['base']}', local_dir='{BASE}')\"")
    train = (f"python train_lora.py --base {BASE} --out {ADAPTER} --epochs {RUN['epochs']} "
             f"--batch {RUN.get('batch', 4)} --grad-accum {RUN.get('grad_accum', 4)}")
    if RUN.get("qlora"):
        train += " --qlora"
    if RUN["oversample"]:
        train += f" --oversample '{RUN['oversample']}'"
    sh(train, cwd=MODEL)
    # Evaluate on the full-precision base even after QLoRA training: that is the model the phone
    # conversion merges into, and inference alone fits in fp16.
    sh(f"python evaluate.py --base {BASE} --adapter {ADAPTER} --name {RUN['name']}-full", cwd=MODEL)

    os.makedirs(f"{WORK}/eval", exist_ok=True)
    for f in glob.glob(f"{MODEL}/out/eval_*"):
        shutil.copy(f, f"{WORK}/eval/")
    for d in glob.glob(f"{ADAPTER}/checkpoint-*"):
        shutil.rmtree(d)
    sh(f"ls -la {WORK}/eval {ADAPTER} && du -sh {WORK}")


def data_counts(folder: str) -> dict:
    """Row counts of both splits plus formality_shift rows in train (the relabel check)."""
    train = [json.loads(l)["row"] for l in open(f"{folder}/sft_train.jsonl", encoding="utf-8") if l.strip()]
    test = sum(1 for l in open(f"{folder}/sft_test.jsonl", encoding="utf-8") if l.strip())
    return {"train": len(train), "test": test,
            "formality_shift": sum(r["category"] == "formality_shift" for r in train)}


def convert():
    """CPU job: merge a LoRA adapter (attached as a private dataset) into its base and export a .litertlm."""
    # litert-torch needs a recent torchao, so Kaggle's torchao is upgraded here rather than removed.
    sh("pip install -q -r requirements-train.txt && pip install -q litert-torch", cwd=MODEL)
    configs = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
    if len(configs) != 1:
        sys.exit(f"expected one adapter in the attached dataset, found {configs}")
    adapter = os.path.dirname(configs[0])
    print("adapter:", adapter, flush=True)
    sh(f"python -c \"from huggingface_hub import snapshot_download; "
       f"snapshot_download('{RUN['base']}', local_dir='{BASE}')\"")
    tmp = "/tmp/litert"  # the merged fp32 checkpoint stays out of the saved output
    sh(f"python convert_litertlm.py --base {BASE} --adapter {adapter} --out {tmp} --quant {RUN['quant']}", cwd=MODEL)
    files = glob.glob(f"{tmp}/*.litertlm")
    if not files:
        sh(f"ls -laR {tmp} | head -50")
        sys.exit("export produced no .litertlm")
    tag = "int4" if "wi4" in RUN["quant"] else "int8" if "wi8" in RUN["quant"] else "float"
    for f in files:
        shutil.copy(f, f"{WORK}/podtekst-{RUN['name']}-{tag}.litertlm")
    sh(f"ls -la {WORK} && du -sh {WORK}")


def short_name(repo_id: str) -> str:
    return "zs-" + repo_id.split("/")[-1].lower().replace(".", "-")


def eval_bases():
    """Zero-shot: each base model answers the test set with the same prompt, no adapter."""
    os.makedirs(f"{WORK}/eval", exist_ok=True)
    for repo_id in RUN["bases"]:
        local = f"/tmp/zs-model"
        shutil.rmtree(local, ignore_errors=True)
        sh(f"python -c \"from huggingface_hub import snapshot_download; "
           f"snapshot_download('{repo_id}', local_dir='{local}')\"")
        name = short_name(repo_id)
        try:
            sh(f"python evaluate.py --base {local} --name {name}", cwd=MODEL)
        except subprocess.CalledProcessError as e:  # one model failing must not lose the others
            print(f"!! {repo_id} failed: {e}", flush=True)
        for f in glob.glob(f"{MODEL}/out/eval_{name}*"):
            shutil.copy(f, f"{WORK}/eval/")
        shutil.rmtree(local, ignore_errors=True)
    sh(f"ls -la {WORK}/eval")


if __name__ == "__main__":
    main()
