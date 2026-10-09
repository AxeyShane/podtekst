# model/ — LoRA fine-tuning

Fine-tunes a small chat model to do Podtekst's one task: translate a RU↔EN message and say,
in a fixed JSON shape, whether a plain translation loses nuance.

```
{"translation": "...", "has_subtext": true, "category": "formality_shift", "nuance_note": "..."}
```

The prompt lives in `podtekst_sft/prompt.py` and is shared by training, evaluation and (later)
the keyboard, so the model sees the same framing everywhere.

## Setup (separate venv)

```powershell
python -m venv D:\podtekst-mm\venv-train
D:\podtekst-mm\venv-train\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA driver
pip install -r requirements-train.txt
```

## Steps

```powershell
cd model
python prepare_sft.py                              # dataset split -> data/sft_{train,test}.jsonl
python evaluate.py --base <base> --name baseline   # untuned model on the 342 test rows
python train_lora.py --base <base> --out out\smoke --max-steps 100   # pipeline smoke run
python train_lora.py --base <base> --out out\v1    # full run (2 epochs)
python evaluate.py --base <base> --adapter out\v1 --name v1
python merge_lora.py --base <base> --adapter out\v1 --out out\v1-merged
```

`<base>` is a local model folder, e.g. `D:\podtekst-mm\models\qvikhr-3-1.7b`
(QVikhr-3-1.7B-Instruction-noreasoning, Apache-2.0, a Russian-tuned Qwen3-1.7B).
For larger bases (Gemma 4 E4B) add `--qlora` to load them in 4-bit.

`evaluate.py` reports JSON validity, has_subtext precision / recall / false-positive rate,
per-category F1 and translation chrF, and writes every reply to `out/eval_<name>_preds.jsonl`
for review. `data/` and `out/` are git-ignored (they hold dataset rows, some derived from
film subtitles, and large checkpoints).

## On Kaggle, unattended (recommended)

`kaggle/launch.py` pushes a background job from the PC: it clones the repo, copies the SFT files
from the private `podtekst-sft` dataset (and refuses old pre-relabel data), trains, evaluates and
deletes the checkpoints. It runs as a saved version, so the browser can stay closed.

```
git push                                   # the job clones GitHub
python model/kaggle/launch.py --name v3 --epochs 2 --oversample "formality_shift=2,emotional_subtext=2"
kaggle kernels status akshaykharvi1/podtekst-train
kaggle kernels output akshaykharvi1/podtekst-train -p D:\podtekst-mm\kaggle-v3
```

Needs the Kaggle CLI with a token in `~/.kaggle` and a phone-verified account (GPU + internet).

## On Kaggle (no local download)

The base model never has to touch your machine: a Kaggle notebook pulls it from Hugging Face at
datacenter speed. Only the two SFT files (~4 MB) go up, as a **private** Kaggle dataset
(they hold film-subtitle sentences, so never make it public).

1. Kaggle → Datasets → New → upload `model/data/sft_train.jsonl` and `sft_test.jsonl`, visibility
   Private, name `podtekst-sft`.
2. New notebook → Settings: Accelerator **GPU T4 x2** (or P100), Internet **on** (needs a verified
   phone number). Add the `podtekst-sft` dataset as input.
3. Cells:

```
!git clone --depth 1 https://github.com/AxeyShane/podtekst
%cd podtekst/model
!pip install -q -r requirements-train.txt
!pip uninstall -y -q torchao   # Kaggle's old torchao (0.10) makes peft refuse to load
!mkdir -p data && find /kaggle/input -name "sft_*.jsonl" -exec cp {} data/ \;
!python -c "from huggingface_hub import snapshot_download; snapshot_download('Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning', local_dir='/kaggle/working/base')"
!python evaluate.py --base /kaggle/working/base --name baseline --limit 40
!python train_lora.py --base /kaggle/working/base --out out/qvikhr-lora-v1 --max-steps 100
```

T4 and P100 have no real bf16, so the scripts switch to fp16 automatically (LoRA weights stay
fp32). Download `out/` from the notebook's Output panel when it finishes (the adapter is tens of
MB). Kaggle gives about 30 GPU hours a week and 12 h per session.

## To the phone

`convert_litertlm.py` merges the adapter into the base and exports a `.litertlm` with LiteRT Torch
(`pip install litert-torch`). It needs Linux, so run it in the same Kaggle notebook after training:

```
!pip install -q litert-torch
!python convert_litertlm.py --base /tmp/base --adapter /kaggle/working/qvikhr-lora-v1 --out /kaggle/working/litert
```

The default is INT8 weights (`dynamic_wi8_afp32`, ~1.8 GB), the safest first test. `--quant
dynamic_wi4b32_afp32` gives INT4 (~1 GB, as in the litert-community builds); compare its answers
with the INT8 file before keeping it. Load the result in AI Edge Gallery on the phone, or check it
on a PC with `pip install litert-lm` and `litert-lm run <file>.litertlm --prompt=...`.
