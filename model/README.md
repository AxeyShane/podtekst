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

The merged checkpoint is converted to `.litertlm` with LiteRT Torch (`litert-torch`) and
quantized to INT4 with AI Edge Quantizer — the same path `litert-community/Qwen3-1.7B` used
(932 MB INT4 build). Check it on the PC with `pip install litert-lm` and
`litert-lm run <file>.litertlm --prompt=...`, then load it in AI Edge Gallery on the phone.
The conversion script is added once the merge step has produced a checkpoint.
