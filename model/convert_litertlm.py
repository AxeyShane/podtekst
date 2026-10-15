"""Turn a trained adapter into a .litertlm file the phone can load (LiteRT-LM / AI Edge Gallery).

    # merge the LoRA adapter into the base, then export (Linux or Kaggle; litert-torch has no Windows build)
    python convert_litertlm.py --base /tmp/base --adapter /kaggle/working/qvikhr-lora-v1 --out /kaggle/working/litert

    # already merged (merge_lora.py) -> export only
    python convert_litertlm.py --merged out/qvikhr-v1-merged --out out/litert --quant dynamic_wi4b32_afp32

Quantization (AI Edge Quantizer recipe names):
  dynamic_wi8_afp32     INT8 weights, the converter's default (~1.8 GB for 1.7B). Safest first test.
  dynamic_wi4b32_afp32  INT4 weights in blocks of 32 (~1 GB), as used by litert-community builds. Smaller and
                        faster; check quality against the INT8 file before keeping it.
  ''                    no quantization (float; for debugging only).

The KV cache is sized for keyboard messages: system prompt + message + JSON answer is ~250 tokens
(train max 254), so 1024 leaves room for long messages without wasting phone memory.
"""
import argparse
import json
from pathlib import Path


MERGE_DTYPES = ("float32", "bfloat16")


def tokenizer_source(base: str, adapter: str) -> str:
    """The adapter folder when it carries tokenizer files, otherwise the base (a QLoRA run can save
    the adapter weights alone)."""
    has = any((Path(adapter) / f).is_file() for f in ("tokenizer.json", "tokenizer_config.json"))
    return adapter if has else base


def merge(base: str, adapter: str, out_dir: Path, dtype: str = "float32") -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=getattr(torch, dtype))
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok = tokenizer_source(base, adapter)
    AutoTokenizer.from_pretrained(tok).save_pretrained(out_dir)
    print(f"merged ({dtype}, tokenizer from {tok}) -> {out_dir}")
    return out_dir


def check_architecture(model_dir: Path) -> str:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    arch = (cfg.get("architectures") or ["?"])[0]
    supported = {"Qwen3ForCausalLM", "Qwen2ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM",
                 "Gemma3ForCausalLM", "Gemma3nForCausalLM", "Gemma4ForCausalLM", "SmolLM3ForCausalLM"}
    if arch not in supported:
        raise SystemExit(f"{arch} is not on litert-torch's export_hf list: {sorted(supported)}")
    return arch


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--merged", help="Folder with an already merged HF checkpoint")
    src.add_argument("--adapter", help="LoRA adapter folder (needs --base); merged first")
    ap.add_argument("--base", help="Base model folder, with --adapter")
    ap.add_argument("--out", required=True, help="Output folder for the .litertlm")
    ap.add_argument("--quant", default="dynamic_wi8_afp32", help="AI Edge Quantizer recipe name or JSON path")
    ap.add_argument("--cache-length", type=int, default=1024)
    ap.add_argument("--merge-dtype", choices=MERGE_DTYPES, default="float32",
                    help="Precision of the merge with --adapter. float32 needs ~4 bytes per parameter of RAM "
                         "(~16 GB for a 4B base); bfloat16 halves that if the merge runs out of memory")
    args = ap.parse_args()
    out = Path(args.out)

    if args.adapter:
        if not args.base:
            raise SystemExit("--adapter needs --base")
        model_dir = merge(args.base, args.adapter, out / "merged", args.merge_dtype)
    else:
        model_dir = Path(args.merged)
    print(f"architecture {check_architecture(model_dir)}; quantization {args.quant or 'none'}; "
          f"cache {args.cache_length}")

    from litert_torch.generative.export_hf.export import export
    export(model=str(model_dir), output_dir=str(out), quantization_recipe=args.quant,
           cache_length=args.cache_length)
    files = sorted(out.glob("*.litertlm"))
    for f in files:
        print(f"-> {f}  ({f.stat().st_size / 1e9:.2f} GB)")
    if not files:
        print(f"export finished but no .litertlm in {out}; look at its contents")


if __name__ == "__main__":
    main()
