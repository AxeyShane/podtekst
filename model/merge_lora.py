"""Merge a LoRA adapter into its base and save a plain bf16 checkpoint -- the input for the
LiteRT-LM conversion step (see README.md).

    python merge_lora.py --base D:\\podtekst-mm\\models\\qvikhr-3-1.7b --adapter out\\qvikhr-lora-v1 --out out\\qvikhr-v1-merged
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
