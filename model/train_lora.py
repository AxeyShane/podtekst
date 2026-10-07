"""LoRA fine-tune on model/data/sft_train.jsonl (run prepare_sft.py first).

Loss is computed on the assistant JSON only; the system prompt and the message are masked.
Defaults fit an 8 GB GPU for a ~1.7B model in bf16. --qlora loads the base in 4-bit
(bitsandbytes NF4) for bigger models such as Gemma 4 E4B.

    python train_lora.py --base D:\\podtekst-mm\\models\\qvikhr-3-1.7b --out out\\qvikhr-lora-v1
    python train_lora.py --base ... --out out\\smoke --max-steps 100      # quick pipeline test
"""
import argparse
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def common_prefix_len(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def as_ids(out) -> list:
    """apply_chat_template(tokenize=True) returns a plain id list in transformers 4.x but a
    BatchEncoding dict in 5.x; normalise both to a flat list of ints."""
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def tokenize_example(tok, messages: list[dict], max_len: int) -> dict | None:
    """input_ids + labels with everything before the assistant answer masked to -100.
    Returns None when the example doesn't fit max_len (it is skipped, never truncated --
    a cut-off JSON target would teach the model to stop mid-object)."""
    prompt_ids = as_ids(tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    full_ids = as_ids(tok.apply_chat_template(messages, tokenize=True))
    # Usually the prompt is an exact prefix. Some templates (Qwen3's empty <think></think> block)
    # render the generation prompt slightly differently, so mask up to the common prefix: the
    # loss then starts at the first token the model actually has to produce.
    cut = common_prefix_len(prompt_ids, full_ids)
    if len(full_ids) > max_len or cut >= len(full_ids):
        return None
    labels = [-100] * cut + full_ids[cut:]
    return {"input_ids": full_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
    lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
    att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}


def supported_kwargs(cls, kw: dict) -> dict:
    """Drop arguments this transformers version no longer accepts (5.x removed several), with a note."""
    import inspect
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    dropped = sorted(k for k in kw if k not in params)
    if dropped:
        print(f"note: this transformers version ignores {', '.join(dropped)}")
    return {k: v for k, v in kw.items() if k in params}


def pick_dtype():
    """bf16 on Ampere or newer (RTX 30/40, A100, L4); fp16 on older cards such as Kaggle's T4 and P100,
    where bf16 is missing or only emulated."""
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Local path or HF id of the base model")
    ap.add_argument("--out", required=True, help="Where the LoRA adapter is saved")
    ap.add_argument("--train", type=Path, default=HERE / "data" / "sft_train.jsonl")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="Override epochs (e.g. 100 for a smoke run)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--qlora", action="store_true", help="Load the base in 4-bit (bitsandbytes)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    data, skipped = [], 0
    for r in rows:
        ex = tokenize_example(tok, r["messages"], args.max_len)
        if ex is None:
            skipped += 1
        else:
            data.append(ex)
    if not data:
        raise SystemExit(f"No usable training examples out of {len(rows)} -- check the chat template output.")
    lens = sorted(len(d["input_ids"]) for d in data)
    print(f"{len(data)} training examples ({skipped} skipped as longer than {args.max_len} tokens); "
          f"median {lens[len(lens) // 2]} tokens, max {lens[-1]}")

    dtype = pick_dtype()
    print(f"compute dtype {dtype}")
    kw = {"torch_dtype": dtype}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)
    # One GPU: a ~2B model fits, and splitting it across Kaggle's 2x T4 only adds transfer time.
    model = AutoModelForCausalLM.from_pretrained(args.base, device_map={"": 0}, **kw)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if dtype == torch.float16:
        # fp16 AMP cannot unscale fp16 gradients, so the LoRA weights themselves train in fp32.
        for p_ in model.parameters():
            if p_.requires_grad:
                p_.data = p_.data.float()
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(data) / (args.batch * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    targs = TrainingArguments(**supported_kwargs(TrainingArguments, dict(
        output_dir=args.out, per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, round(0.05 * total_steps)), logging_steps=10,
        save_strategy="epoch", save_total_limit=2, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        report_to=[], seed=args.seed, group_by_length=True,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch", remove_unused_columns=False)))
    print(f"~{steps_per_epoch} optimizer steps per epoch")
    trainer = Trainer(model=model, args=targs, train_dataset=data,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(Path(args.out) / "podtekst_train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"train": str(args.train), "examples": len(data)}, f, indent=2)
    print(f"Adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
