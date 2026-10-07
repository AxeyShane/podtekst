"""Run a model (base, or base + LoRA adapter) on model/data/sft_test.jsonl and score it.

    python evaluate.py --base D:\\podtekst-mm\\models\\qvikhr-3-1.7b --name baseline
    python evaluate.py --base D:\\podtekst-mm\\models\\qvikhr-3-1.7b --adapter out\\qvikhr-lora-v1 --name v1

    python evaluate.py --endpoint http://127.0.0.1:8080/v1 --name gemma4-e4b-gguf   # llama-server / any OpenAI-compatible server

Writes out/eval_<name>.json (metrics) and out/eval_<name>_preds.jsonl (every reply, for review).
Greedy decoding, so runs are comparable.
"""
import argparse
import json
from pathlib import Path

from podtekst_sft.metrics import summarize
from podtekst_sft.prompt import build_messages, parse_reply

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="Local model folder or HF id (transformers)")
    ap.add_argument("--endpoint", default=None,
                    help="OpenAI-compatible server instead of --base, e.g. llama-server at http://127.0.0.1:8080/v1")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", type=Path, default=HERE / "data" / "sft_test.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N rows (quick look)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--qlora", action="store_true", help="Load the base in 4-bit")
    args = ap.parse_args()
    if bool(args.base) == bool(args.endpoint):
        raise SystemExit("Pass exactly one of --base or --endpoint.")

    items = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[:args.limit]
    rows = [it["row"] for it in items]

    if args.endpoint:
        replies = [chat_endpoint(args.endpoint, build_messages(r, with_answer=False), args.max_new_tokens)
                   for r in progress(rows)]
    else:
        replies = generate_local(args, rows)
    finish(args, rows, replies)


def progress(rows):
    for i, r in enumerate(rows, 1):
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)
        yield r


def chat_endpoint(endpoint: str, messages: list[dict], max_tokens: int, post=None) -> str:
    """One greedy chat completion from an OpenAI-compatible server (llama-server, LM Studio, vLLM)."""
    if post is None:
        import requests
        post = requests.post
    r = post(endpoint.rstrip("/") + "/chat/completions",
             json={"messages": messages, "temperature": 0, "max_tokens": max_tokens}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def generate_local(args, rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter or args.base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.bfloat16}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base, device_map="auto", **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    prompts = [tok.apply_chat_template(build_messages(r, with_answer=False), add_generation_prompt=True,
                                       tokenize=False) for r in rows]
    replies = []
    for i in range(0, len(prompts), args.batch):
        enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for seq in out:
            replies.append(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"  {min(i + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    return replies


def finish(args, rows, replies):
    preds = [parse_reply(t) for t in replies]
    report = summarize(rows, preds) | {"name": args.name, "base": args.base or args.endpoint, "adapter": args.adapter}
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"eval_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / f"eval_{args.name}_preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r, t, p in zip(rows, replies, preds):
            f.write(json.dumps({"row": r, "reply": t, "parsed": p}, ensure_ascii=False) + "\n")
    d, c = report["detection"], report["category"]
    print(f"\n{args.name}: json_valid {report['json_valid_rate']}  chrF {report['chrf']}  "
          f"category acc {c['accuracy']}")
    print(f"  subtext precision {d['precision']}  recall {d['recall']}  false-positive rate {d['false_positive_rate']}")
    for cat in ("formality_shift", "idiom", "sarcasm", "emotional_subtext"):
        print(f"  {cat:18} F1 {c[cat]['f1']}  (n={c[cat]['support']})")
    print(f"-> out/eval_{args.name}.json")


if __name__ == "__main__":
    main()
