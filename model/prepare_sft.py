"""Turn the curated dataset split (data-pipeline/dataset/{train,test}.jsonl, written by
curate_dataset.py) into chat-format SFT files under model/data/.

Each output line: {"messages": [system, user, assistant], "row": <gold row>}.
model/data/ is git-ignored: rows can carry film/subtitle-derived text.

    python prepare_sft.py
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from podtekst_sft.prompt import build_messages

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE.parent / "data-pipeline" / "dataset"
KEEP = ("source_lang", "source_text", "translation", "has_subtext", "category", "nuance_note",
        "speaker_gender", "addressee_gender")


def convert(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        gold = {k: r[k] for k in KEEP if k in r}
        out.append({"messages": build_messages(r), "row": gold})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        src = args.in_dir / f"{split}.jsonl"
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        out = convert(rows)
        dst = args.out_dir / f"sft_{split}.jsonl"
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        cats = Counter(r["category"] for r in rows)
        print(f"{split}: {len(out)} rows -> {dst}  {dict(cats)}")


if __name__ == "__main__":
    main()
