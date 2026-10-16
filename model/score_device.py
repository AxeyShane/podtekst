"""Score Podtekst Lab results pulled from the phone with the same metrics as evaluate.py, so the
on-device (INT4) numbers line up with the Kaggle runs.

    adb pull /sdcard/Android/data/com.podtekst.lab/files/results/ D:\\podtekst-mm\\device-results
    python score_device.py --name rua-q3-int4-phone D:\\podtekst-mm\\device-results\\*.jsonl

Each results file is grouped by the probe file it came from (the middle part of
<model>__<probe>__<time>.jsonl): crude_probe -> crude slice, profanity_test -> profanity report,
anything else (sft_test) -> main metrics. Writes out/eval_<name>.json like evaluate.py.
"""
import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate import finish


def kind(path) -> str:
    name = Path(path).name
    if "crude_probe" in name:
        return "probe"
    if "profanity_test" in name:
        return "prof"
    return "test"


def model_name(path) -> str:
    return Path(path).name.split("__")[0]


def expand(patterns) -> list[Path]:
    """Expands globs here because Windows shells pass them through unexpanded."""
    paths = []
    for p in patterns:
        paths += [Path(x) for x in sorted(glob.glob(str(p)))] or ([Path(p)] if Path(p).is_file() else [])
    return paths


def load_results(paths) -> dict:
    groups = {"test": ([], []), "probe": ([], []), "prof": ([], [])}
    for path in paths:
        rows, replies = groups[kind(path)]
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    rows.append(item["row"])
                    replies.append(item.get("reply") or "")
    return groups


def score(patterns, name: str, out_dir=None) -> dict:
    paths = expand(patterns)
    if not paths:
        raise SystemExit(f"no results files match {patterns}")
    g = load_results(paths)
    models = sorted({model_name(p) for p in paths})
    args = SimpleNamespace(name=name, base="device:" + ",".join(models), adapter=None)
    return finish(args, *g["test"], *g["probe"], *g["prof"], out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Report name, e.g. rua-q3-int4-phone")
    ap.add_argument("results", nargs="+", help="Results .jsonl files or globs")
    args = ap.parse_args()
    score(args.results, args.name)


if __name__ == "__main__":
    main()
