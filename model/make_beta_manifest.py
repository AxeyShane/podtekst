"""Write manifest.json for the private beta model repo: tier, file, version, size and SHA-256 per model.
The Podtekst β app reads it to pick and verify its download.

    python model/make_beta_manifest.py --version rua-q3 --out D:\\podtekst-mm\\beta\\manifest.json ^
        D:\\podtekst-mm\\podtekst-rua-q3-int4.litertlm D:\\podtekst-mm\\convert-int8-log\\podtekst-rua-q3-int8.litertlm
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def tier_of(name: str) -> str:
    """The 1.7B model for 6 GB phones is named *-small.litertlm (it is 4-bit too)."""
    m = re.search(r"-(small|int4|int8)\.litertlm$", name)
    if not m:
        raise ValueError(f"cannot tell small/int4/int8 from the file name: {name}")
    return m.group(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(paths, version: str) -> dict:
    models = []
    for p in map(Path, paths):
        models.append({"tier": tier_of(p.name), "file": p.name, "version": version,
                       "bytes": p.stat().st_size, "sha256": sha256(p)})
    tiers = [m["tier"] for m in models]
    if len(set(tiers)) != len(tiers):
        raise ValueError(f"more than one file for a tier: {tiers}")
    return {"models": models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--version", required=True, help="Model version shown in the app, e.g. rua-q3")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = build(args.files, args.version)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for m in manifest["models"]:
        print(f'{m["tier"]}: {m["file"]} {m["bytes"]:,} bytes sha256 {m["sha256"]}')


if __name__ == "__main__":
    main()
