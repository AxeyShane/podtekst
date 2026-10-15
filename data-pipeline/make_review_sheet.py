"""
Native-speaker yes/no sheet ("так пишут?") in the same layout as the earlier phrase checks: the earlier
.xlsx is used as a template (styles, yellow dropdown column, frozen header), only the rows are replaced.

    python make_review_sheet.py --seeds seeds_register1.txt --n 40 \
        --template D:/podtekst-mm/review/Podtekst_proverka_slang.xlsx \
        --out D:/podtekst-mm/review/Podtekst_proverka_mat.xlsx [--translations stage_a_register1.clean.jsonl]

Picks --n random Russian seeds (fixed --random-seed). The hidden _id sheet maps each sheet row to the
seed's line number in --seeds, as in the earlier sheets. Stdlib only (zipfile), no openpyxl.
"""
import argparse
import json
import os
import random
import re
import zipfile
from xml.sax.saxutils import escape

NOTE = ("Для каждой фразы выбери в жёлтом столбце: «да» — так реально пишут; «нет» — так никто не пишет; "
        "«исправить» — смысл нормальный, но сказали бы иначе (тогда напиши свой вариант рядом). В этих фразах "
        "много мата и сленга — так и задумано, оцениваем только, звучит ли это естественно. Перевод на английский "
        "проверять не обязательно — если заметишь, что смысл неверный, напиши в комментарии.")


def cell(ref, style, text=None, num=None):
    if num is not None:
        return f'<c r="{ref}" s="{style}" t="n"><v>{num}</v></c>'
    if text is None:
        return f'<c r="{ref}" s="{style}" t="n"></c>'
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--translations", default=None, help="Stage A JSONL; first candidate's translation is shown")
    ap.add_argument("--random-seed", type=int, default=13)
    args = ap.parse_args()
    if os.path.exists(args.out):  # review sheets are hand-edited; never overwrite one
        raise SystemExit(f"{args.out} exists; pick a new name")

    lines = [l.strip() for l in open(args.seeds, encoding="utf-8")]
    ru = [(i + 1, s) for i, s in enumerate(lines) if s and re.search(r"[а-яё]", s, re.I)]
    picked = random.Random(args.random_seed).sample(ru, args.n)
    trans = {}
    if args.translations:
        for l in open(args.translations, encoding="utf-8"):
            r = json.loads(l)
            trans.setdefault(r["source_text"], r.get("translation", ""))

    first, last = 5, 4 + args.n
    rows = []
    for k, (_, s) in enumerate(picked):
        r = first + k
        rows.append(f'<row r="{r}">' + cell(f"A{r}", 4, num=k + 1) + cell(f"B{r}", 4, s) + cell(f"C{r}", 5)
                    + cell(f"D{r}", 5) + cell(f"E{r}", 6, trans.get(s) or None) + cell(f"F{r}", 4) + "</row>")
    total = last + 2
    rows.append(f'<row r="{total}">' + cell(f"B{total}", 7, "Проверено:")
                + f'<c r="C{total}"><f>COUNTA(C{first}:C{last})&amp;" из {args.n}"</f><v></v></c></row>')

    with zipfile.ZipFile(args.template) as z:
        files = {n: z.read(n) for n in z.namelist()}
    x = files["xl/worksheets/sheet1.xml"].decode()
    head = re.search(r"^.*?<row r=\"4\".*?</row>", x, re.S).group(0)
    head = re.sub(r'(<row r="2"[^>]*>.*?<t>).*?(</t>)', lambda m: m.group(1) + escape(NOTE) + m.group(2), head, flags=re.S)
    head = head.replace('<dimension ref="A1:F66"/>', f'<dimension ref="A1:F{total}"/>')
    tail = x[x.index("</sheetData>"):]
    sqref = " ".join(f"C{r}" for r in range(first, last + 1))
    tail = re.sub(r'(<dataValidation sqref=")[^"]*', lambda m: m.group(1) + sqref, tail)
    files["xl/worksheets/sheet1.xml"] = (head + "".join(rows) + tail).encode()
    ids = "".join(f'<row r="{k + 1}"><c r="A{k + 1}" t="n"><v>{k + 1}</v></c><c r="B{k + 1}" t="n"><v>{line}</v></c></row>'
                  for k, (line, _) in enumerate(picked))
    y = files["xl/worksheets/sheet2.xml"].decode()
    y = re.sub(r"<sheetData>.*</sheetData>", f"<sheetData>{ids}</sheetData>", y, flags=re.S)
    files["xl/worksheets/sheet2.xml"] = re.sub(r'<dimension ref="[^"]*"/>', f'<dimension ref="A1:B{args.n}"/>', y).encode()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)
    print(f"{args.n} Russian seeds -> {args.out} ({sum(1 for _, s in picked if trans.get(s))} with a translation)")


if __name__ == "__main__":
    main()
