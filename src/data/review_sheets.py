"""Render contact sheets of duplicate pairs for human review.

Pairs rendered: every cross-class pair and every cross-split pair.
Fill data/manifests_v2/duplicate_review.csv (a,b,decision[same|different],reviewer,note).
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

CELL, PAD, LAB = 170, 6, 44


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/preprocessed/plantdoc_224_v2")
    ap.add_argument("--pairs", default="data/manifests_v2/duplicate_pairs.csv")
    ap.add_argument("--out", default="reports/dataset_audit/review")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args()
    p = pd.read_csv(a.pairs)
    p = p[p.cross_class | p.cross_split].reset_index(drop=True)
    p.insert(0, "pair_no", range(1, len(p) + 1))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out / "review_pairs.csv", index=False)
    per = a.cols * a.rows
    for s in range((len(p) + per - 1) // per):
        chunk = p.iloc[s * per:(s + 1) * per]
        w = a.cols * (2 * CELL + 3 * PAD)
        h = a.rows * (CELL + LAB + PAD)
        sheet = Image.new("RGB", (w, h), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk.itertuples()):
            x0 = (k % a.cols) * (2 * CELL + 3 * PAD) + PAD
            y0 = (k // a.cols) * (CELL + LAB + PAD)
            for n, name in enumerate((r.a, r.b)):
                im = Image.open(Path(a.data_dir) / name).convert("RGB").resize((CELL, CELL))
                sheet.paste(im, (x0 + n * (CELL + PAD), y0 + LAB))
            d.text((x0, y0 + 2), f"#{r.pair_no} {r.kind} h={r.hamming}", fill=(0, 0, 0))
            d.text((x0, y0 + 14), f"L:{r.a_class[:24]} [{r.a_split}]", fill=(160, 0, 0))
            d.text((x0, y0 + 26), f"R:{r.b_class[:24]} [{r.b_split}]", fill=(0, 0, 160))
        sheet.save(out / f"sheet_{s + 1}.png")
    print(f"{len(p)} pairs -> {(len(p)+per-1)//per} sheets in {out}")


if __name__ == "__main__":
    main()
