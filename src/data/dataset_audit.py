"""Reproducible manifest-level dataset audit (integrity, balance, leakage).

Usage: python src/data/dataset_audit.py [--manifests data/manifests]
                                         [--out reports/dataset_audit]
Reads train/val/test manifests, verifies every file, hashes images
(SHA-256 for exact duplicates, 64-bit dHash for near-duplicates) and reports
cross-split / cross-class collisions. Never modifies or deletes data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

from leaflens.manifest_paths import resolve_image_path

NEAR_DUP_HAMMING = 4  # dHash bits; <=4/64 treated as near-duplicate


def dhash(im, n=8):
    g = np.asarray(im.convert("L").resize((n + 1, n), Image.Resampling.LANCZOS), np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def popcount64(x):
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, np.uint8)
    for s in range(64):
        c += ((x >> np.uint64(s)) & np.uint64(1)).astype(np.uint8)
    return c


def audit(manifest_dir, root="."):
    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(Path(manifest_dir) / f"{split}_manifest.csv")
        df["split"] = split
        frames.append(df[["image_path", "class", "split"]])
    df = pd.concat(frames, ignore_index=True)
    rec = []
    for r in df.itertuples():
        p = resolve_image_path(r.image_path, root)
        row = dict(split=r.split, cls=r._2, path=r.image_path, exists=p.exists(),
                   readable=False, w=None, h=None, mode=None, sha=None, dh=None)
        if row["exists"]:
            try:
                raw = p.read_bytes()
                row["sha"] = hashlib.sha256(raw).hexdigest()
                with Image.open(p) as im:
                    row["w"], row["h"], row["mode"] = im.width, im.height, im.mode
                    im = im.convert("RGB")
                    row["dh"] = dhash(im)
                row["readable"] = True
            except Exception:
                pass
        rec.append(row)
    d = pd.DataFrame(rec)
    ok = d[d.readable].reset_index(drop=True)

    out = {"images_listed": len(d), "missing_files": int((~d.exists).sum()),
           "unreadable": int((d.exists & ~d.readable).sum()),
           "missing_by_split": d[~d.exists].split.value_counts().to_dict(),
           "n_classes": int(d.cls.nunique()),
           "sizes": {f"{w}x{h}": int(c) for (w, h), c in ok.groupby(["w", "h"]).size().items()},
           "modes": ok["mode"].value_counts().to_dict()}
    counts = d.groupby(["cls", "split"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    out["imbalance_ratio_max_over_min_train"] = round(counts.train.max() / counts.train.min(), 2)
    out["min_test_per_class"] = int(counts.test.min())
    out["max_test_per_class"] = int(counts.test.max())

    # exact duplicates
    g = ok.groupby("sha")
    exact = [grp for _, grp in g if len(grp) > 1]
    out["exact_duplicate_groups"] = len(exact)
    out["exact_cross_split_groups"] = sum(grp.split.nunique() > 1 for grp in exact)
    out["exact_cross_class_groups"] = sum(grp.cls.nunique() > 1 for grp in exact)

    # near duplicates (pairwise dHash)
    h = ok.dh.astype(np.uint64).values
    pairs = []
    for i in range(len(h)):
        dist = popcount64(h[i + 1:] ^ h[i])
        for j in np.nonzero(dist <= NEAR_DUP_HAMMING)[0]:
            pairs.append((i, i + 1 + int(j), int(dist[j])))
    cs = [p for p in pairs if ok.split[p[0]] != ok.split[p[1]]]
    cc = [p for p in pairs if ok.cls[p[0]] != ok.cls[p[1]]]
    out.update(near_duplicate_pairs=len(pairs), near_dup_cross_split_pairs=len(cs),
               near_dup_cross_class_pairs=len(cc),
               near_dup_pairs_touching_test=sum("test" in (ok.split[a], ok.split[b]) and ok.split[a] != ok.split[b] for a, b, _ in pairs))
    pair_rows = [dict(a=ok.path[a], a_split=ok.split[a], a_class=ok.cls[a],
                      b=ok.path[b], b_split=ok.split[b], b_class=ok.cls[b], hamming=dd)
                 for a, b, dd in pairs]
    return out, counts, pd.DataFrame(pair_rows), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--out", default="reports/dataset_audit")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    out, counts, pairs, d = audit(a.manifests, a.root)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(out, indent=2))
    counts.to_csv(o / "class_counts.csv")
    pairs.to_csv(o / "near_duplicate_pairs.csv", index=False)
    d[~d.exists].to_csv(o / "missing_files.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
