"""Duplicate clustering over the preprocessed dataset (all splits at once).

Edges: identical source SHA-256 ("exact") or 64-bit dHash Hamming <= T
("near_strong" <=1, "near_weak" 2..T). Connected components = duplicate groups.
Human decisions in a review CSV (columns a,b,decision with decision in
{same, different}) override: "different" removes an edge, "same" adds one.

Outputs (default data/manifests_v2/): duplicate_pairs.csv, duplicate_groups.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image

from dataset_audit import dhash, popcount64

NEAR_T = 4


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def compute_pairs(df, hashes, near_t=NEAR_T):
    """df: index rows (output, source_sha256, ...). Returns list of (i, j, kind, dist)."""
    h = np.asarray(hashes, dtype=np.uint64)
    sha = df.source_sha256.values
    pairs = []
    for i in range(len(h) - 1):
        d = popcount64(h[i + 1:] ^ h[i])
        for off in np.nonzero(d <= near_t)[0]:
            j = i + 1 + int(off)
            if sha[i] == sha[j]:
                kind = "exact"
            else:
                kind = "near_strong" if d[off] <= 1 else "near_weak"
            pairs.append((i, j, kind, int(d[off])))
    # exact duplicates whose dHash differs cannot exist (same pixels), but be explicit
    return pairs


def build_groups(df, pairs, decisions=None):
    uf = UnionFind(len(df))
    key = {o: i for i, o in enumerate(df.output)}
    dec = {}
    for r in (decisions.itertuples() if decisions is not None else []):
        dec[frozenset((r.a, r.b))] = r.decision
    kept = []
    for i, j, kind, dist in pairs:
        d = dec.get(frozenset((df.output[i], df.output[j])))
        if d == "different":
            continue
        uf.union(i, j)
        kept.append((i, j, kind, dist))
    for k, d in dec.items():
        if d == "same":
            a, b = tuple(k)
            if a in key and b in key:
                uf.union(key[a], key[b])
    roots = [uf.find(i) for i in range(len(df))]
    out = df.copy()
    out["group_id"] = [f"g{r:05d}" for r in roots]
    sizes = out.groupby("group_id").output.transform("size")
    out["group_size"] = sizes
    out["group_classes"] = out.groupby("group_id").cls.transform(lambda s: "|".join(sorted(set(s))))
    out["group_splits"] = out.groupby("group_id").split.transform(lambda s: "|".join(sorted(set(s))))
    out["label_conflict"] = out.group_classes.str.contains(r"\|")
    out["crosses_splits"] = out.group_splits.str.contains(r"\|")
    return out, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/preprocessed/plantdoc_224_v2")
    ap.add_argument("--out_dir", default="data/manifests_v2")
    ap.add_argument("--review", default="data/manifests_v2/duplicate_review.csv")
    ap.add_argument("--near_threshold", type=int, default=NEAR_T)
    a = ap.parse_args()
    root = Path(a.data_dir)
    df = pd.read_csv(root / "preprocess_index.csv").reset_index(drop=True)
    hashes = []
    for o in df.output:
        with Image.open(root / o) as im:
            hashes.append(dhash(im.convert("RGB")))
    pairs = compute_pairs(df, hashes, a.near_threshold)
    dec = pd.read_csv(a.review) if Path(a.review).exists() else None
    groups, kept = build_groups(df, pairs, dec)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    prow = [dict(a=df.output[i], a_split=df.split[i], a_class=df.cls[i],
                 b=df.output[j], b_split=df.split[j], b_class=df.cls[j],
                 kind=k, hamming=d,
                 cross_split=df.split[i] != df.split[j],
                 cross_class=df.cls[i] != df.cls[j]) for i, j, k, d in pairs]
    pd.DataFrame(prow).to_csv(out / "duplicate_pairs.csv", index=False)
    groups.to_csv(out / "duplicate_groups.csv", index=False)
    multi = groups[groups.group_size > 1]
    print(f"images={len(df)} pairs={len(pairs)} multi-image groups={multi.group_id.nunique()} "
          f"cross-split groups={multi[multi.crosses_splits].group_id.nunique()} "
          f"label-conflict groups={multi[multi.label_conflict].group_id.nunique()}")


if __name__ == "__main__":
    main()
