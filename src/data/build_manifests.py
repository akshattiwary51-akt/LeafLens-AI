"""Build group-aware, leak-free manifests (data/manifests_v2).

Policy (all decisions are written to quarantine.csv / split_report.json):
  * test split = the official PlantDoc test split, FROZEN (hash in TEST_FROZEN.sha256)
  * classes with < --min_class_images images are excluded (recorded)
  * duplicate groups whose members carry conflicting class labels are
    quarantined until a human resolves them (never auto-labelled)
  * train-origin members of any group that touches the test set are
    quarantined as ``leaks_into_frozen_test`` (test stays untouched)
  * remaining groups are split train/val with StratifiedGroupKFold(seed) so a
    duplicate group can never straddle train and val
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build(groups, min_class_images=10, val_folds=5, seed=42):
    g = groups.copy()
    g["reason"] = ""
    cls_n = g[g.split == "train"].groupby("cls").size()
    small = set(cls_n[cls_n < min_class_images].index) | set(
        g[~g.cls.isin(cls_n.index)].cls)
    g.loc[g.cls.isin(small), "reason"] = "class_too_small"
    live = g.reason == ""
    g.loc[live & g.label_conflict, "reason"] = "label_conflict_pending_review"
    live = g.reason == ""
    touches_test = g.groupby("group_id").split.transform(lambda s: (s == "test").any())
    g.loc[live & touches_test & (g.split == "train"), "reason"] = "leaks_into_frozen_test"
    live = g.reason == ""
    g.loc[live & (g.split == "test"), "assigned"] = "test"
    tr = g[live & (g.split == "train")]
    sgkf = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(tr, tr.cls, tr.group_id))
    g.loc[tr.index, "assigned"] = "train"
    g.loc[tr.index[val_idx], "assigned"] = "val"
    # test images whose group has a label conflict are quarantined (never silently)
    return g, sorted(c for c in set(g.cls) - small)


def write(g, classes, out_dir, data_dir_rel, refreeze=False):
    out = Path(out_dir)
    cols = ["image_path", "class", "split", "group_id", "source_sha256"]
    act = g[g.assigned.isin(["train", "val", "test"])].copy()
    act["image_path"] = data_dir_rel + "/" + act.output
    act = act.rename(columns={"cls": "class"})
    act["split"] = act.assigned
    for s in ("train", "val", "test"):
        act[act.split == s][cols].sort_values("image_path").to_csv(
            out / f"{s}_manifest.csv", index=False)
    act[act.split.isin(["train", "val"])][cols].sort_values("image_path").to_csv(
        out / "train_val_manifest.csv", index=False)
    q = g[~g.assigned.isin(["train", "val", "test"])]
    q[["output", "cls", "split", "group_id", "reason", "group_classes", "group_splits"]].to_csv(
        out / "quarantine.csv", index=False)
    (out / "classes.json").write_text(json.dumps(
        {"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}, indent=2))
    h = hashlib.sha256((out / "test_manifest.csv").read_bytes()).hexdigest()
    frozen = out / "TEST_FROZEN.sha256"
    if frozen.exists() and frozen.read_text().strip() != h and not refreeze:
        sys.exit("test manifest differs from the frozen hash; pass --refreeze only if intended")
    frozen.write_text(h + "\n")
    counts = act.groupby(["class", "split"]).size().unstack(fill_value=0)
    counts.to_csv(out / "class_counts.csv")
    rep = {"n_classes": len(classes),
           "split_sizes": act.split.value_counts().to_dict(),
           "quarantined": q.reason.value_counts().to_dict(),
           "quarantined_images": int(len(q)),
           "min_test_per_class": int(counts.test.min()),
           "test_sha256": h}
    (out / "split_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="data/manifests_v2/duplicate_groups.csv")
    ap.add_argument("--out_dir", default="data/manifests_v2")
    ap.add_argument("--data_dir", default="data/preprocessed/plantdoc_224_v2")
    ap.add_argument("--min_class_images", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refreeze", action="store_true")
    a = ap.parse_args()
    g = pd.read_csv(a.groups)
    g, classes = build(g, a.min_class_images, seed=a.seed)
    print(json.dumps(write(g, classes, a.out_dir, a.data_dir, a.refreeze), indent=2))


if __name__ == "__main__":
    main()
