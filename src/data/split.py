"""
Reproducible train/validation/test split — Milestone 1 §10.

Rules implemented:
- Fixed random seed (default 42) for reproducibility.
- Stratified by class where possible.
- If a grouping column (e.g. plant_id / session_id) is present in the
  manifest, splits by GROUP so the same plant/session never appears in
  more than one split (prevents leakage). PlantDoc/PlantVillage do not
  ship this metadata, so for those the limitation is reported explicitly
  rather than silently assumed away.
- Prefers an INDEPENDENT field-image source as the test set over a random
  split of the same pool, when one is available (spec §10). For PlantDoc
  we use the dataset's own official train/test folders directly for this
  reason instead of re-splitting them (see notebooks/01_dataset_audit.ipynb).
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_manifest(manifest_csv: str, out_dir: str, group_col: str = None,
                    seed: int = 42, train_frac=0.70, val_frac=0.15):
    df = pd.read_csv(manifest_csv)
    os.makedirs(out_dir, exist_ok=True)

    limitation_note = None
    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
        train_g, temp_g = train_test_split(groups, train_size=train_frac, random_state=seed)
        val_g, test_g = train_test_split(
            temp_g, train_size=val_frac / (1 - train_frac), random_state=seed
        )
        train_df = df[df[group_col].isin(train_g)]
        val_df = df[df[group_col].isin(val_g)]
        test_df = df[df[group_col].isin(test_g)]
    else:
        limitation_note = (
            f"No group column ('{group_col}') found in manifest — this dataset does not ship "
            "plant/session metadata, so this split is stratified by class only. Near-duplicate "
            "leakage across splits is possible; run src/data/audit.py's phash output to spot-check "
            "before trusting held-out numbers."
        )
        # Stratification requires >=2 members per class in each resulting split.
        # Real datasets (e.g. PlantDoc's "Tomato two spotted spider mites leaf",
        # which has only 2 images total) can violate this — fall back to
        # unstratified split for those cases rather than crashing, and say so.
        class_counts = df["class"].value_counts() if "class" in df.columns else None
        can_stratify = class_counts is not None and class_counts.min() >= 3
        stratify_col = df["class"] if can_stratify else None
        if not can_stratify and class_counts is not None:
            too_small = class_counts[class_counts < 3].to_dict()
            note2 = f"Stratified split disabled — classes with <3 samples exist: {too_small}"
            limitation_note = (limitation_note + " | " + note2) if limitation_note else note2

        train_df, temp_df = train_test_split(
            df, train_size=train_frac, random_state=seed, stratify=stratify_col
        )
        strat2 = None
        if can_stratify and "class" in temp_df.columns:
            temp_counts = temp_df["class"].value_counts()
            if temp_counts.min() >= 2:
                strat2 = temp_df["class"]
        val_df, test_df = train_test_split(
            temp_df, train_size=val_frac / (1 - train_frac), random_state=seed, stratify=strat2
        )

    train_df.to_csv(os.path.join(out_dir, "train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_manifest.csv"), index=False)

    summary = {
        "seed": seed,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "group_based": bool(group_col and group_col in df.columns),
        "limitation": limitation_note,
    }
    with open(os.path.join(out_dir, "split_manifest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--group_col", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_manifest(args.manifest_csv, args.out_dir, args.group_col, args.seed)
