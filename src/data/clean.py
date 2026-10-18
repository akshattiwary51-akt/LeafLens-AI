"""
Cleaning step — Milestone 1 §8/§9.

Consumes the CSV produced by src/data/audit.py and produces a cleaned
manifest: drops corrupt files, flags (does not silently delete) exact
duplicates and cross-class duplicate conflicts for manual review, since
automatically picking a "winner" between two conflicting labels for the
same image is a labeling decision a human should make, not a script.
"""
import argparse
import os

import pandas as pd


def clean(audit_csv: str, out_manifest: str, out_review_csv: str):
    df = pd.read_csv(audit_csv)
    n_before = len(df)

    corrupt = df[df["corrupt"] == True]  # noqa: E712
    valid = df[df["corrupt"] == False].copy()  # noqa: E712

    # find phash collisions that span more than one class -> label conflict
    conflict_rows = []
    for h, group in valid.groupby("phash"):
        classes = group["class"].unique()
        if len(classes) > 1:
            conflict_rows.append(group.assign(conflict_type="cross_class_duplicate"))

    conflicts = pd.concat(conflict_rows) if conflict_rows else pd.DataFrame()

    # within-class exact duplicates: keep first occurrence, drop the rest
    valid["is_dup_to_drop"] = valid.duplicated(subset=["class", "phash"], keep="first")
    # never auto-drop a cross-class conflict — that needs human review
    if not conflicts.empty:
        conflict_keys = set(zip(conflicts["class"], conflicts["filename"]))
        valid["in_conflict"] = valid.apply(lambda r: (r["class"], r["filename"]) in conflict_keys, axis=1)
        valid.loc[valid["in_conflict"], "is_dup_to_drop"] = False
    else:
        valid["in_conflict"] = False

    cleaned = valid[~valid["is_dup_to_drop"]]

    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    cleaned.to_csv(out_manifest, index=False)
    if not conflicts.empty:
        conflicts.to_csv(out_review_csv, index=False)

    print(f"Input rows: {n_before}")
    print(f"Corrupt (excluded): {len(corrupt)}")
    print(f"Within-class exact duplicates dropped: {int(valid['is_dup_to_drop'].sum())}")
    print(f"Cross-class label conflicts flagged for MANUAL REVIEW (not dropped): {len(conflicts)}")
    print(f"Final cleaned manifest rows: {len(cleaned)} -> {out_manifest}")
    if not conflicts.empty:
        print(f"Review file: {out_review_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_csv", required=True)
    ap.add_argument("--out_manifest", default="data/interim/cleaned_manifest.csv")
    ap.add_argument("--out_review_csv", default="reports/label_conflicts_for_review.csv")
    args = ap.parse_args()
    clean(args.audit_csv, args.out_manifest, args.out_review_csv)
