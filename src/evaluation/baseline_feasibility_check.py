"""
Milestone-1 quick feasibility probe.

This is NOT the final model. torch/torchvision could not be installed in
this sandboxed research session in reasonable time (no GPU present, and
the sandbox's package mirror pulls full CUDA wheels even for a CPU
install). To still get real, non-fabricated evidence that visual disease
signal is separable in the chosen data, this script trains a classical
color/texture-histogram + Random Forest classifier using only
scikit-learn/OpenCV (already available), on the official PlantDoc
train/test split (no leakage — the split is the one released by the
dataset authors).

Interpretation:
- This is a WEAK baseline. A MobileNetV3/EfficientNet-Lite CNN (the
  actual milestone target, see src/models/) is expected to substantially
  outperform it.
- If even this weak baseline clearly beats random/majority-class
  guessing, that is meaningful evidence of feasibility.
- If it performs at chance level, that's a signal to look more closely
  at the data before committing to a CNN.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

IMG_EXT = {".jpg", ".jpeg", ".png"}


def extract_features(img_path, size=128):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    feats = []
    for channel in cv2.split(hsv):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        feats.extend(hist.flatten())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # texture / sharpness proxy
    feats.append(lap_var)
    feats.append(gray.mean())
    feats.append(gray.std())
    return np.array(feats, dtype=np.float32)


def load_split(split_dir, classes):
    X, y, paths = [], [], []
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() not in IMG_EXT:
                continue
            fpath = os.path.join(cls_dir, fname)
            feat = extract_features(fpath)
            if feat is not None:
                X.append(feat)
                y.append(cls)
                paths.append(fpath)
    return np.array(X), np.array(y), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out_md", default="reports/baseline_feasibility_result.md")
    ap.add_argument("--min_test_per_class", type=int, default=3)
    args = ap.parse_args()

    X_train, y_train, _ = load_split(args.train_dir, args.classes)
    X_test, y_test, _ = load_split(args.test_dir, args.classes)

    clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, zero_division=0)

    majority_class = pd.Series(y_train).value_counts().idxmax()
    majority_baseline_acc = (y_test == majority_class).mean()

    lines = [
        "# Milestone 1 — Quick Classical-CV Feasibility Probe",
        "",
        "**This is a weak, classical-features baseline (HSV color histograms + Laplacian "
        "texture variance + Random Forest) — NOT the target CNN.** It exists only to check, "
        "with a real experiment rather than an assumption, whether the chosen classes are "
        "visually separable at all in this dataset.",
        "",
        f"- Train images used: {len(X_train)}",
        f"- Test images used: {len(X_test)}",
        f"- Classes: {args.classes}",
        f"- Train/test split: official PlantDoc author-released split (no custom leakage risk)",
        "",
        f"- **Majority-class baseline accuracy: {majority_baseline_acc:.3f}**",
        f"- **Classical-features RF accuracy: {acc:.3f}**",
        f"- **Classical-features RF macro-F1: {macro_f1:.3f}**",
        "",
        "## Per-class report",
        "```",
        report,
        "```",
        "",
        "## Interpretation",
        f"- The Random Forest baseline {'beats' if acc > majority_baseline_acc else 'does NOT clearly beat'} "
        f"the majority-class baseline ({acc:.3f} vs {majority_baseline_acc:.3f}), which is the minimum bar "
        "for 'there is learnable visual signal in this data.'",
        "- A lightweight CNN (MobileNetV3-Small / EfficientNet-Lite0) with ImageNet-pretrained weights "
        "is expected to substantially exceed this classical baseline, since it can learn lesion shape "
        "and spatial texture patterns that color histograms discard.",
        "- Small per-class test counts (single digits for several classes) mean these numbers have wide "
        "confidence intervals and should not be read as final performance figures.",
    ]

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
