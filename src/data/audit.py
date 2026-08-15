"""
Dataset audit script — Milestone 1.

Walks a folder-per-class image dataset and produces verified statistics:
image counts, class imbalance, resolution distribution, corrupt files,
approximate duplicates (perceptual hash), and file-format breakdown.

This script only reports what it can measure directly from the files on
disk. It never guesses field/early-stage/severity labels — those are
recorded manually in reports/dataset_research.md only when a dataset's
official documentation states them explicitly.

Usage:
    python src/data/audit.py --data_dir data/raw/plantdoc/train \
        --dataset_name plantdoc_train \
        --out_csv reports/dataset_audit.csv \
        --out_md reports/dataset_audit.md \
        --figures_dir reports/figures
"""
import argparse
import hashlib
import os
from collections import defaultdict

import pandas as pd
from PIL import Image, ImageStat

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash(img: Image.Image, hash_size: int = 8) -> str:
    """Simple average-hash (aHash) for approximate duplicate detection.
    Not a cryptographic hash — used only to flag likely near-duplicates.

    Known limitation (caught by tests/test_audit.py): a perfectly flat,
    zero-variance image hashes to all-zero bits regardless of color, so
    two different solid-color images would collide. Real leaf photographs
    are never perfectly flat, so this has not produced false positives in
    the PlantDoc audit, but a production version should upgrade to a
    proper perceptual hash (e.g. imagehash.phash) with a similarity
    threshold rather than exact-match grouping.
    """
    img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)


def audit_dataset(data_dir: str, dataset_name: str):
    rows = []
    hash_to_files = defaultdict(list)
    class_dirs = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )

    if not class_dirs:
        raise SystemExit(f"No class subdirectories found under {data_dir}")

    for cls in class_dirs:
        cls_path = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_path) if os.path.splitext(f)[1].lower() in IMG_EXT]
        for fname in files:
            fpath = os.path.join(cls_path, fname)
            record = {
                "dataset": dataset_name,
                "class": cls,
                "filename": fname,
                "filesize_kb": None,
                "width": None,
                "height": None,
                "format": None,
                "mode": None,
                "mean_brightness": None,
                "corrupt": False,
                "phash": None,
            }
            try:
                record["filesize_kb"] = round(os.path.getsize(fpath) / 1024, 1)
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    record["width"], record["height"] = img.size
                    record["format"] = img.format or os.path.splitext(fname)[1].lstrip(".").upper()
                    record["mode"] = img.mode
                    stat = ImageStat.Stat(img)
                    record["mean_brightness"] = round(sum(stat.mean) / 3, 2)
                    h = phash(img)
                    record["phash"] = h
                    hash_to_files[h].append(f"{cls}/{fname}")
            except Exception as e:  # noqa: BLE001
                record["corrupt"] = True
                record["error"] = str(e)
            rows.append(record)

    df = pd.DataFrame(rows)

    # near-duplicate groups (identical aHash — exact matches only; a real
    # perceptual-similarity pass with a distance threshold should be run
    # before finalizing splits)
    dup_groups = {h: files for h, files in hash_to_files.items() if len(files) > 1}

    return df, dup_groups


def write_reports(df: pd.DataFrame, dup_groups: dict, dataset_name: str,
                   out_csv: str, out_md: str, figures_dir: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_total = len(df)
    n_corrupt = int(df["corrupt"].sum())
    valid = df[~df["corrupt"]]
    class_counts = valid["class"].value_counts()

    lines = []
    lines.append(f"# Dataset Audit — {dataset_name}\n")
    lines.append(f"- Total files scanned: **{n_total}**")
    lines.append(f"- Corrupt / unreadable files: **{n_corrupt}**")
    lines.append(f"- Valid images: **{len(valid)}**")
    lines.append(f"- Number of classes: **{valid['class'].nunique()}**")
    lines.append(f"- Exact-duplicate (aHash) groups found: **{len(dup_groups)}** "
                 f"(covering {sum(len(v) for v in dup_groups.values())} files)\n")

    lines.append("## Images per class\n")
    lines.append("| Class | Count |")
    lines.append("|---|---|")
    for cls, cnt in class_counts.items():
        lines.append(f"| {cls} | {cnt} |")

    if len(class_counts) > 0:
        imbalance_ratio = class_counts.max() / max(class_counts.min(), 1)
        lines.append(f"\n**Class imbalance ratio (max/min): {imbalance_ratio:.1f}x**\n")

    if len(valid) > 0:
        lines.append("## Resolution summary\n")
        lines.append(f"- Width  — min {valid['width'].min()}, median {int(valid['width'].median())}, max {valid['width'].max()}")
        lines.append(f"- Height — min {valid['height'].min()}, median {int(valid['height'].median())}, max {valid['height'].max()}")
        lines.append(f"\n## File formats\n")
        for fmt, cnt in valid["format"].value_counts().items():
            lines.append(f"- {fmt}: {cnt}")

    if dup_groups:
        lines.append("\n## Sample near-duplicate groups (first 5)\n")
        for i, (h, files) in enumerate(list(dup_groups.items())[:5]):
            lines.append(f"- Hash `{h}`: {files}")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, max(4, 0.3 * len(class_counts))))
        class_counts.sort_values().plot(kind="barh")
        plt.title(f"{dataset_name}: images per class")
        plt.xlabel("image count")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{dataset_name}_class_distribution.png"), dpi=120)
        plt.close()

        if len(valid) > 0:
            plt.figure(figsize=(6, 5))
            plt.scatter(valid["width"], valid["height"], s=5, alpha=0.4)
            plt.xlabel("width (px)")
            plt.ylabel("height (px)")
            plt.title(f"{dataset_name}: resolution distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"{dataset_name}_resolution_distribution.png"), dpi=120)
            plt.close()
    except ImportError:
        print("matplotlib not installed — skipping figures")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Figures in {figures_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--out_csv", default="reports/dataset_audit.csv")
    ap.add_argument("--out_md", default="reports/dataset_audit.md")
    ap.add_argument("--figures_dir", default="reports/figures")
    args = ap.parse_args()

    df, dup_groups = audit_dataset(args.data_dir, args.dataset_name)
    write_reports(df, dup_groups, args.dataset_name, args.out_csv, args.out_md, args.figures_dir)
