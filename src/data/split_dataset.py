from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def collect_images(data_dir):
    """
    Collect all preprocessed images and their class labels.
    """

    data_dir = Path(data_dir)

    rows = []

    for class_dir in sorted(data_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image_path in sorted(class_dir.rglob("*")):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                rows.append({
                    "image_path": str(image_path),
                    "class": class_name
                })

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser(
        description="Create stratified train/validation manifests."
    )

    parser.add_argument(
        "--data_dir",
        default="data/preprocessed/plantdoc_27class_224/train"
    )

    parser.add_argument(
        "--output_dir",
        default="data/manifests"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("PlantDoc Train / Validation Split")
    print("=" * 70)

    print(f"Dataset : {data_dir}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Seed    : {args.seed}")

    # ---------------------------------------------------------
    # Collect images
    # ---------------------------------------------------------

    df = collect_images(data_dir)

    if df.empty:
        raise RuntimeError(
            "No images found."
        )

    print()
    print(f"Total images: {len(df)}")
    print(
        f"Number of classes: "
        f"{df['class'].nunique()}"
    )

    # ---------------------------------------------------------
    # Display original distribution
    # ---------------------------------------------------------

    print()
    print("Original class distribution:")

    print(
        df["class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # Stratified split
    # ---------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df["class"]
    )

    # ---------------------------------------------------------
    # Add split information
    # ---------------------------------------------------------

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"

    # Sort for reproducibility/readability
    train_df = train_df.sort_values(
        ["class", "image_path"]
    ).reset_index(drop=True)

    val_df = val_df.sort_values(
        ["class", "image_path"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save manifests
    # ---------------------------------------------------------

    train_path = (
        output_dir / "train_manifest.csv"
    )

    val_path = (
        output_dir / "val_manifest.csv"
    )

    combined_path = (
        output_dir / "train_val_manifest.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    combined_df = pd.concat(
        [train_df, val_df],
        ignore_index=True
    )

    combined_df.to_csv(
        combined_path,
        index=False
    )

    # ---------------------------------------------------------
    # Validation checks
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(
        f"Training images  : {len(train_df)}"
    )

    print(
        f"Validation images: {len(val_df)}"
    )

    print(
        f"Total             : "
        f"{len(train_df) + len(val_df)}"
    )

    print()
    print("Training distribution:")

    print(
        train_df["class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Validation distribution:")

    print(
        val_df["class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # Verify all classes exist in both splits
    # ---------------------------------------------------------

    train_classes = set(
        train_df["class"]
    )

    val_classes = set(
        val_df["class"]
    )

    all_classes = set(
        df["class"]
    )

    missing_train = (
        all_classes - train_classes
    )

    missing_val = (
        all_classes - val_classes
    )

    if missing_train:
        raise RuntimeError(
            f"Classes missing from training split: "
            f"{missing_train}"
        )

    if missing_val:
        raise RuntimeError(
            f"Classes missing from validation split: "
            f"{missing_val}"
        )

    print()
    print(
        "Classes in training   : "
        f"{len(train_classes)}"
    )

    print(
        "Classes in validation : "
        f"{len(val_classes)}"
    )

    # ---------------------------------------------------------
    # Check overlap
    # ---------------------------------------------------------

    train_paths = set(
        train_df["image_path"]
    )

    val_paths = set(
        val_df["image_path"]
    )

    overlap = train_paths & val_paths

    if overlap:
        raise RuntimeError(
            f"Data leakage detected! "
            f"{len(overlap)} images appear in both splits."
        )

    print()
    print(
        "Train/validation overlap: 0"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(
        f"Train manifest: {train_path}"
    )

    print(
        f"Val manifest  : {val_path}"
    )

    print(
        f"Combined       : {combined_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()