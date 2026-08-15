import os
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
import imagehash


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(data_dir):
    """
    Collect all images recursively.

    Returns:
        list of tuples: (image_path, class_name)
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    images = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((file_path, class_name))

    return images


def compute_hash(image_path):
    """
    Compute perceptual hash using aHash.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return str(imagehash.average_hash(img))
    except Exception as e:
        print(f"[WARNING] Could not read {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Detect cross-split duplicate/near-duplicate images."
    )

    parser.add_argument(
        "--train_dir",
        required=True,
        help="Training dataset directory"
    )

    parser.add_argument(
        "--test_dir",
        required=True,
        help="Test dataset directory"
    )

    parser.add_argument(
        "--out_csv",
        default="reports/cross_split_duplicates.csv",
        help="Output CSV path"
    )

    parser.add_argument(
        "--out_md",
        default="reports/cross_split_duplicates.md",
        help="Output Markdown report path"
    )

    parser.add_argument(
        "--max_distance",
        type=int,
        default=0,
        help="Maximum Hamming distance between hashes. 0 = exact aHash match."
    )

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    print("=" * 70)
    print("Cross-Split Duplicate Detection")
    print("=" * 70)

    print(f"Train directory: {train_dir}")
    print(f"Test directory : {test_dir}")
    print(f"Hash distance  : {args.max_distance}")
    print()

    # ---------------------------------------------------------
    # Collect images
    # ---------------------------------------------------------

    print("[1/4] Collecting training images...")
    train_images = collect_images(train_dir)

    print(f"       Found {len(train_images)} training images.")

    print("[2/4] Collecting test images...")
    test_images = collect_images(test_dir)

    print(f"       Found {len(test_images)} test images.")

    # ---------------------------------------------------------
    # Compute training hashes
    # ---------------------------------------------------------

    print()
    print("[3/4] Computing training image hashes...")

    train_hashes = []

    for index, (path, class_name) in enumerate(train_images, start=1):

        if index % 100 == 0 or index == len(train_images):
            print(f"       Processed {index}/{len(train_images)}")

        hash_value = compute_hash(path)

        if hash_value is not None:
            train_hashes.append({
                "path": str(path),
                "class": class_name,
                "hash": hash_value
            })

    # ---------------------------------------------------------
    # Compute test hashes
    # ---------------------------------------------------------

    print()
    print("[4/4] Computing test image hashes...")

    test_hashes = []

    for index, (path, class_name) in enumerate(test_images, start=1):

        if index % 50 == 0 or index == len(test_images):
            print(f"       Processed {index}/{len(test_images)}")

        hash_value = compute_hash(path)

        if hash_value is not None:
            test_hashes.append({
                "path": str(path),
                "class": class_name,
                "hash": hash_value
            })

    # ---------------------------------------------------------
    # Compare train vs test
    # ---------------------------------------------------------

    print()
    print("Comparing train and test hashes...")

    matches = []

    for test_item in test_hashes:

        test_hash = imagehash.hex_to_hash(test_item["hash"])

        for train_item in train_hashes:

            train_hash = imagehash.hex_to_hash(train_item["hash"])

            distance = test_hash - train_hash

            if distance <= args.max_distance:

                label_status = (
                    "SAME_CLASS"
                    if test_item["class"] == train_item["class"]
                    else "LABEL_CONFLICT"
                )

                matches.append({
                    "train_path": train_item["path"],
                    "train_class": train_item["class"],
                    "test_path": test_item["path"],
                    "test_class": test_item["class"],
                    "hash_distance": distance,
                    "label_status": label_status
                })

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    columns = [
        "train_path",
        "train_class",
        "test_path",
        "test_class",
        "hash_distance",
        "label_status"
    ]

    df = pd.DataFrame(matches, columns=columns)

    if not df.empty:
        df = df.sort_values(
            by=["label_status", "hash_distance", "train_class"]
        )

    df.to_csv(args.out_csv, index=False)

    # ---------------------------------------------------------
    # Create Markdown report
    # ---------------------------------------------------------

    same_class_count = 0
    conflict_count = 0

    if not df.empty:
        same_class_count = int(
            (df["label_status"] == "SAME_CLASS").sum()
        )

        conflict_count = int(
            (df["label_status"] == "LABEL_CONFLICT").sum()
        )

    unique_train_matches = (
        df["train_path"].nunique()
        if not df.empty else 0
    )

    unique_test_matches = (
        df["test_path"].nunique()
        if not df.empty else 0
    )

    with open(args.out_md, "w", encoding="utf-8") as f:

        f.write("# Cross-Split Duplicate Audit\n\n")

        f.write(f"- Training images scanned: **{len(train_images)}**\n")
        f.write(f"- Test images scanned: **{len(test_images)}**\n")
        f.write(
            f"- Hash method: **aHash**\n"
        )
        f.write(
            f"- Maximum hash distance: **{args.max_distance}**\n"
        )
        f.write(
            f"- Cross-split matching pairs: **{len(df)}**\n"
        )
        f.write(
            f"- Unique training images involved: **{unique_train_matches}**\n"
        )
        f.write(
            f"- Unique test images involved: **{unique_test_matches}**\n"
        )
        f.write(
            f"- Same-class matches: **{same_class_count}**\n"
        )
        f.write(
            f"- Label conflicts: **{conflict_count}**\n\n"
        )

        f.write("## Interpretation\n\n")

        if df.empty:
            f.write(
                "No cross-split matches were detected at the selected "
                "hash-distance threshold.\n\n"
            )
        else:
            if conflict_count > 0:
                f.write(
                    "⚠️ Label conflicts were detected. These should be "
                    "manually inspected before model training.\n\n"
                )

            if same_class_count > 0:
                f.write(
                    "⚠️ Same-class train/test duplicates were detected. "
                    "These may cause data leakage and should be reviewed.\n\n"
                )

        f.write("## Matching pairs\n\n")

        if df.empty:
            f.write("No matching pairs found.\n")
        else:

            f.write(
                "| Train class | Test class | Hash distance | "
                "Status | Train file | Test file |\n"
            )

            f.write(
                "|---|---|---:|---|---|---|\n"
            )

            for _, row in df.iterrows():

                f.write(
                    f"| {row['train_class']} "
                    f"| {row['test_class']} "
                    f"| {row['hash_distance']} "
                    f"| {row['label_status']} "
                    f"| `{row['train_path']}` "
                    f"| `{row['test_path']}` |\n"
                )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Training images : {len(train_images)}")
    print(f"Test images     : {len(test_images)}")
    print(f"Matches         : {len(df)}")
    print(f"Same class      : {same_class_count}")
    print(f"Label conflicts : {conflict_count}")

    print()
    print(f"CSV report      : {args.out_csv}")
    print(f"Markdown report : {args.out_md}")
    print("=" * 70)


if __name__ == "__main__":
    main()