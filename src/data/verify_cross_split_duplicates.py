import argparse
import hashlib
from pathlib import Path
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def collect_images(directory):
    directory = Path(directory)

    results = []

    for class_dir in sorted(directory.iterdir()):

        if not class_dir.is_dir():
            continue

        for path in class_dir.rglob("*"):

            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                results.append({
                    "path": str(path),
                    "class": class_dir.name
                })

    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        required=True
    )

    parser.add_argument(
        "--test_dir",
        required=True
    )

    parser.add_argument(
        "--out_csv",
        default="reports/exact_cross_split_duplicates.csv"
    )

    parser.add_argument(
        "--out_md",
        default="reports/exact_cross_split_duplicates.md"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EXACT CROSS-SPLIT DUPLICATE VERIFICATION")
    print("=" * 70)

    train_images = collect_images(args.train_dir)
    test_images = collect_images(args.test_dir)

    print(f"Training images: {len(train_images)}")
    print(f"Test images    : {len(test_images)}")
    print()

    # ---------------------------------------------------------
    # Hash training files
    # ---------------------------------------------------------

    print("Hashing training images...")

    train_hashes = {}

    for i, item in enumerate(train_images, 1):

        digest = sha256_file(item["path"])

        train_hashes.setdefault(digest, []).append(item)

        if i % 200 == 0 or i == len(train_images):
            print(f"  {i}/{len(train_images)}")

    # ---------------------------------------------------------
    # Compare test files
    # ---------------------------------------------------------

    print()
    print("Checking test images...")

    matches = []

    for i, test_item in enumerate(test_images, 1):

        digest = sha256_file(test_item["path"])

        if digest in train_hashes:

            for train_item in train_hashes[digest]:

                status = (
                    "SAME_CLASS"
                    if train_item["class"] == test_item["class"]
                    else "LABEL_CONFLICT"
                )

                matches.append({
                    "train_path": train_item["path"],
                    "train_class": train_item["class"],
                    "test_path": test_item["path"],
                    "test_class": test_item["class"],
                    "sha256": digest,
                    "status": status
                })

        if i % 50 == 0 or i == len(test_images):
            print(f"  {i}/{len(test_images)}")

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    columns = [
        "train_path",
        "train_class",
        "test_path",
        "test_class",
        "sha256",
        "status"
    ]

    df = pd.DataFrame(matches, columns=columns)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.out_csv, index=False)

    same_class = 0
    conflicts = 0

    if not df.empty:

        same_class = int(
            (df["status"] == "SAME_CLASS").sum()
        )

        conflicts = int(
            (df["status"] == "LABEL_CONFLICT").sum()
        )

    # ---------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------

    with open(args.out_md, "w", encoding="utf-8") as f:

        f.write("# Exact Cross-Split Duplicate Verification\n\n")

        f.write(f"- Training images scanned: **{len(train_images)}**\n")
        f.write(f"- Test images scanned: **{len(test_images)}**\n")
        f.write(f"- Exact SHA-256 matches: **{len(df)}**\n")
        f.write(f"- Same-class exact matches: **{same_class}**\n")
        f.write(f"- Label conflicts: **{conflicts}**\n\n")

        if df.empty:

            f.write(
                "No exact byte-for-byte duplicates were found between "
                "training and test sets.\n"
            )

        else:

            f.write("## Exact duplicates\n\n")

            f.write(
                "| Train class | Test class | Status | "
                "Train file | Test file |\n"
            )

            f.write(
                "|---|---|---|---|---|\n"
            )

            for _, row in df.iterrows():

                f.write(
                    f"| {row['train_class']} "
                    f"| {row['test_class']} "
                    f"| {row['status']} "
                    f"| `{row['train_path']}` "
                    f"| `{row['test_path']}` |\n"
                )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Exact matches   : {len(df)}")
    print(f"Same class      : {same_class}")
    print(f"Label conflicts : {conflicts}")

    print()
    print(f"CSV: {args.out_csv}")
    print(f"MD : {args.out_md}")

    print("=" * 70)


if __name__ == "__main__":
    main()