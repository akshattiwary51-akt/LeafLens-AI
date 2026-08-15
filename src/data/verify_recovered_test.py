import hashlib
from pathlib import Path
import pandas as pd


TRAIN_DIR = Path("data/raw/PlantDoc-Dataset/train")
RECOVERED_DIR = Path("data/recovered_test")

OUTPUT_CSV = Path(
    "reports/recovered_test_duplicates.csv"
)

OUTPUT_MD = Path(
    "reports/recovered_test_duplicates.md"
)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def collect_images(directory):
    images = []

    for path in directory.rglob("*"):
        if path.is_file():
            images.append(path)

    return images


def main():

    print("=" * 70)
    print("VERIFY RECOVERED TEST IMAGES")
    print("=" * 70)

    train_images = collect_images(TRAIN_DIR)
    recovered_images = collect_images(RECOVERED_DIR)

    print(f"Training images       : {len(train_images)}")
    print(f"Recovered test images : {len(recovered_images)}")
    print()

    # ---------------------------------------------------------
    # Hash training images
    # ---------------------------------------------------------

    print("Hashing training images...")

    train_hashes = {}

    for i, path in enumerate(train_images, 1):

        digest = sha256_file(path)

        train_hashes.setdefault(
            digest,
            []
        ).append(path)

        if i % 200 == 0 or i == len(train_images):
            print(f"  {i}/{len(train_images)}")

    # ---------------------------------------------------------
    # Compare recovered files
    # ---------------------------------------------------------

    matches = []

    print()
    print("Checking recovered test images...")

    for path in recovered_images:

        digest = sha256_file(path)

        if digest in train_hashes:

            test_class = path.parent.name

            for train_path in train_hashes[digest]:

                train_class = train_path.parent.name

                status = (
                    "SAME_CLASS"
                    if train_class == test_class
                    else "LABEL_CONFLICT"
                )

                matches.append({
                    "train_path": str(train_path),
                    "train_class": train_class,
                    "recovered_test_path": str(path),
                    "test_class": test_class,
                    "status": status,
                    "sha256": digest
                })

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    columns = [
        "train_path",
        "train_class",
        "recovered_test_path",
        "test_class",
        "status",
        "sha256"
    ]

    df = pd.DataFrame(
        matches,
        columns=columns
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ---------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------

    same_class = 0
    conflicts = 0

    if not df.empty:

        same_class = int(
            (df["status"] == "SAME_CLASS").sum()
        )

        conflicts = int(
            (df["status"] == "LABEL_CONFLICT").sum()
        )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Recovered Test Image Duplicate Verification\n\n"
        )

        f.write(
            f"- Training images scanned: **{len(train_images)}**\n"
        )

        f.write(
            f"- Recovered test images scanned: "
            f"**{len(recovered_images)}**\n"
        )

        f.write(
            f"- Exact SHA-256 matches: **{len(df)}**\n"
        )

        f.write(
            f"- Same-class matches: **{same_class}**\n"
        )

        f.write(
            f"- Label conflicts: **{conflicts}**\n\n"
        )

        if df.empty:

            f.write(
                "No exact duplicates were found between the "
                "recovered test images and the training set.\n"
            )

        else:

            f.write("## Matches\n\n")

            f.write(
                "| Train class | Test class | Status | "
                "Train file | Recovered test file |\n"
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
                    f"| `{row['recovered_test_path']}` |\n"
                )

    # ---------------------------------------------------------
    # Console result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Exact matches   : {len(df)}")
    print(f"Same class      : {same_class}")
    print(f"Label conflicts : {conflicts}")

    print()
    print(f"CSV: {OUTPUT_CSV}")
    print(f"MD : {OUTPUT_MD}")

    print("=" * 70)


if __name__ == "__main__":
    main()