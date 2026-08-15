import shutil
from pathlib import Path
import pandas as pd


TRAIN_DIR = Path("data/raw/PlantDoc-Dataset/train")
TEST_DIR = Path("data/raw/PlantDoc-Dataset/test")

OUTPUT_DIR = Path("data/processed/plantdoc_27class")

QUARANTINE_DIR = Path(
    "data/quarantine/cross_split_duplicates"
)

DUPLICATE_REPORT = Path(
    "reports/exact_cross_split_duplicates.csv"
)

EXCLUDED_CLASS = "Tomato two spotted spider mites leaf"


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def normalize_path(path):
    """
    Convert Windows/Unix paths to a normalized string.
    """
    return str(Path(str(path).replace("\\", "/")))


def main():

    print("=" * 70)
    print("PlantDoc Cross-Split Cleanup")
    print("=" * 70)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    if not DUPLICATE_REPORT.exists():
        raise FileNotFoundError(
            f"Duplicate report not found: {DUPLICATE_REPORT}"
        )

    # ---------------------------------------------------------
    # Load exact duplicate report
    # ---------------------------------------------------------

    df = pd.read_csv(DUPLICATE_REPORT)

    print(f"Exact duplicate records: {len(df)}")

    # ---------------------------------------------------------
    # Convert CSV paths to paths relative to TRAIN_DIR
    # ---------------------------------------------------------

    duplicate_lookup = {}

    for _, row in df.iterrows():

        csv_train_path = Path(
            str(row["train_path"]).replace("\\", "/")
        )

        # Find the "train" component and everything after it.
        parts = csv_train_path.parts

        try:
            train_index = parts.index("train")
        except ValueError:
            raise ValueError(
                f"Could not locate 'train' in path: "
                f"{csv_train_path}"
            )

        relative_path = Path(
            *parts[train_index + 1:]
        )

        key = normalize_path(relative_path)

        duplicate_lookup[key] = {
            "status": row["status"],
            "test_path": row["test_path"],
            "test_class": row["test_class"]
        }

    print(
        f"Training duplicate paths loaded: "
        f"{len(duplicate_lookup)}"
    )

    # ---------------------------------------------------------
    # Create directories
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    QUARANTINE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Process training dataset
    # ---------------------------------------------------------

    print()
    print("Processing training dataset...")

    clean_train_count = 0
    duplicate_count = 0
    excluded_class_count = 0

    for class_dir in sorted(TRAIN_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # -----------------------------------------------------
        # Exclude the 2-image class
        # -----------------------------------------------------

        if class_name == EXCLUDED_CLASS:

            print(
                f"[EXCLUDED CLASS] {class_name}"
            )

            for image_path in class_dir.rglob("*"):

                if not image_path.is_file():
                    continue

                destination = (
                    QUARANTINE_DIR
                    / "excluded_2_image_class"
                    / class_name
                    / image_path.name
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                excluded_class_count += 1

            continue

        # -----------------------------------------------------
        # Process images
        # -----------------------------------------------------

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            relative_path = image_path.relative_to(
                TRAIN_DIR
            )

            key = normalize_path(relative_path)

            # -------------------------------------------------
            # Exact cross-split duplicate
            # -------------------------------------------------

            if key in duplicate_lookup:

                info = duplicate_lookup[key]

                status = info["status"]

                if status == "SAME_CLASS":
                    reason = "same_class_leak"
                else:
                    reason = "label_conflict"

                destination = (
                    QUARANTINE_DIR
                    / reason
                    / class_name
                    / image_path.name
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                duplicate_count += 1

                print(
                    f"[QUARANTINE] {status}: "
                    f"{relative_path}"
                )

                continue

            # -------------------------------------------------
            # Normal image
            # -------------------------------------------------

            destination = (
                OUTPUT_DIR
                / "train"
                / relative_path
            )

            copy_file(
                image_path,
                destination
            )

            clean_train_count += 1

    # ---------------------------------------------------------
    # Copy test set unchanged
    # ---------------------------------------------------------

    print()
    print("Copying test dataset unchanged...")

    clean_test_count = 0

    for class_dir in sorted(TEST_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            relative_path = image_path.relative_to(
                TEST_DIR
            )

            destination = (
                OUTPUT_DIR
                / "test"
                / relative_path
            )

            copy_file(
                image_path,
                destination
            )

            clean_test_count += 1

    # ---------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------

    manifest_rows = []

    for _, row in df.iterrows():

        manifest_rows.append({
            "train_file": row["train_path"],
            "train_class": row["train_class"],
            "test_file": row["test_path"],
            "test_class": row["test_class"],
            "status": row["status"],
            "action": "TRAINING_COPY_QUARANTINED"
        })

    for image_path in (
        TRAIN_DIR / EXCLUDED_CLASS
    ).rglob("*"):

        if image_path.is_file():

            manifest_rows.append({
                "train_file": str(image_path),
                "train_class": EXCLUDED_CLASS,
                "test_file": "",
                "test_class": "",
                "status": "EXCLUDED_CLASS",
                "action": "QUARANTINED"
            })

    manifest_path = (
        QUARANTINE_DIR /
        "cleanup_manifest.csv"
    )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        manifest_path,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)

    print(
        f"Clean training images       : "
        f"{clean_train_count}"
    )

    print(
        f"Clean test images           : "
        f"{clean_test_count}"
    )

    print(
        f"Duplicate train files       : "
        f"{duplicate_count}"
    )

    print(
        f"Excluded-class files        : "
        f"{excluded_class_count}"
    )

    print()
    print(
        f"Clean dataset: {OUTPUT_DIR}"
    )

    print(
        f"Quarantine: {QUARANTINE_DIR}"
    )

    print()
    print(
        "Original data/raw dataset was NOT modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()