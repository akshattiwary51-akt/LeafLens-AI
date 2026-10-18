from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def analyze_image(path):

    image = cv2.imread(str(path))

    if image is None:
        return {
            "image_path": str(path),
            "blur_score": np.nan,
            "brightness": np.nan,
            "contrast": np.nan,
            "width": 0,
            "height": 0,
            "status": "unreadable",
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    # Higher = sharper.
    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # Mean pixel intensity.
    brightness = float(
        gray.mean()
    )

    # Pixel standard deviation.
    contrast = float(
        gray.std()
    )

    return {
        "image_path": str(path),
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "width": width,
        "height": height,
        "status": "valid",
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="data/manifests/val_manifest.csv"
    )

    parser.add_argument(
        "--out_csv",
        default="reports/image_quality/val_quality.csv"
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    )

    output_path = Path(
        args.out_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        manifest_path
    )

    print("=" * 70)
    print("VALIDATION IMAGE QUALITY ANALYSIS")
    print("=" * 70)

    print(
        f"Manifest : {manifest_path}"
    )

    print(
        f"Images   : {len(df)}"
    )

    print(
        f"Classes  : {df['class'].nunique()}"
    )

    results = []

    for i, row in enumerate(
        df.itertuples(index=False),
        1
    ):

        path = Path(
            row.image_path
        )

        if not path.exists():

            result = {
                "image_path": str(path),
                "class": row._1
                if hasattr(row, "_1")
                else "",
                "blur_score": np.nan,
                "brightness": np.nan,
                "contrast": np.nan,
                "width": 0,
                "height": 0,
                "status": "missing",
            }

        else:

            result = analyze_image(
                path
            )

            result["class"] = row._1 \
                if hasattr(row, "_1") \
                else ""

        results.append(
            result
        )

        if (
            i % 50 == 0
            or i == len(df)
        ):
            print(
                f"  Processed "
                f"{i}/{len(df)}"
            )

    result_df = pd.DataFrame(
        results
    )

    # Ensure class column is correctly
    # populated from the manifest.
    result_df["class"] = df[
        "class"
    ].values

    result_df.to_csv(
        output_path,
        index=False
    )

    valid = result_df[
        result_df["status"] == "valid"
    ]

    print()
    print("=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    print(
        f"Valid images      : {len(valid)}"
    )

    print(
        f"Missing/unreadable: "
        f"{len(result_df) - len(valid)}"
    )

    if len(valid) > 0:

        for column in [
            "blur_score",
            "brightness",
            "contrast",
            "width",
            "height",
        ]:

            print()
            print(
                f"--- {column} ---"
            )

            print(
                valid[column].describe()
            )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"CSV: {output_path}"
    )


if __name__ == "__main__":
    main()