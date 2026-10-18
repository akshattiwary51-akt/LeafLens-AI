from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def analyze_image(path):
    image = cv2.imread(str(path))

    if image is None:
        return {
            "image_path": str(path),
            "width": 0,
            "height": 0,
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "status": "unreadable",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # Laplacian variance = sharpness/blur indicator.
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Mean grayscale intensity.
    brightness = float(gray.mean())

    # Standard deviation = contrast.
    contrast = float(gray.std())

    return {
        "image_path": str(path),
        "width": width,
        "height": height,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "status": "valid",
    }


def main():

    parser = argparse.ArgumentParser(
        description="Analyze image quality for PlantDoc images."
    )

    parser.add_argument(
        "--data_dir",
        default="data/preprocessed/plantdoc_27class_224/test",
    )

    parser.add_argument(
        "--out_csv",
        default="reports/image_quality/test_quality.csv",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.out_csv)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    files = [
        p
        for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
    ]

    print("=" * 70)
    print("IMAGE QUALITY ANALYSIS")
    print("=" * 70)

    print(f"Input directory : {data_dir}")
    print(f"Images found    : {len(files)}")

    results = []

    for i, path in enumerate(files, 1):

        results.append(
            analyze_image(path)
        )

        if (
            i % 50 == 0
            or i == len(files)
        ):
            print(
                f"  Processed "
                f"{i}/{len(files)}"
            )

    df = pd.DataFrame(results)

    df.to_csv(
        output,
        index=False
    )

    print()
    print("=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    valid = df[
        df["status"] == "valid"
    ]

    print(
        f"Valid images : {len(valid)}"
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
            print(column)

            print(
                valid[column].describe()
            )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"CSV: {output}"
    )


if __name__ == "__main__":
    main()