from pathlib import Path
import argparse
import shutil

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(data_dir):
    """Return all supported image files recursively."""
    data_dir = Path(data_dir)

    images = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                images.append(
                    (image_path, class_dir.name)
                )

    return images


def preprocess_image(
    image_path,
    output_path,
    image_size=224
):
    """
    Convert image to RGB, preserve aspect ratio,
    resize, and pad to a square.
    """

    with Image.open(image_path) as img:

        # Convert everything to RGB
        img = img.convert("RGB")

        # Preserve aspect ratio
        img.thumbnail(
            (image_size, image_size),
            Image.Resampling.LANCZOS
        )

        # Create square canvas
        canvas = Image.new(
            "RGB",
            (image_size, image_size),
            (0, 0, 0)
        )

        # Center image
        x = (image_size - img.width) // 2
        y = (image_size - img.height) // 2

        canvas.paste(img, (x, y))

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save as JPEG
        canvas.save(
            output_path,
            format="JPEG",
            quality=95
        )


def process_split(
    input_dir,
    output_dir,
    image_size
):
    """Process one dataset split."""

    images = collect_images(input_dir)

    print(
        f"\nProcessing: {input_dir}"
    )

    print(
        f"Images found: {len(images)}"
    )

    processed = 0
    failed = 0

    for index, (image_path, class_name) in enumerate(
        images,
        start=1
    ):

        # Keep class structure
        relative_name = image_path.stem + ".jpg"

        output_path = (
            output_dir
            / class_name
            / relative_name
        )

        try:

            preprocess_image(
                image_path,
                output_path,
                image_size
            )

            processed += 1

        except Exception as e:

            failed += 1

            print(
                f"[FAILED] {image_path}"
            )

            print(
                f"         {e}"
            )

        if (
            index % 200 == 0
            or index == len(images)
        ):
            print(
                f"  {index}/{len(images)}"
            )

    return processed, failed


def main():

    parser = argparse.ArgumentParser(
        description="Preprocess PlantDoc dataset"
    )

    parser.add_argument(
        "--input_dir",
        default="data/processed/plantdoc_27class"
    )

    parser.add_argument(
        "--output_dir",
        default="data/preprocessed/plantdoc_27class_224"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_input = input_dir / "train"
    test_input = input_dir / "test"

    train_output = output_dir / "train"
    test_output = output_dir / "test"

    if not train_input.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_input}"
        )

    if not test_input.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_input}"
        )

    print("=" * 70)
    print("PlantDoc Preprocessing")
    print("=" * 70)

    print(
        f"Input : {input_dir}"
    )

    print(
        f"Output: {output_dir}"
    )

    print(
        f"Size  : {args.image_size}x{args.image_size}"
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    train_processed, train_failed = process_split(
        train_input,
        train_output,
        args.image_size
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    test_processed, test_failed = process_split(
        test_input,
        test_output,
        args.image_size
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Train processed : {train_processed}"
    )

    print(
        f"Train failed    : {train_failed}"
    )

    print(
        f"Test processed  : {test_processed}"
    )

    print(
        f"Test failed     : {test_failed}"
    )

    print()
    print(
        f"Output directory: {output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()