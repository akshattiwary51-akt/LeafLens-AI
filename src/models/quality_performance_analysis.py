# LEGACY: superseded by the canonical pipeline (src/models/train.py, calibrate.py,
# select_threshold.py, evaluate.py). Kept only to reproduce historical numbers; it reads the
# old, leaky data/manifests (v1) and legacy checkpoint layout. See docs/IMPLEMENTATION_PLAN.md.
import warnings as _w
_w.warn("legacy LeafLens script: use the canonical pipeline in docs/IMPLEMENTATION_PLAN.md",
        DeprecationWarning, stacklevel=1)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaflens.manifest_paths import resolve_image_path
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TEMPERATURE = 1.267833
BATCH_SIZE = 16


# ============================================================
# Dataset
# ============================================================

class PlantDocDataset(Dataset):

    def __init__(
        self,
        dataframe,
        class_to_idx,
        transform
    ):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            resolve_image_path(row["image_path"])
        ).convert("RGB")

        image = self.transform(image)

        label = self.class_to_idx[
            row["class"]
        ]

        return image, label


# ============================================================
# Image quality
# ============================================================

def analyze_quality(path):

    image = cv2.imread(str(path))

    if image is None:
        return {
            "blur_score": np.nan,
            "brightness": np.nan,
            "contrast": np.nan,
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        gray.mean()
    )

    contrast = float(
        gray.std()
    )

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
    }


# ============================================================
# Model
# ============================================================

def load_model(
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    classes = checkpoint["classes"]

    class_to_idx = checkpoint["class_to_idx"]

    model = models.mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, classes, class_to_idx


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="data/manifests/val_manifest.csv"
    )

    parser.add_argument(
        "--checkpoint",
        default="reports/mobilenetv3_baseline/best_model.pth"
    )

    parser.add_argument(
        "--out",
        default="reports/image_quality/quality_performance.csv"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("IMAGE QUALITY → MODEL PERFORMANCE ANALYSIS")
    print("=" * 70)

    print(
        f"Device      : {device}"
    )

    print(
        f"Temperature : {TEMPERATURE}"
    )

    # --------------------------------------------------------
    # Load validation manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        args.manifest
    )

    print(
        f"Validation images: {len(df)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, classes, class_to_idx = load_model(
        args.checkpoint,
        device
    )

    idx_to_class = {
        v: k
        for k, v in class_to_idx.items()
    }

    transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        ),
    ])

    dataset = PlantDocDataset(
        df,
        class_to_idx,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    predictions = []
    confidences = []
    true_labels = []

    print()
    print("Running model inference...")

    with torch.no_grad():

        processed = 0

        for images, labels in loader:

            images = images.to(device)

            logits = model(images)

            calibrated_probs = torch.softmax(
                logits / TEMPERATURE,
                dim=1
            )

            confidence, prediction = (
                calibrated_probs.max(dim=1)
            )

            predictions.extend(
                prediction.cpu().numpy()
            )

            confidences.extend(
                confidence.cpu().numpy()
            )

            true_labels.extend(
                labels.numpy()
            )

            processed += len(images)

            print(
                f"  Processed "
                f"{processed}/{len(dataset)}"
            )

    # --------------------------------------------------------
    # Attach model results
    # --------------------------------------------------------

    df["predicted_class"] = [
        idx_to_class[int(x)]
        for x in predictions
    ]

    df["confidence"] = confidences

    df["correct"] = (
        np.array(predictions)
        ==
        np.array(true_labels)
    )

    # --------------------------------------------------------
    # Quality analysis
    # --------------------------------------------------------

    print()
    print("Analyzing image quality...")

    quality = []

    for i, path in enumerate(
        df["image_path"],
        1
    ):

        quality.append(
            analyze_quality(path)
        )

        if (
            i % 50 == 0
            or i == len(df)
        ):
            print(
                f"  Quality "
                f"{i}/{len(df)}"
            )

    quality_df = pd.DataFrame(
        quality
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            quality_df
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Save per-image report
    # --------------------------------------------------------

    output = Path(args.out)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("QUALITY / PERFORMANCE SUMMARY")
    print("=" * 70)

    print(
        f"Overall accuracy: "
        f"{df['correct'].mean():.4f}"
    )

    for feature in [
        "blur_score",
        "brightness",
        "contrast"
    ]:

        print()
        print(
            f"--- {feature} ---"
        )

        correct = df[
            df["correct"] == True
        ][feature].mean()

        incorrect = df[
            df["correct"] == False
        ][feature].mean()

        print(
            f"Correct mean   : "
            f"{correct:.2f}"
        )

        print(
            f"Incorrect mean : "
            f"{incorrect:.2f}"
        )

    # --------------------------------------------------------
    # Quality quartiles
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PERFORMANCE BY QUALITY QUARTILE")
    print("=" * 70)

    for feature in [
        "blur_score",
        "brightness",
        "contrast"
    ]:

        print()
        print(
            f"FEATURE: {feature}"
        )

        try:

            df[f"{feature}_quartile"] = pd.qcut(
                df[feature],
                4,
                labels=[
                    "Q1_low",
                    "Q2",
                    "Q3",
                    "Q4_high"
                ],
                duplicates="drop"
            )

            summary = (
                df
                .groupby(
                    f"{feature}_quartile",
                    observed=True
                )
                .agg(
                    images=("correct", "size"),
                    accuracy=("correct", "mean"),
                    mean_confidence=(
                        "confidence",
                        "mean"
                    )
                )
            )

            summary["accuracy_percent"] = (
                summary["accuracy"] * 100
            )

            print(
                summary.to_string()
            )

        except ValueError:

            print(
                "Unable to create quartiles."
            )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("QUALITY / CORRECTNESS CORRELATION")
    print("=" * 70)

    for feature in [
        "blur_score",
        "brightness",
        "contrast"
    ]:

        correlation = df[
            [feature, "correct"]
        ].corr().iloc[0, 1]

        print(
            f"{feature:12s}: "
            f"{correlation:.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"Report: {output}"
    )


if __name__ == "__main__":
    main()