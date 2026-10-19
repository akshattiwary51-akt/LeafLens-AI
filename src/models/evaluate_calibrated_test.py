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
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# IMPORTANT:
# Learned ONLY from validation set.
TEMPERATURE = 1.267833


# ============================================================
# Dataset
# ============================================================

class PlantDocDataset(Dataset):

    def __init__(
        self,
        dataframe,
        class_to_idx,
        transform=None
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

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[
            row["class"]
        ]

        return image, label


# ============================================================
# ECE + Reliability Data
# ============================================================

def reliability_data(
    probabilities,
    labels,
    n_bins=10
):

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)

    correct = (
        predictions == labels
    )

    rows = []

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                &
                (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                &
                (confidences <= upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "count": 0,
                "mean_confidence": np.nan,
                "accuracy": np.nan,
                "gap": np.nan,
            })
            continue

        mean_confidence = float(
            confidences[mask].mean()
        )

        accuracy = float(
            correct[mask].mean()
        )

        gap = abs(
            accuracy -
            mean_confidence
        )

        fraction = (
            count /
            len(labels)
        )

        ece += fraction * gap

        rows.append({
            "bin": i + 1,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "gap": gap,
        })

    return float(ece), pd.DataFrame(rows)


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

    model = model.to(device)
    model.eval()

    return (
        model,
        classes,
        class_to_idx
    )


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = Path(
        "reports/mobilenetv3_baseline/best_model.pth"
    )

    test_manifest = Path(
        "data/manifests/test_manifest.csv"
    )

    output_dir = Path(
        "reports/calibration"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("FINAL TEST CALIBRATION EVALUATION")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Frozen temperature: {TEMPERATURE}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, classes, class_to_idx = load_model(
        checkpoint_path,
        device
    )

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    df = pd.read_csv(
        test_manifest
    )

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
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Test images: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Collect logits
    # --------------------------------------------------------

    logits_list = []
    labels_list = []

    print()
    print("Running test inference...")

    with torch.no_grad():

        for batch_idx, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(device)

            logits = model(images)

            logits_list.append(
                logits.cpu()
            )

            labels_list.append(
                labels
            )

            processed = min(
                (batch_idx + 1) * 16,
                len(dataset)
            )

            print(
                f"  Processed "
                f"{processed}/{len(dataset)}"
            )

    logits = torch.cat(
        logits_list
    )

    labels = torch.cat(
        labels_list
    )

    y_true = labels.numpy()

    # --------------------------------------------------------
    # BEFORE calibration
    # --------------------------------------------------------

    raw_probabilities = torch.softmax(
        logits,
        dim=1
    ).numpy()

    raw_predictions = (
        raw_probabilities.argmax(axis=1)
    )

    raw_accuracy = accuracy_score(
        y_true,
        raw_predictions
    )

    raw_macro_f1 = f1_score(
        y_true,
        raw_predictions,
        average="macro",
        zero_division=0
    )

    raw_nll = log_loss(
        y_true,
        raw_probabilities,
        labels=list(range(len(classes)))
    )

    raw_ece, raw_reliability = (
        reliability_data(
            raw_probabilities,
            y_true
        )
    )

    # --------------------------------------------------------
    # AFTER calibration
    # --------------------------------------------------------

    calibrated_probabilities = (
        torch.softmax(
            logits / TEMPERATURE,
            dim=1
        ).numpy()
    )

    calibrated_predictions = (
        calibrated_probabilities.argmax(axis=1)
    )

    calibrated_accuracy = accuracy_score(
        y_true,
        calibrated_predictions
    )

    calibrated_macro_f1 = f1_score(
        y_true,
        calibrated_predictions,
        average="macro",
        zero_division=0
    )

    calibrated_nll = log_loss(
        y_true,
        calibrated_probabilities,
        labels=list(range(len(classes)))
    )

    calibrated_ece, calibrated_reliability = (
        reliability_data(
            calibrated_probabilities,
            y_true
        )
    )

    # --------------------------------------------------------
    # Confidence statistics
    # --------------------------------------------------------

    raw_confidence = (
        raw_probabilities.max(axis=1)
    )

    calibrated_confidence = (
        calibrated_probabilities.max(axis=1)
    )

    correct_mask = (
        calibrated_predictions == y_true
    )

    # --------------------------------------------------------
    # Save reliability data
    # --------------------------------------------------------

    raw_reliability.to_csv(
        output_dir /
        "test_reliability_before.csv",
        index=False
    )

    calibrated_reliability.to_csv(
        output_dir /
        "test_reliability_after.csv",
        index=False
    )

    # --------------------------------------------------------
    # Reliability diagram
    # --------------------------------------------------------

    plt.figure(figsize=(7, 7))

    valid_raw = (
        raw_reliability[
            raw_reliability["count"] > 0
        ]
    )

    valid_cal = (
        calibrated_reliability[
            calibrated_reliability["count"] > 0
        ]
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        valid_raw["mean_confidence"],
        valid_raw["accuracy"],
        marker="o",
        label="Before calibration"
    )

    plt.plot(
        valid_cal["mean_confidence"],
        valid_cal["accuracy"],
        marker="o",
        label="After calibration"
    )

    plt.xlabel(
        "Mean predicted confidence"
    )

    plt.ylabel(
        "Observed accuracy"
    )

    plt.title(
        "MobileNetV3-Small Reliability Diagram"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "test_reliability_diagram.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Save final results
    # --------------------------------------------------------

    results = {

        "model":
            "MobileNetV3-Small",

        "test_images":
            len(dataset),

        "classes":
            len(classes),

        "temperature":
            TEMPERATURE,

        "before_calibration": {

            "accuracy":
                float(raw_accuracy),

            "macro_f1":
                float(raw_macro_f1),

            "nll":
                float(raw_nll),

            "ece":
                float(raw_ece),

            "mean_confidence":
                float(
                    raw_confidence.mean()
                ),
        },

        "after_calibration": {

            "accuracy":
                float(calibrated_accuracy),

            "macro_f1":
                float(calibrated_macro_f1),

            "nll":
                float(calibrated_nll),

            "ece":
                float(calibrated_ece),

            "mean_confidence":
                float(
                    calibrated_confidence.mean()
                ),

            "correct_mean_confidence":
                float(
                    calibrated_confidence[
                        correct_mask
                    ].mean()
                ),

            "incorrect_mean_confidence":
                float(
                    calibrated_confidence[
                        ~correct_mask
                    ].mean()
                ),
        }
    }

    with open(
        output_dir /
        "test_calibration_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST CALIBRATION RESULTS")
    print("=" * 70)

    print()
    print("BEFORE CALIBRATION")

    print(
        f"Accuracy          : "
        f"{raw_accuracy:.4f} "
        f"({raw_accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1          : "
        f"{raw_macro_f1:.4f}"
    )

    print(
        f"NLL               : "
        f"{raw_nll:.4f}"
    )

    print(
        f"ECE               : "
        f"{raw_ece:.4f}"
    )

    print(
        f"Mean Confidence   : "
        f"{raw_confidence.mean():.4f}"
    )

    print()
    print("AFTER CALIBRATION")

    print(
        f"Accuracy          : "
        f"{calibrated_accuracy:.4f} "
        f"({calibrated_accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1          : "
        f"{calibrated_macro_f1:.4f}"
    )

    print(
        f"NLL               : "
        f"{calibrated_nll:.4f}"
    )

    print(
        f"ECE               : "
        f"{calibrated_ece:.4f}"
    )

    print(
        f"Mean Confidence   : "
        f"{calibrated_confidence.mean():.4f}"
    )

    print(
        f"Correct Confidence: "
        f"{calibrated_confidence[correct_mask].mean():.4f}"
    )

    print(
        f"Incorrect Conf.   : "
        f"{calibrated_confidence[~correct_mask].mean():.4f}"
    )

    print()
    print(
        "Reports saved to:",
        output_dir
    )

    print("=" * 70)


if __name__ == "__main__":
    main()