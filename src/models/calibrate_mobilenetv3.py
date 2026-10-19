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


IMAGE_SIZE = 224

MEAN = [
    0.485,
    0.456,
    0.406,
]

STD = [
    0.229,
    0.224,
    0.225,
]


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
        class_to_idx,
        checkpoint
    )


# ============================================================
# Collect validation logits
# ============================================================

def collect_logits(
    model,
    loader,
    device
):

    logits_list = []
    labels_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            logits = model(images)

            logits_list.append(
                logits.cpu()
            )

            labels_list.append(
                labels
            )

    logits = torch.cat(
        logits_list
    )

    labels = torch.cat(
        labels_list
    )

    return logits, labels


# ============================================================
# Temperature scaling
# ============================================================

def find_temperature(
    logits,
    labels
):

    temperature = torch.ones(
        1,
        requires_grad=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.01,
        max_iter=100
    )

    def closure():

        optimizer.zero_grad()

        loss = criterion(
            logits / temperature,
            labels
        )

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    temperature_value = (
        temperature.detach()
        .clamp(min=0.05)
        .item()
    )

    return temperature_value


# ============================================================
# ECE
# ============================================================

def expected_calibration_error(
    probabilities,
    labels,
    n_bins=10
):

    confidences = probabilities.max(
        axis=1
    )

    predictions = probabilities.argmax(
        axis=1
    )

    accuracies = (
        predictions == labels
    )

    ece = 0.0

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    for lower, upper in zip(
        bins[:-1],
        bins[1:]
    ):

        mask = (
            (confidences > lower)
            &
            (confidences <= upper)
        )

        if not np.any(mask):
            continue

        bin_accuracy = (
            accuracies[mask].mean()
        )

        bin_confidence = (
            confidences[mask].mean()
        )

        bin_fraction = (
            mask.mean()
        )

        ece += (
            bin_fraction
            *
            abs(
                bin_accuracy
                -
                bin_confidence
            )
        )

    return float(ece)


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

    val_manifest = Path(
        "data/manifests/val_manifest.csv"
    )

    output_dir = Path(
        "reports/calibration"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("MOBILENETV3 CONFIDENCE CALIBRATION")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        classes,
        class_to_idx,
        checkpoint
    ) = load_model(
        checkpoint_path,
        device
    )

    print(
        f"Classes: {len(classes)}"
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    df = pd.read_csv(
        val_manifest
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
        f"Validation images: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Collect logits
    # --------------------------------------------------------

    print()
    print("Collecting validation logits...")

    logits, labels = collect_logits(
        model,
        loader,
        device
    )

    # --------------------------------------------------------
    # Before calibration
    # --------------------------------------------------------

    raw_probabilities = torch.softmax(
        logits,
        dim=1
    ).numpy()

    y_true = labels.numpy()

    raw_predictions = (
        raw_probabilities.argmax(
            axis=1
        )
    )

    raw_ece = expected_calibration_error(
        raw_probabilities,
        y_true
    )

    raw_nll = log_loss(
        y_true,
        raw_probabilities,
        labels=list(range(len(classes)))
    )

    raw_accuracy = accuracy_score(
        y_true,
        raw_predictions
    )

    raw_f1 = f1_score(
        y_true,
        raw_predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Find temperature
    # --------------------------------------------------------

    print()
    print("Optimizing temperature...")

    temperature = find_temperature(
        logits,
        labels
    )

    print(
        f"Optimal temperature: "
        f"{temperature:.6f}"
    )

    # --------------------------------------------------------
    # After calibration
    # --------------------------------------------------------

    calibrated_probabilities = torch.softmax(
        logits / temperature,
        dim=1
    ).numpy()

    calibrated_predictions = (
        calibrated_probabilities.argmax(
            axis=1
        )
    )

    calibrated_ece = (
        expected_calibration_error(
            calibrated_probabilities,
            y_true
        )
    )

    calibrated_nll = log_loss(
        y_true,
        calibrated_probabilities,
        labels=list(range(len(classes)))
    )

    calibrated_accuracy = accuracy_score(
        y_true,
        calibrated_predictions
    )

    calibrated_f1 = f1_score(
        y_true,
        calibrated_predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Save calibration results
    # --------------------------------------------------------

    results = {

        "model":
            "MobileNetV3-Small",

        "validation_images":
            len(dataset),

        "temperature":
            temperature,

        "before_calibration": {

            "accuracy":
                float(raw_accuracy),

            "macro_f1":
                float(raw_f1),

            "nll":
                float(raw_nll),

            "ece":
                float(raw_ece),
        },

        "after_calibration": {

            "accuracy":
                float(calibrated_accuracy),

            "macro_f1":
                float(calibrated_f1),

            "nll":
                float(calibrated_nll),

            "ece":
                float(calibrated_ece),
        }
    }

    with open(
        output_dir /
        "calibration_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(
        f"Temperature: "
        f"{temperature:.4f}"
    )

    print()
    print("BEFORE CALIBRATION")

    print(
        f"Accuracy : "
        f"{raw_accuracy:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{raw_f1:.4f}"
    )

    print(
        f"NLL      : "
        f"{raw_nll:.4f}"
    )

    print(
        f"ECE      : "
        f"{raw_ece:.4f}"
    )

    print()
    print("AFTER CALIBRATION")

    print(
        f"Accuracy : "
        f"{calibrated_accuracy:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{calibrated_f1:.4f}"
    )

    print(
        f"NLL      : "
        f"{calibrated_nll:.4f}"
    )

    print(
        f"ECE      : "
        f"{calibrated_ece:.4f}"
    )

    print()
    print(
        "Saved:",
        output_dir /
        "calibration_results.json"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()