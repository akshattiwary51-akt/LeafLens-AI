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
)


# ============================================================
# Configuration
# ============================================================

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

# Learned ONLY from validation calibration.
TEMPERATURE = 1.267833

BATCH_SIZE = 16

THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
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
# Load model
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

    val_manifest = Path(
        "data/manifests/val_manifest.csv"
    )

    output_dir = Path(
        "reports/confidence_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Temperature: {TEMPERATURE}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        classes,
        class_to_idx
    ) = load_model(
        checkpoint_path,
        device
    )

    # --------------------------------------------------------
    # Validation dataset
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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Validation images: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    logits_list = []

    labels_list = []

    print()
    print("Running validation inference...")

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
                (batch_idx + 1) * BATCH_SIZE,
                len(dataset)
            )

            print(
                f"  Processed "
                f"{processed}/{len(dataset)}"
            )

    logits = torch.cat(
        logits_list
    )

    y_true = torch.cat(
        labels_list
    ).numpy()

    # --------------------------------------------------------
    # Calibrated probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits / TEMPERATURE,
        dim=1
    ).numpy()

    predictions = (
        probabilities.argmax(
            axis=1
        )
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    correct = (
        predictions == y_true
    )

    # --------------------------------------------------------
    # Overall calibrated validation metrics
    # --------------------------------------------------------

    overall_accuracy = accuracy_score(
        y_true,
        predictions
    )

    overall_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print(
        f"Overall Accuracy : "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Overall Macro F1 : "
        f"{overall_f1:.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    for threshold in THRESHOLDS:

        accepted = (
            confidence >= threshold
        )

        accepted_count = int(
            accepted.sum()
        )

        rejected_count = (
            len(accepted)
            -
            accepted_count
        )

        coverage = (
            accepted_count
            /
            len(accepted)
        )

        if accepted_count > 0:

            accepted_accuracy = (
                accuracy_score(
                    y_true[accepted],
                    predictions[accepted]
                )
            )

            accepted_error_rate = (
                1.0 -
                accepted_accuracy
            )

            accepted_macro_f1 = (
                f1_score(
                    y_true[accepted],
                    predictions[accepted],
                    average="macro",
                    zero_division=0
                )
            )

            mean_confidence = float(
                confidence[accepted].mean()
            )

        else:

            accepted_accuracy = np.nan
            accepted_error_rate = np.nan
            accepted_macro_f1 = np.nan
            mean_confidence = np.nan

        # Rejected images that were actually
        # correct/incorrect.
        if rejected_count > 0:

            rejected_correct = int(
                correct[~accepted].sum()
            )

            rejected_incorrect = (
                rejected_count
                -
                rejected_correct
            )

        else:

            rejected_correct = 0
            rejected_incorrect = 0

        results.append({

            "threshold":
                threshold,

            "accepted":
                accepted_count,

            "rejected":
                rejected_count,

            "coverage":
                coverage,

            "coverage_percent":
                coverage * 100,

            "accepted_accuracy":
                accepted_accuracy,

            "accepted_accuracy_percent":
                accepted_accuracy * 100
                if not np.isnan(
                    accepted_accuracy
                )
                else np.nan,

            "accepted_error_rate":
                accepted_error_rate,

            "accepted_macro_f1":
                accepted_macro_f1,

            "mean_confidence":
                mean_confidence,

            "rejected_correct":
                rejected_correct,

            "rejected_incorrect":
                rejected_incorrect,
        })

        print(
            f"Threshold {threshold:.2f} | "
            f"Accepted {accepted_count:3d} | "
            f"Rejected {rejected_count:3d} | "
            f"Coverage {coverage * 100:6.2f}% | "
            f"Accepted Acc "
            f"{accepted_accuracy * 100:6.2f}%"
            if accepted_count > 0
            else
            f"Threshold {threshold:.2f} | "
            f"Accepted   0 | "
            f"Rejected {rejected_count:3d}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_dir /
        "threshold_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # Find useful candidate thresholds
    # --------------------------------------------------------

    candidates = []

    for target_accuracy in [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
    ]:

        eligible = results_df[
            results_df[
                "accepted_accuracy"
            ]
            >= target_accuracy
        ]

        if len(eligible) > 0:

            best = eligible.sort_values(
                "coverage",
                ascending=False
            ).iloc[0]

            candidates.append({

                "target_accuracy":
                    target_accuracy,

                "threshold":
                    float(best["threshold"]),

                "coverage":
                    float(best["coverage"]),

                "accepted_accuracy":
                    float(
                        best[
                            "accepted_accuracy"
                        ]
                    ),

                "accepted_macro_f1":
                    float(
                        best[
                            "accepted_macro_f1"
                        ]
                    ),
            })

    candidates_df = pd.DataFrame(
        candidates
    )

    candidates_df.to_csv(
        output_dir /
        "threshold_candidates.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "model":
            "MobileNetV3-Small",

        "validation_images":
            len(dataset),

        "temperature":
            TEMPERATURE,

        "overall_accuracy":
            float(overall_accuracy),

        "overall_macro_f1":
            float(overall_f1),

        "thresholds_evaluated":
            THRESHOLDS,
    }

    with open(
        output_dir /
        "threshold_analysis.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"CSV: "
        f"{output_dir / 'threshold_analysis.csv'}"
    )

    print(
        f"Candidates: "
        f"{output_dir / 'threshold_candidates.csv'}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()