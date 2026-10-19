import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaflens.manifest_paths import resolve_image_path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
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

class PlantDocTestDataset(Dataset):

    def __init__(
        self,
        dataframe,
        class_to_idx,
        transform=None,
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
# Main
# ============================================================

def main():

    test_manifest = Path(
        "data/manifests/test_manifest.csv"
    )

    checkpoint_path = Path(
        "reports/mobilenetv3_baseline/best_model.pth"
    )

    output_dir = Path(
        "reports/final_test_evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load test manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        test_manifest
    )

    print(
        f"Test images: {len(df)}"
    )

    print(
        f"Classes: {df['class'].nunique()}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print("Loading best EXP-001 checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    classes = checkpoint["classes"]

    class_to_idx = checkpoint["class_to_idx"]

    num_classes = len(classes)

    print(
        f"Model classes: {num_classes}"
    )

    print(
        f"Best validation Macro F1: "
        f"{checkpoint.get('best_val_macro_f1', 'N/A')}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = models.mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Test transform
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    dataset = PlantDocTestDataset(
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

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    inference_times = []

    print()
    print("Running inference...")

    with torch.no_grad():

        for batch_idx, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(device)

            labels = labels.to(device)

            start = time.perf_counter()

            outputs = model(images)

            elapsed = (
                time.perf_counter()
                - start
            )

            inference_times.append(
                elapsed / len(images)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = (
                probabilities.argmax(
                    dim=1
                )
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            processed = min(
                (batch_idx + 1) * 16,
                len(dataset)
            )

            print(
                f"  Processed "
                f"{processed}/{len(dataset)}"
            )

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    probabilities = np.array(
        all_probabilities
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        output_dir /
        "classification_report.csv"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        output_dir /
        "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    prediction_confidence = (
        probabilities.max(axis=1)
    )

    prediction_rows = []

    for i in range(len(df)):

        true_idx = y_true[i]

        pred_idx = y_pred[i]

        prediction_rows.append({

            "image_path":
                df.iloc[i]["image_path"],

            "true_class":
                classes[true_idx],

            "predicted_class":
                classes[pred_idx],

            "correct":
                bool(true_idx == pred_idx),

            "confidence":
                float(
                    prediction_confidence[i]
                ),
        })

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    prediction_df.to_csv(
        output_dir /
        "test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Confidence statistics
    # --------------------------------------------------------

    correct_mask = (
        y_true == y_pred
    )

    correct_confidence = (
        prediction_confidence[
            correct_mask
        ]
    )

    incorrect_confidence = (
        prediction_confidence[
            ~correct_mask
        ]
    )

    # --------------------------------------------------------
    # Inference statistics
    # --------------------------------------------------------

    mean_inference_time = float(
        np.mean(inference_times)
    )

    median_inference_time = float(
        np.median(inference_times)
    )

    # --------------------------------------------------------
    # Final metrics JSON
    # --------------------------------------------------------

    results = {

        "model":
            "MobileNetV3-Small",

        "checkpoint":
            str(checkpoint_path),

        "test_images":
            len(df),

        "classes":
            num_classes,

        "validation_macro_f1":
            float(
                checkpoint.get(
                    "best_val_macro_f1",
                    -1
                )
            ),

        "test_accuracy":
            float(accuracy),

        "test_macro_precision":
            float(macro_precision),

        "test_macro_recall":
            float(macro_recall),

        "test_macro_f1":
            float(macro_f1),

        "test_weighted_f1":
            float(weighted_f1),

        "mean_confidence":
            float(
                np.mean(
                    prediction_confidence
                )
            ),

        "mean_correct_confidence":
            float(
                np.mean(
                    correct_confidence
                )
            ) if len(correct_confidence)
            else None,

        "mean_incorrect_confidence":
            float(
                np.mean(
                    incorrect_confidence
                )
            ) if len(incorrect_confidence)
            else None,

        "mean_inference_time_seconds":
            mean_inference_time,

        "median_inference_time_seconds":
            median_inference_time,
    }

    with open(
        output_dir /
        "final_metrics.json",
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
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test Accuracy       : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision     : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall        : "
        f"{macro_recall:.4f}"
    )

    print(
        f"★ Macro F1          : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Mean Confidence     : "
        f"{np.mean(prediction_confidence):.4f}"
    )

    print(
        f"Correct Confidence  : "
        f"{np.mean(correct_confidence):.4f}"
        if len(correct_confidence)
        else "Correct Confidence  : N/A"
    )

    print(
        f"Incorrect Confidence: "
        f"{np.mean(incorrect_confidence):.4f}"
        if len(incorrect_confidence)
        else "Incorrect Confidence: N/A"
    )

    print(
        f"Mean inference time : "
        f"{mean_inference_time * 1000:.2f} ms/image"
    )

    print(
        f"Median inference    : "
        f"{median_inference_time * 1000:.2f} ms/image"
    )

    print()
    print(
        f"Reports saved to: "
        f"{output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()