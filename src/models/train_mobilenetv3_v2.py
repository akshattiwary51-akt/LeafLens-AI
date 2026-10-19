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
import copy
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
    f1_score,
    classification_report,
    confusion_matrix,
)


IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


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
# Transforms
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=12
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])

    val_transform = transforms.Compose([

        transforms.Resize(
            IMAGE_SIZE
        ),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])

    return train_transform, val_transform


# ============================================================
# Class weights
# ============================================================

def calculate_class_weights(
    dataframe,
    class_to_idx
):

    counts = dataframe["class"].value_counts()

    total = len(dataframe)

    num_classes = len(class_to_idx)

    weights = []

    for class_name, _ in sorted(
        class_to_idx.items(),
        key=lambda x: x[1]
    ):

        count = counts[class_name]

        weight = total / (
            num_classes * count
        )

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# Model
# ============================================================

def build_model(num_classes):

    print("\nLoading MobileNetV3-Small...")

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model = models.mobilenet_v3_small(
        weights=weights
    )

    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# ============================================================
# Freeze / unfreeze
# ============================================================

def freeze_backbone(model):

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):

    for parameter in model.parameters():
        parameter.requires_grad = True


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    labels_all = []
    predictions_all = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        labels_all.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        predictions_all.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0
    )

    return loss, accuracy, macro_f1


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    labels_all = []
    predictions_all = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        predictions_all.extend(
            predictions.cpu().numpy()
        )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        labels_all,
        predictions_all,
        average="weighted",
        zero_division=0
    )

    return (
        loss,
        accuracy,
        macro_f1,
        weighted_f1,
        labels_all,
        predictions_all
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_manifest",
        default="data/manifests/train_manifest.csv"
    )

    parser.add_argument(
        "--val_manifest",
        default="data/manifests/val_manifest.csv"
    )

    parser.add_argument(
        "--output_dir",
        default="reports/mobilenetv3_exp002"
    )

    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--phase2_epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--phase1_lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--phase2_lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("MobileNetV3-Small Experiment 002")
    print("=" * 70)

    print(f"Device: {device}")
    print(
        f"Phase 1 epochs: {args.phase1_epochs}"
    )
    print(
        f"Phase 2 epochs: {args.phase2_epochs}"
    )
    print(
        f"Batch size: {args.batch_size}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = pd.read_csv(
        args.train_manifest
    )

    val_df = pd.read_csv(
        args.val_manifest
    )

    classes = sorted(
        train_df["class"].unique()
    )

    class_to_idx = {
        name: i
        for i, name in enumerate(classes)
    }

    num_classes = len(classes)

    print()
    print(f"Classes: {num_classes}")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")

    train_transform, val_transform = (
        get_transforms()
    )

    train_dataset = PlantDocDataset(
        train_df,
        class_to_idx,
        train_transform
    )

    val_dataset = PlantDocDataset(
        val_df,
        class_to_idx,
        val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        num_classes
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    weights = calculate_class_weights(
        train_df,
        class_to_idx
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    history = []

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    total_start = time.time()

    # ========================================================
    # PHASE 1
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 1 — CLASSIFIER WARM-UP")
    print("=" * 70)

    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.phase1_lr,
        weight_decay=args.weight_decay
    )

    for epoch in range(
        1,
        args.phase1_epochs + 1
    ):

        start = time.time()

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        (
            val_loss,
            val_acc,
            val_macro_f1,
            val_weighted_f1,
            labels,
            predictions
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print()
        print(
            f"Phase 1 Epoch "
            f"{epoch}/{args.phase1_epochs}"
        )

        print(
            f"  Train Loss : {train_loss:.4f}"
        )

        print(
            f"  Train Acc  : {train_acc:.4f}"
        )

        print(
            f"  Val Loss   : {val_loss:.4f}"
        )

        print(
            f"  Val Acc    : {val_acc:.4f}"
        )

        print(
            f"  Val MacroF1: {val_macro_f1:.4f}"
        )

        history.append({
            "phase": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "learning_rate": args.phase1_lr,
            "seconds": time.time() - start
        })

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_epoch = epoch

            print(
                "  ★ New best checkpoint."
            )

    # ========================================================
    # PHASE 2
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 2 — FULL FINE-TUNING")
    print("=" * 70)

    unfreeze_all(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.phase2_lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    for epoch in range(
        1,
        args.phase2_epochs + 1
    ):

        start = time.time()

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        (
            val_loss,
            val_acc,
            val_macro_f1,
            val_weighted_f1,
            labels,
            predictions
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_macro_f1
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()
        print(
            f"Phase 2 Epoch "
            f"{epoch}/{args.phase2_epochs}"
        )

        print(
            f"  Train Loss : {train_loss:.4f}"
        )

        print(
            f"  Train Acc  : {train_acc:.4f}"
        )

        print(
            f"  Train F1   : {train_f1:.4f}"
        )

        print(
            f"  Val Loss   : {val_loss:.4f}"
        )

        print(
            f"  Val Acc    : {val_acc:.4f}"
        )

        print(
            f"  Val MacroF1: {val_macro_f1:.4f}"
        )

        print(
            f"  Val F1     : {val_weighted_f1:.4f}"
        )

        print(
            f"  LR         : {current_lr:.6f}"
        )

        history.append({
            "phase": 2,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "learning_rate": current_lr,
            "seconds": time.time() - start
        })

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = (
                args.phase1_epochs + epoch
            )

            epochs_without_improvement = 0

            print(
                "  ★ New best checkpoint."
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print()
            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Restore best
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    (
        val_loss,
        val_acc,
        val_macro_f1,
        val_weighted_f1,
        labels,
        predictions
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        labels,
        predictions,
        labels=list(range(num_classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        output_dir / "classification_report.csv"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(num_classes))
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        output_dir / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Normalized confusion matrix
    # --------------------------------------------------------

    row_sums = cm.sum(
        axis=1,
        keepdims=True
    )

    normalized_cm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=float
        ),
        where=row_sums != 0
    )

    normalized_df = pd.DataFrame(
        normalized_cm,
        index=classes,
        columns=classes
    )

    normalized_df.to_csv(
        output_dir / "confusion_matrix_normalized.csv"
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        output_dir / "training_history.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    checkpoint = {
        "model_state_dict": best_state,
        "class_to_idx": class_to_idx,
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "best_val_macro_f1": best_macro_f1,
        "best_epoch": best_epoch
    }

    torch.save(
        checkpoint,
        output_dir / "best_model.pth"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "experiment": "EXP-002",
        "model": "MobileNetV3-Small",
        "classes": num_classes,
        "train_images": len(train_df),
        "validation_images": len(val_df),
        "image_size": IMAGE_SIZE,
        "phase1_epochs": args.phase1_epochs,
        "phase1_lr": args.phase1_lr,
        "phase2_epochs_requested": args.phase2_epochs,
        "phase2_lr": args.phase2_lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "validation_accuracy": val_acc,
        "validation_weighted_f1": val_weighted_f1,
        "device": str(device),
        "seed": args.seed,
        "training_time_seconds":
            time.time() - total_start
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT 002 COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch       : {best_epoch}"
    )

    print(
        f"Best Macro F1    : "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Validation Acc   : "
        f"{val_acc:.4f}"
    )

    print(
        f"Validation F1    : "
        f"{val_weighted_f1:.4f}"
    )

    print()
    print(
        f"Output: {output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()