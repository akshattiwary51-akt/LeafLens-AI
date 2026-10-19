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

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


IMAGE_SIZE = 224

MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
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

    # Keep augmentation close to EXP-001.
    # We are testing SAMPLING, not augmentation.

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )
    ])

    val_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )
    ])

    return (
        train_transform,
        val_transform
    )


# ============================================================
# Model
# ============================================================

def build_model(num_classes):

    print()
    print("Loading MobileNetV3-Small...")

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
# Weighted sampler
# ============================================================

def create_sampler(
    dataframe,
    class_to_idx
):

    class_counts = (
        dataframe["class"]
        .value_counts()
    )

    # Inverse-frequency class weights
    class_weights = {
        class_name:
        1.0 / count

        for class_name, count
        in class_counts.items()
    }

    sample_weights = [
        class_weights[row["class"]]
        for _, row in dataframe.iterrows()
    ]

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

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

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

    loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        loss,
        accuracy,
        macro_f1
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        loss,
        accuracy,
        macro_f1,
        weighted_f1,
        all_labels,
        all_predictions
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
        default="reports/mobilenetv3_exp003"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

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
    print("MobileNetV3-Small Experiment 003")
    print("=" * 70)

    print(
        f"Device       : {device}"
    )

    print(
        f"Epochs       : {args.epochs}"
    )

    print(
        f"Batch size   : {args.batch_size}"
    )

    print(
        f"Learning rate: {args.lr}"
    )

    print()
    print(
        "Strategy: WeightedRandomSampler"
    )

    # --------------------------------------------------------
    # Load manifests
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
        name: index
        for index, name
        in enumerate(classes)
    }

    num_classes = len(classes)

    print()
    print(
        f"Classes      : {num_classes}"
    )

    print(
        f"Train images : {len(train_df)}"
    )

    print(
        f"Val images   : {len(val_df)}"
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform, val_transform = (
        get_transforms()
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sampler
    # --------------------------------------------------------

    sampler = create_sampler(
        train_df,
        class_to_idx
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
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
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Normal CrossEntropy
    #
    # Important:
    # We do NOT use class weights here because
    # WeightedRandomSampler already handles balancing.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_macro_f1 = -1.0

    best_state = None

    best_epoch = 0

    patience_counter = 0

    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1
    ):

        epoch_start = time.time()

        (
            train_loss,
            train_accuracy,
            train_macro_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        (
            val_loss,
            val_accuracy,
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

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch:02d}/{args.epochs}"
        )

        print(
            f"  Train Loss : {train_loss:.4f}"
        )

        print(
            f"  Train Acc  : {train_accuracy:.4f}"
        )

        print(
            f"  Train F1   : {train_macro_f1:.4f}"
        )

        print(
            f"  Val Loss   : {val_loss:.4f}"
        )

        print(
            f"  Val Acc    : {val_accuracy:.4f}"
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

        print(
            f"  Time       : {epoch_time:.1f}s"
        )

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "train_macro_f1": train_macro_f1,

            "val_loss": val_loss,

            "val_accuracy": val_accuracy,

            "val_macro_f1": val_macro_f1,

            "val_weighted_f1": val_weighted_f1,

            "learning_rate": current_lr,

            "epoch_time_seconds":
                epoch_time
        })

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            patience_counter = 0

            print(
                "  ★ New best model saved."
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= args.patience
        ):

            print()
            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    (
        final_loss,
        final_accuracy,
        final_macro_f1,
        final_weighted_f1,
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
        output_dir /
        "classification_report.csv"
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
        output_dir /
        "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        output_dir /
        "training_history.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint = {

        "model_state_dict":
            best_state,

        "class_to_idx":
            class_to_idx,

        "classes":
            classes,

        "image_size":
            IMAGE_SIZE,

        "best_val_macro_f1":
            best_macro_f1,

        "best_epoch":
            best_epoch
    }

    torch.save(
        checkpoint,
        output_dir /
        "best_model.pth"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "experiment":
            "EXP-003",

        "model":
            "MobileNetV3-Small",

        "strategy":
            "WeightedRandomSampler",

        "loss":
            "CrossEntropyLoss",

        "classes":
            num_classes,

        "train_images":
            len(train_df),

        "validation_images":
            len(val_df),

        "image_size":
            IMAGE_SIZE,

        "epochs_requested":
            args.epochs,

        "epochs_completed":
            len(history),

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_macro_f1,

        "validation_accuracy":
            final_accuracy,

        "validation_weighted_f1":
            final_weighted_f1,

        "device":
            str(device),

        "seed":
            args.seed,

        "training_time_seconds":
            time.time() - start_time
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
    print("EXPERIMENT 003 COMPLETE")
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
        f"{final_accuracy:.4f}"
    )

    print(
        f"Validation F1    : "
        f"{final_weighted_f1:.4f}"
    )

    print()
    print(
        f"Output: {output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()