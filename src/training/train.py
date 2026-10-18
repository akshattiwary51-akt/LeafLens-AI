"""
Baseline training loop — Milestone 1 §"baseline model pipeline".

Requires torch/torchvision (not installed in this research sandbox; see
README "Environment limitations"). This script IS the artifact Milestone 2
runs first, once the dev environment has full package/network access.

Implements, per spec:
- automatic CUDA/MPS/CPU device selection (src/models/mobilenet.py)
- class-weighted loss for imbalanced data (confirmed 90x imbalance in
  PlantDoc — see reports/dataset_audit_plantdoc_train.md)
- deterministic seeding
- checkpointing best-val-macro-F1 model (not just best accuracy, because
  accuracy alone is misleading under heavy class imbalance)
"""
import argparse
import json
import os
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models.mobilenet import build_mobilenet_v3_small, get_device
from src.preprocessing.transforms import get_train_transforms, get_eval_transforms
from src.training.config import TrainConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def compute_class_weights(dataset) -> "torch.Tensor":
    counts = np.bincount([label for _, label in dataset.samples])
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def train(cfg: TrainConfig, train_dir: str, val_dir: str):
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch/torchvision are required to run training and are not installed in this "
            "sandbox. Install via `pip install torch torchvision` in the full dev environment "
            "(see requirements.txt) and re-run this script."
        )

    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = ImageFolder(train_dir, transform=get_train_transforms(cfg.image_size))
    val_ds = ImageFolder(val_dir, transform=get_eval_transforms(cfg.image_size))
    cfg.class_names = train_ds.classes
    cfg.num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = build_mobilenet_v3_small(cfg.num_classes, pretrained=cfg.pretrained).to(device)

    class_weights = compute_class_weights(train_ds).to(device) if cfg.class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_macro_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        if epoch < cfg.freeze_backbone_epochs:
            for p in model.features.parameters():
                p.requires_grad = False
        else:
            for p in model.features.parameters():
                p.requires_grad = True

        running_loss = 0.0
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % cfg.log_every_n_steps == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        from sklearn.metrics import f1_score, accuracy_score
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(f"epoch {epoch}: train_loss={running_loss/len(train_loader):.4f} "
              f"val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}")
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader),
                         "val_acc": val_acc, "val_macro_f1": val_macro_f1})

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print("Early stopping.")
                break

    with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
        json.dump({"history": history, "class_names": cfg.class_names,
                    "best_val_macro_f1": best_macro_f1}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--output_dir", default="models/run_001")
    args = ap.parse_args()

    cfg = TrainConfig(epochs=args.epochs, output_dir=args.output_dir)
    train(cfg, args.train_dir, args.val_dir)
