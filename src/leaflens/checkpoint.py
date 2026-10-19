"""Checkpoint schema v1 + safe loading.

A checkpoint is a plain dict (loaded with ``weights_only=True``):
  schema_version, arch, num_classes, classes, class_to_idx, image_size, mean,
  std, preprocessing_id, model_state_dict, meta
``meta`` records experiment, seed, manifest hashes, best epoch/metric and
``random_init`` (True => smoke-test artifact, never a result).
Calibration artifacts bind to ``checkpoint_sha256(path)`` + ``class_hash``.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from . import config as cfg
from .inference import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class CheckpointError(ValueError):
    pass


def class_hash(classes):
    return hashlib.sha256(json.dumps(list(classes)).encode()).hexdigest()[:16]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(num_classes, pretrained):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def validate_class_mapping(classes, class_to_idx):
    if not classes or len(set(classes)) != len(classes):
        raise CheckpointError("classes empty or contains duplicates")
    if sorted(class_to_idx.values()) != list(range(len(classes))):
        raise CheckpointError("class_to_idx is not a dense 0..C-1 mapping")
    if any(class_to_idx.get(c) != i for i, c in enumerate(classes)):
        raise CheckpointError("classes order disagrees with class_to_idx")


def save_checkpoint(path, model, classes, meta):
    classes = list(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    validate_class_mapping(classes, class_to_idx)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": cfg.CHECKPOINT_SCHEMA_VERSION,
        "arch": cfg.ARCH, "num_classes": len(classes),
        "classes": classes, "class_to_idx": class_to_idx,
        "image_size": IMAGE_SIZE,
        "mean": [float(x) for x in IMAGENET_MEAN], "std": [float(x) for x in IMAGENET_STD],
        "preprocessing_id": cfg.PREPROCESSING_ID,
        "model_state_dict": model.state_dict(),
        "meta": {**meta, "saved_utc": datetime.now(timezone.utc).isoformat()},
    }, path)


def load_checkpoint(path, device="cpu"):
    """Return (model.eval(), info dict). Raises CheckpointError / FileNotFoundError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        raise CheckpointError(f"cannot read checkpoint safely: {e}") from e
    if not isinstance(ck, dict) or ck.get("schema_version") != cfg.CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported or legacy checkpoint schema (expected "
                              f"schema_version={cfg.CHECKPOINT_SCHEMA_VERSION})")
    for k in ("arch", "classes", "class_to_idx", "model_state_dict", "preprocessing_id", "image_size"):
        if k not in ck:
            raise CheckpointError(f"checkpoint missing '{k}'")
    if ck["arch"] != cfg.ARCH:
        raise CheckpointError(f"unsupported arch {ck['arch']}")
    if ck["preprocessing_id"] != cfg.PREPROCESSING_ID or ck["image_size"] != IMAGE_SIZE:
        raise CheckpointError("checkpoint preprocessing does not match this code's inference contract")
    validate_class_mapping(ck["classes"], ck["class_to_idx"])
    model = build_model(len(ck["classes"]), pretrained=False)
    try:
        model.load_state_dict(ck["model_state_dict"])
    except RuntimeError as e:
        raise CheckpointError(f"weights do not fit the architecture: {e}") from e
    info = {"classes": ck["classes"], "class_to_idx": ck["class_to_idx"],
            "meta": ck.get("meta", {}), "sha256": file_sha256(path),
            "class_hash": class_hash(ck["classes"])}
    return model.to(device).eval(), info
