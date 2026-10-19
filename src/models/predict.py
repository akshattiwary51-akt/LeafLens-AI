"""Single-image prediction CLI using the canonical contract.

Pipeline: schema-validated checkpoint (weights_only) -> calibration artifact bound to
that checkpoint (T and threshold from VALIDATION data) -> pad-to-square preprocessing
identical to training -> logits / T -> one softmax -> accept / low_confidence.

A missing or mismatched calibration artifact is an error unless --allow_uncalibrated is
given, in which case results are labelled calibrated=false and are never "accepted".
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from leaflens import config as cfg
from leaflens import inference as inf
from leaflens.calibration_artifact import CalibrationError, load_policy
from leaflens.checkpoint import CheckpointError, load_checkpoint


def predict(image_path, model, info, policy, quality=True):
    idx_to_class = {v: k for k, v in info["class_to_idx"].items()}
    try:
        x = inf.preprocess(image_path)
    except FileNotFoundError:
        raise
    except inf.InvalidImageError as e:
        return {**inf.invalid_result(e), "image": str(image_path)}
    with torch.inference_mode():
        logits = model(torch.from_numpy(x))[0].numpy()
    res = inf.decide(logits, idx_to_class, policy)
    res.update(image=str(image_path), model="MobileNetV3-Small", num_classes=len(idx_to_class),
               random_init_checkpoint=bool(info["meta"].get("random_init")))
    if quality:
        res["image_quality_advisory"] = inf.quality_advisory(image_path)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="LeafLens AI single-image prediction")
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default=f"{cfg.BASELINE_DIR}/{cfg.CHECKPOINT_NAME}")
    ap.add_argument("--allow_uncalibrated", action="store_true")
    a = ap.parse_args(argv)
    try:
        model, info = load_checkpoint(a.checkpoint)
        policy, _ = load_policy(a.checkpoint, info, require=not a.allow_uncalibrated)
    except (FileNotFoundError, CheckpointError, CalibrationError) as e:
        sys.exit(f"ERROR: {e}")
    try:
        print(json.dumps(predict(a.image, model, info, policy), indent=2))
    except FileNotFoundError:
        sys.exit(f"ERROR: image not found: {a.image}")


if __name__ == "__main__":
    main()
