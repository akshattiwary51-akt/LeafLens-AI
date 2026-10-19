"""Single-image prediction CLI. Uses the shared inference contract in
src/leaflens/inference.py (pad-to-square preprocessing identical to training,
logits -> one temperature -> one softmax -> confidence policy)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from leaflens import inference as inf


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    if sorted(class_to_idx.values()) != list(range(len(classes))):
        raise ValueError("class_to_idx is not a dense 0..C-1 mapping")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), classes, class_to_idx


def predict(image_path, model, class_to_idx, policy, device="cpu"):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    try:
        x = inf.preprocess(image_path)
    except inf.InvalidImageError as e:
        return {**inf.invalid_result(e), "image": str(image_path)}
    with torch.inference_mode():
        logits = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
    return {**inf.decide(logits, idx_to_class, policy), "image": str(image_path),
            "model": "MobileNetV3-Small", "num_classes": len(idx_to_class)}


def load_policy(checkpoint, temperature, threshold):
    """Temperature: CLI > calibration.json next to checkpoint > default."""
    if temperature is None:
        cal = Path(checkpoint).with_name("calibration.json")
        if cal.exists():
            temperature = float(json.loads(cal.read_text())["temperature"])
        else:
            print(f"WARNING: no calibration.json beside checkpoint; using default "
                  f"T={inf.DEFAULT_TEMPERATURE} (may not match this checkpoint)",
                  file=sys.stderr)
            temperature = inf.DEFAULT_TEMPERATURE
    return inf.Policy(threshold=threshold, temperature=temperature)


def main():
    ap = argparse.ArgumentParser(description="LeafLens AI single-image prediction")
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="reports/mobilenetv3_baseline/best_model.pth")
    ap.add_argument("--threshold", type=float, default=inf.DEFAULT_THRESHOLD)
    ap.add_argument("--temperature", type=float, default=None)
    a = ap.parse_args()
    device = "cpu"
    try:
        model, _, class_to_idx = load_model(a.checkpoint, device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as e:
        sys.exit(f"Could not load model '{a.checkpoint}': {e}")
    policy = load_policy(a.checkpoint, a.temperature, a.threshold)
    print(json.dumps(predict(a.image, model, class_to_idx, policy, device), indent=2))


if __name__ == "__main__":
    main()
