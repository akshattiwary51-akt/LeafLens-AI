from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


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

# Learned from validation calibration
TEMPERATURE = 1.267833

# Initial deployment policy.
# We will tune this later using a dedicated validation policy experiment.
CONFIDENCE_THRESHOLD = 0.70

TOP_K = 3


def load_model(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    classes = checkpoint["classes"]

    class_to_idx = checkpoint["class_to_idx"]

    model = models.mobilenet_v3_small(
        weights=None
    )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, classes, class_to_idx


def get_transform():

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        ),
    ])


def predict(
    image_path,
    model,
    class_to_idx,
    transform,
    device,
    confidence_threshold
):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():

        logits = model(tensor)

        # Temperature scaling
        calibrated_logits = (
            logits / TEMPERATURE
        )

        probabilities = torch.softmax(
            calibrated_logits,
            dim=1
        )[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(TOP_K, len(probabilities))
    )

    idx_to_class = {
        value: key
        for key, value in class_to_idx.items()
    }

    top_predictions = []

    for probability, index in zip(
        top_probs,
        top_indices
    ):

        top_predictions.append({
            "class": idx_to_class[int(index)],
            "probability": round(
                float(probability),
                4
            )
        })

    predicted_class = top_predictions[0]["class"]
    confidence = top_predictions[0]["probability"]

    if confidence >= confidence_threshold:
        status = "accepted"
    else:
        status = "uncertain"

    return {
        "image": str(image_path),
        "prediction": predicted_class,
        "confidence": confidence,
        "status": status,
        "calibrated": True,
        "temperature": TEMPERATURE,
        "threshold": confidence_threshold,
        "top_predictions": top_predictions,
        "model": "MobileNetV3-Small",
        "num_classes": len(class_to_idx)
    }

def main():

    parser = argparse.ArgumentParser(
        description=(
            "LeafLens AI single-image "
            "disease prediction"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to leaf image"
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "reports/"
            "mobilenetv3_baseline/"
            "best_model.pth"
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("LEAFLENS AI — SINGLE IMAGE PREDICTION")
    print("=" * 70)

    print(f"Device      : {device}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Threshold   : {args.threshold}")

    model, classes, class_to_idx = load_model(
        args.checkpoint,
        device
    )

    transform = get_transform()

    result = predict(
        image_path=args.image,
        model=model,
        class_to_idx=class_to_idx,
        transform=transform,
        device=device,
        confidence_threshold=args.threshold
    )

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print(
        f"Class      : "
        f"{result['prediction']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Status     : "
        f"{result['status']}"
    )

    print()
    print("TOP-3 PREDICTIONS")

    for i, item in enumerate(
        result["top_predictions"],
        1
    ):

        print(
            f"{i}. "
            f"{item['class']} "
            f"({item['probability']:.4f})"
        )

    print()
    print("=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()