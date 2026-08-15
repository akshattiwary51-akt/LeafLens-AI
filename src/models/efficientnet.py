"""
Alternative lightweight backbone — EfficientNet-Lite0 / EfficientNet-B0.

Kept as a second option to benchmark against MobileNetV3-Small in
notebooks/06_mobile_benchmark.ipynb: EfficientNet-B0 (~5.3M params) is
typically a few points more accurate than MobileNetV3-Small (~2.5M params)
but slower/larger on-device. Milestone 1 should report the accuracy-vs-size
trade-off with real numbers before Milestone 2 commits to one.
"""
try:
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_efficientnet_b0(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    if not TORCH_AVAILABLE:
        raise ImportError("torch/torchvision not installed in this environment")

    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
