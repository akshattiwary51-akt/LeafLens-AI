"""
Lightweight model definition — MobileNetV3-Small backbone.

Chosen as the Milestone-1 target architecture because:
- ~2.5M parameters, ~10-25MB as INT8 TFLite/ONNX — realistic for an
  offline Android app (spec's mobile/edge constraint).
- ImageNet-pretrained weights available in torchvision, which matters a
  lot given our largest verified dataset (PlantDoc, real field images) is
  only ~2.6k images — transfer learning is not optional, it's required.
- torchvision ships an official ONNX-exportable definition, satisfying
  the ONNX/ONNX Runtime requirement in the environment spec.

Requires torch + torchvision (see requirements.txt). Not runnable in this
research sandbox — see README "Environment limitations".
"""
try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def build_mobilenet_v3_small(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    if not TORCH_AVAILABLE:
        raise ImportError("torch/torchvision not installed in this environment")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_device():
    """Detect CUDA -> MPS -> CPU, per spec §7."""
    if not TORCH_AVAILABLE:
        return "cpu (torch not installed)"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
