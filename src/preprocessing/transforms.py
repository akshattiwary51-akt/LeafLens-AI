"""
Train/val/test image transforms — Milestone 1 §11.

Requires torchvision (see requirements.txt). Written now so Milestone 2
can start training immediately once the full dev environment (with
torch/torchvision) is provisioned — this sandbox could not install a
CPU-only torch build without pulling multi-GB CUDA wheels (see README
"Environment limitations" section), so these transforms are unit-tested
against plain PIL logic in tests/test_transforms.py rather than end-to-end.

Design rules followed (per spec §11):
- Augmentation must not destroy disease symptoms: no aggressive color
  jitter, no vertical flips (leaves have a canonical up/down orientation
  a farmer's photo would still respect), no heavy cutout/erasing that
  could remove the lesion itself.
- Validation/test transforms are fully deterministic (no randomness).
"""
try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = IMAGE_SIZE):
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision not installed in this environment")
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),          # biologically fine: mirrored leaf still valid
        transforms.RandomRotation(degrees=15),            # small — camera tilt, not symptom-destroying
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # mild — field lighting varies
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),  # mimics phone-camera softness
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = IMAGE_SIZE):
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision not installed in this environment")
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
