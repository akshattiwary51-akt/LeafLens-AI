"""Single source of truth for the canonical LeafLens pipeline."""
from dataclasses import asdict, dataclass

ARCH = "mobilenet_v3_small"
PREPROCESSING_ID = "pad_square_224_exif_imagenet_v1"  # see leaflens/inference.py
CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_DIR = "data/manifests_v2"
BASELINE_DIR = "reports/mobilenetv3_baseline"  # path every downstream script reads
CHECKPOINT_NAME = "best_model.pth"
CALIBRATION_NAME = "calibration.json"
SELECTION_METRIC = "val_macro_f1"  # predeclared; never test metrics

EXPERIMENTS = ("standard", "weighted_sampler", "class_weighted_loss")


@dataclass
class TrainConfig:
    manifest_dir: str = MANIFEST_DIR
    root: str = "."
    experiment: str = "standard"
    output_dir: str = BASELINE_DIR
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 7
    seed: int = 42
    num_workers: int = 0
    allow_random_init: bool = False  # smoke tests only; stamps the checkpoint

    def to_dict(self):
        return asdict(self)
