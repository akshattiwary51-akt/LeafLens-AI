"""Training configuration — single source of truth, YAML-overridable."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    dataset_manifest_dir: str = "data/processed"
    model_name: str = "mobilenet_v3_small"   # or "efficientnet_b0"
    num_classes: int = 8
    class_names: List[str] = field(default_factory=list)
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    pretrained: bool = True
    freeze_backbone_epochs: int = 3   # warm up classifier head before unfreezing
    class_weighted_loss: bool = True  # PlantDoc has 90x imbalance — see reports/dataset_audit
    early_stopping_patience: int = 7
    seed: int = 42
    output_dir: str = "models/run_001"
    log_every_n_steps: int = 20
