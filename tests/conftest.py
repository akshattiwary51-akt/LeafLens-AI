import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


import json as _json

import numpy as _np
import pandas as _pd
import pytest as _pytest
from PIL import Image as _Image


@_pytest.fixture
def tiny_dataset(tmp_path):
    """3 classes x 12 images, already-224 padded files + v2-style manifests."""
    root = tmp_path
    classes = ["Class A", "Class B", "Class C"]
    rows = {"train": [], "val": [], "test": []}
    rng = _np.random.default_rng(0)
    for ci, c in enumerate(classes):
        for k in range(12):
            split = "train" if k < 8 else ("val" if k < 10 else "test")
            base = _np.full((224, 224, 3), 40 + 70 * ci, _np.uint8)
            base = (base + rng.integers(0, 25, base.shape)).clip(0, 255).astype(_np.uint8)
            rel = f"img/{split}/{c}/{k}.jpg"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            _Image.fromarray(base).save(p, quality=95)
            rows[split].append(dict(image_path=rel, **{"class": c}, split=split,
                                    group_id=f"g{ci}_{k}", source_sha256=f"{ci}-{k}"))
    md = root / "manifests"
    md.mkdir()
    for s, r in rows.items():
        _pd.DataFrame(r).to_csv(md / f"{s}_manifest.csv", index=False)
    (md / "classes.json").write_text(_json.dumps(
        {"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}))
    return dict(root=root, manifests=md, classes=classes)
