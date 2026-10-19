# Model card — LeafLens MobileNetV3-Small (PlantDoc, 27 classes)

**Status: no trained model is released with this repository; no performance is claimed.**

| | |
|---|---|
| Architecture | torchvision MobileNetV3-Small, ImageNet init, 27-way head, 224×224 |
| Preprocessing | EXIF, RGB, aspect-preserving resize + black pad to 224², ImageNet mean/std (`leaflens/inference.py`) |
| Data | PlantDoc (CC BY 4.0). v2 manifests: 1822/456/227; duplicate-group-aware; 73 images quarantined (see `data/manifests_v2/quarantine.csv`) |
| Training | `src/models/train.py`; three single-factor variants; selection by validation macro-F1 |
| Calibration | single temperature fitted on validation; artifact bound to checkpoint hash |
| Decision | accepted / low_confidence / invalid_image; threshold from validation risk policy (may be null ⇒ never accepted) |
| Intended use | research prototype; a model prediction is **not** an expert diagnosis; no treatment advice |
| Measured results | **none yet** (blocked on pretrained weights; see docs/EVALUATION.md §B) |
| Assumed (not verified) | previously reported T=1.267833 and threshold 0.70 |
| Out of scope | unsupported crops/diseases (no OOD detector), non-leaf images, field-photo guarantees |
| Explanations | Grad-CAM = model attention on the PyTorch model only; not evidence of a correct diagnosis |
