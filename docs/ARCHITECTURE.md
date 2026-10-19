# Architecture (as implemented)

```
raw photo ─► decode (PIL, EXIF, RGB) ─► reject if undecodable / <32px
          ─► long side→224 (LANCZOS), centre on black 224² canvas   [== training preprocessing]
          ─► /255, ImageNet mean/std, NCHW float32
          ─► MobileNetV3-Small (27-way) ─► raw logits
          ─► logits / T  ─► softmax (exactly once)
          ─► confidence ≥ threshold ? accepted : low_confidence
```
Code: `src/leaflens/inference.py` (contract), `src/models/predict.py` (CLI/model loading), `src/leaflens/calibration.py` (NLL, Brier, ECE, temperature fit, coverage–error curve).

**Output statuses:** `accepted`, `low_confidence`, `invalid_image`. Every result carries "Model output, not an expert diagnosis." An OOD/"unknown disease" outcome is *not* implemented; high confidence does not imply the input is in-distribution.

**Explainability:** Grad-CAM (`src/explainability/gradcam.py`) works on the PyTorch model with `model.features[-1]` as target layer. It is unavailable under ONNX/TFLite inference.

**Not present:** mobile app, backend/API, database, Docker, CI.

## Pipeline (canonical)
```
raw PlantDoc -> preprocess.py -> duplicate_groups.py -> (human review) -> build_manifests.py -> verify_manifests.py
   -> train.py (x3 experiments) -> calibrate.py (val) -> select_threshold.py (val) -> evaluate.py (val; test once, frozen)
   -> predict.py / export_onnx.py
```
Checkpoint schema v1 and `calibration.json` are the only interface between stages; both are hash-bound.
