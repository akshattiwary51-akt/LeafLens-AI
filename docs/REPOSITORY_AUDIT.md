# LeafLens AI — Repository Audit

Audited commit: `35c2d6e` (branch `audit/harden-inference` carries the fixes). Evidence is source code, manifests and
measured outputs; README statements were not treated as proof.

## 1. Actual structure
| Path | Contents | Status |
|---|---|---|
| `src/data/` | audit, clean, dedup, preprocess (pad-to-224), split, quality scripts (+ new `dataset_audit.py`) | runs on a rebuilt dataset |
| `src/models/` | `train_mobilenetv3_v2.py` (EXP-002), `_v3.py` (EXP-003, sampler), `calibrate_mobilenetv3.py`, `evaluate_*.py`, `confidence_threshold_analysis.py`, `quality_performance_analysis.py`, `predict.py` | scripts present; **no checkpoint** |
| `src/leaflens/` (new) | inference contract, calibration metrics, portable manifest paths | tested |
| `src/explainability/gradcam.py` | Grad-CAM (rewritten) | tested on random-init model |
| `src/training/` + `src/preprocessing/` + `src/models/mobilenet.py`/`efficientnet.py` | an older generic pipeline (ImageFolder, 8 classes) | **dead / inconsistent**, see §8 |
| `data/manifests/` | train 1782 / val 446 / test 236 rows, 27 classes | present |
| `tests/` | 2 files (5 tests) originally; now 5 test files, 25 tests | |
| Flutter/Android, backend, Docker, CI | **do not exist** | N/A |

## 2. Verified stack
Python 3.13 sandbox; PyTorch 2.14 / torchvision 0.29, onnx 1.23, onnxruntime 1.29. `requirements.txt` has lower bounds only (unpinned); `albumentations` is listed but unused.

## 3–4. Architecture actually implemented
PlantDoc (CC BY 4.0), 27 classes (the 28th raw class, "Tomato two spotted spider mites leaf", has 2 images and is excluded). Images are pre-processed by `src/data/preprocess.py`: RGB, aspect-preserving `thumbnail` (LANCZOS), black-padded to 224×224, saved as JPEG q95. Model: torchvision MobileNetV3-Small, ImageNet init, last Linear → 27. Normalisation: ImageNet mean/std. Calibration: single temperature T fitted by LBFGS on validation NLL (`calibrate_mobilenetv3.py`); `T=1.267833` is hard-coded in `predict.py` and `evaluate_calibrated_test.py`. Policy: accepted if calibrated max-prob ≥ 0.70.
Not present: any TFLite/ONNX export script, mobile app, API, database.

## 5. Data flow (what exists)
image file → `predict.py` → model → logits/T → softmax → threshold → JSON. There is no capture, history or UI.

## 7. Feature status
| Feature | Status |
|---|---|
| Training, evaluation, calibration, threshold analysis scripts | implemented; not reproducible here (see §9) |
| Image-quality scoring | implemented (`src/data/image_quality*.py`, `preprocessing/quality.py`); **not wired into `predict.py`** |
| Grad-CAM | was broken for MobileNetV3 (see D4); fixed |
| ONNX/TFLite, offline mobile app, backend | not implemented (README only says "prepare for") |
| OOD / unknown rejection | not implemented |

## 8. Defects found
| # | Finding | Evidence | Action |
|---|---|---|---|
| D1 | **Train/inference preprocessing mismatch.** Training/eval data are pad-to-square; `predict.py` did `Resize((224,224))` on raw photos (squash). On a 300×180 image the mean absolute pixel difference between the two is 61.8/255. | old `predict.py` L67-80 vs `preprocess.py` L36-66 | fixed: `leaflens/inference.py`, regression test |
| D2 | Hard-coded `TEMPERATURE` duplicated in 2 files; can silently diverge from the checkpoint. | `predict.py` L26, `evaluate_calibrated_test.py` L30 | `predict.py` now reads `calibration.json` beside checkpoint, else warns and uses default |
| D3 | Manifests contain Windows `\` paths → every training/eval script fails on Linux/macOS. | `data/manifests/*.csv` | `leaflens/manifest_paths.py`, patched 8 scripts |
| D4 | Grad-CAM used `register_full_backward_hook` (incompatible with MobileNetV3's in-place Hardswish), never removed hooks, ran `backward` on a tensor not requiring grad. | old `gradcam.py` L27-28 | rewritten + tests |
| D5 | Inconsistent eval transforms: EXP-002 uses `Resize(224)+CenterCrop`, EXP-003/calibration/eval use `Resize((224,224))`. Identical on pre-padded 224² files, divergent for raw images. | `train_mobilenetv3_v2.py` L124-130 vs `_v3.py` L138 | documented; contract now pad-to-square |
| D6 | `src/training/train.py` imports `src.…`, uses `ImageFolder`, `Resize(255)+CenterCrop`, `num_classes=8`, claims "90x imbalance"; measured train imbalance is 3.9×. | `training/config.py` L10,19 | marked dead; not deleted |
| D7 | `preprocess.py` names outputs `stem + ".jpg"` so `a.jpg` and `a.png` in one class silently overwrite each other. | `preprocess.py` L88 | open |
| D8 | Many near-copies of `PlantDocDataset`/transform code across 6 scripts. | `src/models/*.py` | open |
| D9 | Data leakage / label noise (see `reports/dataset_audit/`): 30 near-duplicate pairs (Hamming ≤4) cross train/val/test, 9 involve the test set; 4 exact-duplicate groups and 52 near-duplicate pairs carry **conflicting class labels**. Val split is a random stratified subset of the original train folder with no grouping. | `reports/dataset_audit/summary.json` | open — needs human label review and grouped re-split |
| D10 | Test set is tiny (236; 4–12 per class) and 5 manifest test files are not reproducible from the public repo (`recover_missing_test.py` origin). Per-class test metrics have wide CIs. | audit | documented |
| D11 | `torch.load` without `weights_only` in 5 scripts (safe only on trusted checkpoints). | `calibrate_*`, `evaluate_*` | open (predict.py fixed) |
| D12 | Default threshold 0.70 is untuned ("initial deployment policy"). | `predict.py` comment | open; needs checkpoint |

## 9. What could not be done (blockers)
* **No trained checkpoint** (`*.pth` is git-ignored; none in repo) and **no ImageNet weights reachable** (download.pytorch.org, huggingface.co return 403 from this sandbox). Training from random init would not be the project's model, so **no baseline or improved accuracy is reported**. Nothing here measures the real model's accuracy, per-class F1 or calibration.
* No Android device/emulator, Flutter, TFLite converter: mobile latency, INT8 accuracy and offline-device tests were not performed.
* No field images: field validation (`src/evaluation/field_eval.py`) remains a stub.

## 10. Backlog (priority)
1. Provide the checkpoint (or ImageNet weights) → run baseline via `evaluate_mobilenetv3.py`, `calibrate_mobilenetv3.py` on the fixed paths.
2. Human-review `near_duplicate_pairs.csv`; re-split with groups (merge duplicate clusters before splitting); fix conflicting labels.
3. Choose threshold from validation coverage–error curve (`leaflens.calibration.coverage_error_curve`), then freeze it.
4. Wire image-quality gate into inference; add unknown/OOD outcome.
5. Export ONNX/TFLite from the contract, compare logits (parity test exists for ONNX on a random-init model).
6. Mobile app/backend only once 1–5 are done.
7. Consolidate duplicated dataset/transform code; pin dependencies; add CI.
