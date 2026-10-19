# Evaluation report — measured results vs assumptions

## A. Measured in this environment (reproducible)
| Item | Result | How |
|---|---|---|
| Test suite | 67 passed, 0 failed, run with `-W error::DeprecationWarning` | `python -m pytest -q tests` |
| Active manifests (v2) | train 1822 / val 456 / test 227; 27 classes; 0 missing; 0 undecodable; all 224×224 RGB | `python src/data/verify_manifests.py` -> "manifests OK" |
| Quarantine | 73 images logged: 66 label-conflict, 5 duplicate a test image, 2 tiny class | `data/manifests_v2/quarantine.csv` |
| Leakage in v2 | 0 duplicate-group, 0 exact-SHA overlaps across splits; remaining 4 cross-split/cross-class near-pairs are exactly the pairs judged "different" in the provisional review | `reports/dataset_audit_v2/summary.json` |
| v1 manifests (for comparison) | 5 missing test files, 30 cross-split near-duplicate pairs, 52 cross-class near-pairs | `reports/dataset_audit/summary.json` |
| Filename collisions found | 12 (case-insensitive) | `preprocess_index.csv` (`name_collision`) |
| Old vs contract preprocessing | mean abs pixel diff 61.8/255 (300×180 test image) | `tests/test_inference_contract.py` |
| Calibration machinery | temperature recovers a known scale on synthetic logits; NLL/ECE/Brier improve on held-out synthetic data; Wilson-bound threshold policy behaves as specified | `tests/test_calibration*.py` (synthetic data only) |
| ONNX vs PyTorch | parity pipeline passes on a **random-init** network (not a model result) | `test_onnx_parity_with_pytorch` |

## B. NOT measured (blocked: pretrained weights + checkpoint unavailable)
Baseline and improved accuracy, macro/weighted F1, per-class metrics, confusion matrix, ECE/NLL/Brier of a real model, the chosen threshold and its coverage/error, experiment comparison (standard vs weighted sampler vs class-weighted loss), quantisation, latency, field accuracy. **No such numbers exist in this repository and none should be quoted.** The historical T=1.267833 / 0.70 are unverified.

## C. Assumptions and caveats
* Duplicate decisions are an AI-assisted visual review awaiting human confirmation (`duplicate_review.csv`, `human_confirmed=False`).
* dHash near-duplicate detection on padded images can false-positive (4 seen) and can miss heavily edited copies.
* Test = official PlantDoc test minus 9 label-conflict images: 227 images, 3–12 per class (Corn Gray leaf spot has 3) → wide intervals; bootstrap CIs treat images as independent.
* Threshold and temperature are both fitted on validation, so validation coverage/error is optimistic; only the frozen test run is unbiased and it is single-use.
* PlantDoc remains web-sourced; benchmark accuracy ≠ field accuracy.

## D. Commands to reproduce
```
pip install -r requirements-lock.txt
git clone https://github.com/pratikkayal/PlantDoc-Dataset data/raw/PlantDoc-Dataset   # use Linux/WSL: some test filenames contain '?' and cannot be checked out on Windows
python src/data/preprocess.py
python src/data/duplicate_groups.py
python src/data/review_sheets.py                 # human review -> edit data/manifests_v2/duplicate_review.csv, set human_confirmed=True, rerun duplicate_groups + build_manifests
python src/data/build_manifests.py
python src/data/verify_manifests.py
python src/data/dataset_audit.py --manifests data/manifests_v2 --out reports/dataset_audit_v2
python src/models/run_experiments.py --seeds 42,43,44       # needs network access for ImageNet weights
python src/models/predict.py --image leaf.jpg               # uses reports/mobilenetv3_baseline/
python src/models/export_onnx.py                            # real checkpoint only
python -m pytest -q tests
```
