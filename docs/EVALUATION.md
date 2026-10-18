# Evaluation
## Measured in this audit
| Item | Result | Command |
|---|---|---|
| Unit/regression tests | see final run in PR notes | `python -m pytest -q tests` |
| Dataset integrity | 2464 listed, 2459 readable 224×224 RGB, 5 test files missing, 27 classes | `python src/data/dataset_audit.py` |
| Class balance (train) | 27 classes, max/min = 3.94 | same |
| Leakage | 6 exact-dup groups (1 cross-split, 4 cross-class); 84 near-dup pairs (30 cross-split, 52 cross-class) | same |
| Old vs new preprocessing | mean abs pixel diff 61.8/255 on a 300×180 test image | `tests/test_inference_contract.py` |
| ONNX parity (random-init net) | max rtol 1e-3 | `test_onnx_parity_with_pytorch` |
The dataset audit needs the images at `data/preprocessed/plantdoc_27class_224/` (rebuild: `git clone https://github.com/pratikkayal/PlantDoc-Dataset`, then `python src/data/preprocess.py --input_dir <clone> --output_dir data/preprocessed/plantdoc_27class_224`).
Near-duplicate = 64-bit dHash Hamming ≤ 4 (≤1 pairs were visually plausible same-image copies; 4-bit pairs may include false positives and need human review).
## NOT measured (blocked)
Baseline / improved accuracy, macro-F1, per-class metrics, confusion matrix, ECE/NLL/Brier of the real model, threshold–coverage curve, quantisation, latency, field accuracy. Prerequisite: checkpoint `reports/mobilenetv3_baseline/best_model.pth`. Then run: `evaluate_mobilenetv3.py`, `calibrate_mobilenetv3.py`, `evaluate_calibrated_test.py` (test used once, after thresholds are frozen).
