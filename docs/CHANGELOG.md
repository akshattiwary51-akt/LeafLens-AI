# Prioritised change log (file by file)

Priority: **P0** correctness/leakage · **P1** pipeline consistency · **P2** reliability/tooling · **P3** docs.

## P0
| File | Change |
|---|---|
| `src/leaflens/inference.py` (new) | Single inference contract: EXIF, RGB, pad-to-square 224, ImageNet norm, logits→÷T→softmax once, Policy validation (`threshold` None ⇒ never accept), advisory quality. |
| `src/models/predict.py` | Rewritten: schema checkpoint + checkpoint-bound calibration artifact; no hard-coded T/threshold; errors on mismatch. (Old version squashed images: mean pixel diff 61.8/255 vs training.) |
| `src/data/preprocess.py` | EXIF-aware, collision-safe, sanitised names, index CSV, failures recorded. |
| `src/data/duplicate_groups.py`, `review_sheets.py`, `build_manifests.py`, `verify_manifests.py` (new) | Duplicate groups, human-review hook, leak-free group-aware manifests, quarantine log, frozen test hash, verifier. |
| `data/manifests_v2/*` (new) | Corrected manifests (POSIX paths), `classes.json`, `quarantine.csv`, `duplicate_*.csv`, `duplicate_review.csv` (provisional), `split_report.json`, `TEST_FROZEN.sha256`. Old `data/manifests/` untouched (legacy, leaky). |
## P1
| File | Change |
|---|---|
| `src/leaflens/{config,checkpoint,data}.py` (new) | Shared config, schema-v1 checkpoint, `weights_only` loading, manifest dataset using inference code for evaluation. |
| `src/models/train.py` (new) | Canonical trainer, 3 single-factor experiments, early stopping on val macro-F1, blocks without pretrained weights, never overwrites. |
| `src/models/calibrate.py`, `select_threshold.py` (new); `src/leaflens/calibration.py`, `calibration_artifact.py` | Val-only temperature (log-space, positive), artifact binding, ECE/NLL/Brier before/after, reliability CSV/PNG, Wilson-bound threshold policy. |
| `src/models/evaluate.py`, `run_experiments.py`, `src/leaflens/metrics.py` (new) | Metrics + bootstrap CIs; guarded single-use test evaluation; experiment orchestration. |
| `src/models/{train_mobilenetv3_v2,_v3,calibrate_mobilenetv3,evaluate_*,confidence_threshold_analysis,quality_performance_analysis}.py`, `src/training/train.py` | LEGACY banner; manifest path portability patch (previous round). |
## P2
| File | Change |
|---|---|
| `src/explainability/gradcam.py` | Hook-safe Grad-CAM, `overlay()` with non-diagnostic caption. |
| `src/models/export_onnx.py` (new) | ONNX export + parity report; refuses smoke checkpoints. |
| `src/data/dataset_audit.py`, `src/data/audit.py` | Reproducible audit; Pillow deprecation fix. |
| `requirements.txt`, `requirements-lock.txt` | Ranges + verified pins; dropped unused `albumentations`. |
| `tests/*` | 5 → 67 tests (see docs/EVALUATION.md). |
## P3
`docs/IMPLEMENTATION_PLAN.md`, `CHANGELOG.md`, `MODEL_CARD.md`, `EVALUATION.md`, `KNOWN_LIMITATIONS.md`, `ARCHITECTURE.md`, `DEPLOYMENT.md`, `REPOSITORY_AUDIT.md` (status addendum), `README.md` (reproduce section).
