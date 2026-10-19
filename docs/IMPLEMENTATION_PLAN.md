# Implementation plan and status

Status key: **DONE** = implemented and tested here; **BLOCKED** = needs a dependency this environment lacks (stated, with the exact next command); **LATER** = out of scope until a real checkpoint exists.

| Phase | Status | Evidence |
|---|---|---|
| 1 Data integrity | DONE (human sign-off pending) | `data/manifests_v2/`, `reports/dataset_audit_v2/`, `tests/test_data_integrity.py` |
| 2 Canonical pipeline | DONE | `src/leaflens/{config,checkpoint,data}.py`, `src/models/train.py`, legacy banners, `tests/test_checkpoint_and_training.py` |
| 3 Calibration / prediction | DONE | `src/leaflens/calibration*.py`, `src/models/{calibrate,select_threshold,predict}.py`, `tests/test_calibration_pipeline.py` |
| 4 Controlled experiments | **BLOCKED** (pipeline DONE) | `src/models/{train,run_experiments,evaluate}.py`, `src/leaflens/metrics.py`, `tests/test_evaluation.py` |
| 5 Reliability / deployment | PARTIAL | advisory quality, safe loading, pinned deps, ONNX export+parity (needs real checkpoint), Grad-CAM labelling; TFLite/INT8/Android/backend LATER |

## Phase 1 — data integrity
* `src/data/preprocess.py`: EXIF-aware, collision-safe (case-insensitive, per class, deterministic `__hash8` suffix on *both* colliders), filename-sanitised, writes `preprocess_index.csv`. Found **12 name collisions** (e.g. `CAR1.jpg`/`car1.jpg`) that would overwrite each other on Windows/macOS. The old script also skipped EXIF while inference applied it; both now share `leaflens.inference`.
* `src/data/duplicate_groups.py`: union-find over exact (SHA-256) + dHash Hamming ≤4 pairs; honours `data/manifests_v2/duplicate_review.csv` (`same`/`different`).
* `src/data/review_sheets.py` -> `reports/dataset_audit/review/sheet_*.png` (44 pairs). **Decisions in `duplicate_review.csv` are marked `assistant_provisional` / `human_confirmed=False`**: 40 `same`, 4 `different`. A human must confirm them.
* `src/data/build_manifests.py`: frozen official test split; group-aware train/val (`StratifiedGroupKFold`, seed 42); quarantines (never silently drops) 66 label-conflict images, 5 train images that duplicate test images, 2 images of a 2-image class. Result: train 1822 / val 456 / test 227, 27 classes.
* `src/data/verify_manifests.py`: paths, decode, 224×224 RGB, labels, class presence, group + SHA leakage across splits, label conflicts, frozen-test hash. Exit 1 on any problem.
* The 5 formerly missing test files are not "recovered" by copying: v2 is rebuilt from the raw dataset with sanitised names, so every active manifest path exists (0 missing).

## Phase 2 — one pipeline
Checkpoint schema v1 (`leaflens/checkpoint.py`): arch, ordered classes, class_to_idx, preprocessing id, mean/std, meta (experiment, seed, manifest hashes, `random_init`). Loaded with `torch.load(weights_only=True)`, validated, rejected on any mismatch. Canonical default path `reports/mobilenetv3_baseline/best_model.pth` is now the *output* of `train.py --experiment standard`, which resolves the old mismatch (training wrote `reports/mobilenetv3_exp00x`, evaluators read `.../baseline`). Legacy scripts (`train_mobilenetv3_v2/_v3`, `calibrate_mobilenetv3`, `evaluate_*`, `confidence_threshold_analysis`, `quality_performance_analysis`, `src/training/train.py`) carry a LEGACY banner + DeprecationWarning; none deleted.

## Phase 3 — calibration and prediction
Temperature optimised over log T (positive by construction, clamped to [0.05, 10], at-bound flagged). Fitted on validation logits only; artifact `calibration.json` binds `checkpoint_sha256` + `class_hash` + val manifest hash and reports NLL/ECE(15 equal-width bins)/Brier/accuracy before and after. Threshold = lowest threshold whose accepted-prediction Wilson-95% upper error ≤ `--max_error` (default 0.10) with ≥ `--min_accepted` (30) samples; if none qualifies the threshold is `null` and nothing is ever accepted. 1.267833 and 0.70 are retained only as `UNVALIDATED_LEGACY_*` constants; they are not defaults anywhere. `predict.py` errors on a missing/mismatched/unreadable artifact; `--allow_uncalibrated` yields `calibrated:false`, never `accepted`. Contract: logits → ÷T once → softmax once → policy. Invalid image, non-finite logits, malformed mapping, bad T all raise or return `invalid_image`.

## Phase 4 — experiments (BLOCKED on weights)
`run_experiments.py` runs `standard`, `weighted_sampler`, `class_weighted_loss` (one factor each; same seed(s), augmentation, AdamW 3e-4, cosine, patience 7, predeclared selection metric = validation macro-F1), then calibration, threshold and validation evaluation per run, picks the winner by mean val macro-F1 (tie → lower val NLL), and evaluates **only the winner, once**, on the frozen test split. `evaluate.py` refuses test evaluation unless the manifest hash matches the frozen one, `--final` is passed, a threshold policy exists, the checkpoint is not random-init, and no prior `test_evaluation.json` exists. Metrics: accuracy, macro/weighted F1, per-class P/R/F1/support/errors, confusion matrix, seeded bootstrap 95% CIs, paired-bootstrap helper.
**Blocker (verified):** `download.pytorch.org` returns HTTP 403 from this environment and no checkpoint exists in the repo; `train.py` and the runner stop with exit code 2/1 and write nothing. Random-init training is available only behind `--allow_random_init`, stamps checkpoints `random_init=true`, and is rejected by test evaluation, ONNX export and the runner's test step.
**Next command (machine with network access):** `python src/models/run_experiments.py --seeds 42,43,44`

## Phase 5
Done: advisory-only image quality in `predict.py` output (`image_quality_advisory`, thresholds flagged unvalidated, cannot change status); safe checkpoint loading; pinned `requirements-lock.txt`; Pillow and ONNX deprecations removed (suite passes with `-W error::DeprecationWarning`); Grad-CAM keeps hook hygiene and `overlay()` stamps "model attention - not a diagnosis"; `export_onnx.py` (real checkpoints only, fixed batch 1, per-image parity report).
Not done / LATER: validating quality thresholds on field photos, TFLite + INT8, Android, backend, latency/memory, offline-on-device tests, OOD rejection.
