# Known limitations
* **No measured model performance**: pretrained weights unreachable here and no checkpoint in the repo (docs/EVALUATION.md §B).
* Duplicate-review decisions are provisional (AI-assisted) until a human confirms `duplicate_review.csv`.
* 66 label-conflict images (mostly Potato/Tomato blight, Corn leaf diseases) are quarantined, not relabelled; PlantDoc label noise likely extends beyond detectable duplicates.
* Test set is tiny (227; Corn Gray leaf spot = 3) → wide confidence intervals; bootstrap treats images as independent.
* Validation is used for both temperature and threshold (optimistic); the single-use test run is the unbiased check.
* dHash near-duplicate detection is imperfect (false positives on padded images; misses heavy edits).
* Quality diagnostics are advisory and their thresholds are unvalidated on field photos; they never reject.
* No OOD / unknown-disease detection; high confidence does not imply in-distribution input.
* Benchmark ≠ field performance; no field dataset. `src/evaluation/field_eval.py` is still a stub.
* Grad-CAM is server-side/PyTorch only and is diagnostic, not proof.
* Not implemented: TFLite/INT8, Android/Flutter app, backend, on-device latency/offline tests.
* Windows cannot check out some PlantDoc test filenames (`?`); build on Linux/WSL.
* Pipeline code paths were exercised with synthetic data and random-init smoke runs only.
