# SIH — AI-Powered Early Crop Disease Detection
## Milestone 1: AI Feasibility & Dataset Validation

This repo contains **only** Milestone 1 (feasibility + dataset validation).
No mobile app, backend, dashboard, or auth — by design, per the task scope.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The classical CV / data-audit stack (numpy, pandas, scikit-learn, OpenCV,
Pillow, matplotlib) is enough to reproduce everything under `reports/`
that is marked "LOCALLY VERIFIED". Training the actual target CNN
(`src/training/train.py`) additionally needs `torch` + `torchvision`.

## Environment limitations hit in this research session (read this first)

This project was built inside a sandboxed research environment with a
**restricted network allowlist** (GitHub, PyPI/npm-style package
registries — no Kaggle, no Hugging Face, no IEEE DataPort downloads) and
**no GPU**. Two concrete consequences:

1. **PlantDoc** (small, CC BY 4.0, hosted on GitHub) was downloaded and
   audited for real — see `reports/dataset_audit_plantdoc_*.md` and
   `reports/baseline_feasibility_result_tomato.md` for genuine, locally
   computed numbers.
2. **PlantVillage** (Hugging Face, ~54k images) and **Paddy Doctor**
   (Kaggle, 16k images) could **not** be downloaded here. Their reported
   statistics in `reports/dataset_research.md` come from cross-checking
   3+ independent published sources, not from a local audit. Pulling
   them and re-running `src/data/audit.py` on them is the **first task**
   for whoever continues this in a full dev environment.
3. `torch`/`torchvision` could not be installed here without pulling
   multi-GB CUDA wheels (irrelevant anyway with no GPU present). So
   `src/models/`, `src/training/`, and `src/explainability/gradcam.py`
   are written and structured but **not run** in this session. In their
   place, a classical-features (HSV histogram + Random Forest) baseline
   was actually trained and evaluated on real PlantDoc data with
   scikit-learn/OpenCV, which are installed, to get genuine (if weak)
   feasibility evidence rather than none at all.

Every number in `reports/` is labeled with how it was obtained
(locally verified vs. literature cross-checked vs. single-source/
unverified) — see `reports/dataset_research.md` for the verification-tier
legend.

## Project structure

```
src/data/         download.py, audit.py, clean.py, split.py   — all runnable now
src/preprocessing/ transforms.py (needs torch), quality.py (runnable now)
src/models/        mobilenet.py, efficientnet.py               — needs torch
src/training/      config.py, train.py                         — needs torch
src/evaluation/    metrics.py, confusion_matrix.py, field_eval.py, baseline_feasibility_check.py (runnable now)
src/explainability/ gradcam.py                                 — needs torch
tests/              real unit tests against real downloaded data
reports/            all Milestone 1 deliverables — see FEASIBILITY_REPORT.md
```

## Reproducing what was actually run in this session

```bash
# 1. get real data
python3 src/data/download.py plantdoc --dest data/raw

# 2. audit it
python3 src/data/audit.py --data_dir data/raw/plantdoc/train --dataset_name plantdoc_train \
    --out_csv reports/dataset_audit_plantdoc_train.csv \
    --out_md reports/dataset_audit_plantdoc_train.md \
    --figures_dir reports/figures

# 3. clean it (flags cross-class label conflicts for human review, drops exact dupes)
python3 src/data/clean.py --audit_csv reports/dataset_audit_plantdoc_train.csv \
    --out_manifest data/interim/plantdoc_train_cleaned_manifest.csv

# 4. split it
python3 src/data/split.py --manifest_csv data/interim/plantdoc_train_cleaned_manifest.csv \
    --out_dir data/processed/plantdoc_train_split

# 5. quick real feasibility probe (classical CV, no torch needed)
python3 src/evaluation/baseline_feasibility_check.py \
    --train_dir data/raw/plantdoc/train --test_dir data/raw/plantdoc/test \
    --classes "Tomato Early blight leaf" "Tomato Septoria leaf spot" "Tomato leaf" \
              "Tomato leaf bacterial spot" "Tomato leaf late blight" \
              "Tomato leaf mosaic virus" "Tomato leaf yellow virus" "Tomato mold leaf"

# 6. once torch is installed in a full dev environment:
python3 src/training/train.py --train_dir data/raw/plantdoc/train --val_dir data/raw/plantdoc/test
```

## Read this next

- `reports/FEASIBILITY_REPORT.md` — the actual Milestone-1 deliverable:
  answers to all 15 success-criteria questions, GO/MODIFY/STOP call.
- `reports/dataset_research.md` — full dataset audit trail.
- `reports/mvp_scope.md` — selected crops/diseases and why.
