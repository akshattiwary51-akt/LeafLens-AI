# 🌿 LeafLens AI — Offline Crop Disease Detection

> **Status (read first):** this repository ships a research pipeline, not a validated model. No checkpoint or accuracy figures are included; earlier numbers (T = 1.267833, threshold 0.70) are unverified. See `docs/IMPLEMENTATION_PLAN.md`, `docs/EVALUATION.md` (measured vs not measured) and `docs/KNOWN_LIMITATIONS.md`. Reproduce with the commands in `docs/EVALUATION.md` §D.


> **AI-powered, lightweight, confidence-aware crop disease detection for mobile and edge deployment.**

LeafLens AI is a lightweight computer-vision system designed to identify crop leaf diseases from images using an **ImageNet-pretrained MobileNetV3-Small** model.

The project focuses on the requirements of real-world agricultural deployment:

- 📱 Lightweight architecture for mobile/edge devices
- 📴 Offline inference capability
- 🔍 Early-stage disease classification
- 🎯 Confidence-aware predictions
- 📊 Probabilistic output rather than only hard classification
- 🧪 Model calibration using temperature scaling
- 🖼️ Image-quality analysis
- 🚨 Confidence thresholding and rejection of uncertain predictions
- 📈 Class-wise and dataset-level evaluation
- 🔬 Explainability support through Grad-CAM
- 🧩 Modular training, evaluation, preprocessing and inference pipeline

---

## 📌 Project Overview

Farmers often face difficulty identifying plant diseases at an early stage, especially when expert agricultural support is unavailable.

LeafLens AI aims to provide a practical computer-vision pipeline where a user can provide an image of a plant leaf and receive:

1. Predicted disease/class
2. Calibrated confidence probability
3. Top-3 predictions
4. Acceptance/rejection status based on confidence
5. Image-quality information
6. Explainability support for understanding model attention

The current Milestone-1 implementation establishes the **research and ML foundation** required for a future offline mobile application.

---

# 🎯 Objectives

### Primary Objective

Build a lightweight image-classification model capable of recognizing plant diseases from leaf images while remaining suitable for eventual mobile/edge deployment.

### Secondary Objectives

- Build a reproducible dataset preprocessing pipeline
- Establish a reliable train/validation/test split
- Train a lightweight CNN using transfer learning
- Evaluate performance using class-balanced metrics
- Analyze failure cases using confusion matrices
- Measure prediction confidence
- Improve confidence calibration
- Determine a practical confidence threshold
- Analyze the effect of image quality on prediction performance
- Prepare the model for future ONNX/TFLite/mobile deployment
- Provide a foundation for explainable and field-oriented inference

---

# 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │      Leaf Image      │
                    │  Camera / Gallery     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Image Validation   │
                    │ Quality / Readability │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Preprocessing     │
                    │ Resize → Normalize   │
                    │     224 × 224        │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │     MobileNetV3-Small      │
                 │   ImageNet Transfer Learn  │
                 └─────────────┬──────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Raw Logits       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Temperature Scaling  │
                    │    T = 1.267833       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Calibrated Softmax   │
                    │   Probabilities      │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Confidence ≥ 0.70      Confidence < 0.70
                    │                     │
                    ▼                     ▼
                ACCEPTED              REVIEW /
                 RESULT              UNCERTAIN
                    │
                    ▼
              Top-3 Predictions
