# Model card — LeafLens MobileNetV3-Small (PlantDoc, 27 classes)
* Architecture: torchvision MobileNetV3-Small, ImageNet init, 27-way head, 224×224 input.
* Data: PlantDoc (CC BY 4.0), web-sourced field-ish photos; 1782/446/236 train/val/test; known leakage and label noise (docs/REPOSITORY_AUDIT.md D9).
* Intended use: research prototype for screening leaf photos; **not** a diagnostic tool and gives no treatment advice.
* Metrics: **none verified** — the checkpoint was not available to this audit.
* Calibration: single temperature (T=1.267833 reported by the original author, unverified); threshold 0.70 untuned.
* Limits: see docs/KNOWN_LIMITATIONS.md.
