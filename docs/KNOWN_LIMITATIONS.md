# Known limitations
* No checkpoint or pretrained weights available in the audit environment → no accuracy/F1/calibration numbers for the real model were measured. The T=1.267833 and 0.70 threshold are carried over unverified.
* PlantDoc is web-scraped; the audit found cross-split near-duplicates and conflicting labels, so any previously reported test accuracy is likely optimistic and noisy (236 test images).
* Benchmark ≠ field performance; no field set exists.
* Image-quality checks exist but are not part of inference.
* No OOD detection; no mobile, backend, offline-device, latency or quantisation measurements.
* Pad-to-square keeps small leaves small (no leaf detection/cropping).
* Grad-CAM is diagnostic, not proof of correctness, and server-side only.
