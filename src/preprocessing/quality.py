"""
Image-quality scoring — Milestone 1 §12.

Runs BEFORE training. Flags images too poor (blurry, dark, low-res) to be
trustworthy for either training or field inference, so the eventual mobile
app can also reuse this to tell a farmer "please retake the photo."

No deep-learning dependency — pure OpenCV/NumPy, so it runs anywhere.
"""
import cv2
import numpy as np


def blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian. Higher = sharper. Rule-of-thumb threshold ~100
    for natural leaf photos, but should be recalibrated once we have a small
    hand-labeled 'blurry vs sharp' set from our own target dataset."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_score(gray: np.ndarray) -> float:
    return float(gray.mean())


def contrast_score(gray: np.ndarray) -> float:
    return float(gray.std())


def exposure_flags(gray: np.ndarray, dark_thresh=40, bright_thresh=215,
                    clip_frac_thresh=0.35) -> dict:
    dark_frac = float((gray < dark_thresh).mean())
    bright_frac = float((gray > bright_thresh).mean())
    return {
        "underexposed": dark_frac > clip_frac_thresh,
        "overexposed": bright_frac > clip_frac_thresh,
        "dark_pixel_fraction": dark_frac,
        "bright_pixel_fraction": bright_frac,
    }


def assess_quality(image_path: str,
                    min_resolution: int = 224,
                    blur_thresh: float = 100.0) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"quality": "UNREADABLE", "reasons": ["file could not be decoded"]}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = blur_score(gray)
    brightness = brightness_score(gray)
    contrast = contrast_score(gray)
    exposure = exposure_flags(gray)

    reasons = []
    if min(h, w) < min_resolution:
        reasons.append(f"resolution too low ({w}x{h}, need >= {min_resolution}px on short side)")
    if blur < blur_thresh:
        reasons.append(f"excessive blur (sharpness score {blur:.1f} < {blur_thresh})")
    if exposure["underexposed"]:
        reasons.append("underexposed / insufficient leaf visibility")
    if exposure["overexposed"]:
        reasons.append("overexposed / blown-out highlights")
    if contrast < 15:
        reasons.append("very low contrast — leaf may not be distinguishable from background")

    quality = "GOOD" if not reasons else ("POOR" if len(reasons) >= 2 else "FAIR")

    return {
        "quality": quality,
        "reasons": reasons,
        "width": w,
        "height": h,
        "blur_score": round(blur, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        **exposure,
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    args = ap.parse_args()
    print(json.dumps(assess_quality(args.image_path), indent=2))
