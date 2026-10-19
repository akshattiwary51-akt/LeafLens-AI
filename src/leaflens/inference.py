"""Single documented inference contract for LeafLens.

Contract (must match training; see docs/ARCHITECTURE.md):
  1. Decode with PIL, apply EXIF orientation, convert to RGB.
  2. Aspect-preserving resize (long side -> 224, LANCZOS) and centre on a
     black 224x224 canvas -- identical to src/data/preprocess.py, which
     produced every training/validation/test image.
  3. ToTensor ([0,1], CHW float32), normalise with ImageNet mean/std.
  4. Model returns raw LOGITS. Temperature is applied exactly once
     (logits / T) followed by exactly one softmax.
  5. Decision: invalid image -> "invalid_image"; confidence >= threshold ->
     "accepted"; otherwise "low_confidence" (never a definitive diagnosis).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_TEMPERATURE = 1.267833  # from the original (unavailable) checkpoint
DEFAULT_THRESHOLD = 0.70        # unvalidated default; tune on validation data
MIN_SIDE = 32                   # reject degenerate inputs


class InvalidImageError(ValueError):
    pass


def load_image(path_or_file):
    try:
        with Image.open(path_or_file) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.load()
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise InvalidImageError(f"cannot decode image: {e}") from e
    if min(im.size) < MIN_SIDE:
        raise InvalidImageError(f"image too small: {im.size}")
    return im


def pad_to_square(im, size=IMAGE_SIZE):
    """Same geometry as src/data/preprocess.py::preprocess_image."""
    im = im.copy()
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def to_array(im):
    """PIL RGB 224x224 -> float32 NCHW (1,3,224,224), normalised."""
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def preprocess(path_or_file):
    return to_array(pad_to_square(load_image(path_or_file)))


@dataclass
class Policy:
    threshold: float = DEFAULT_THRESHOLD
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self):
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


def decide(logits, idx_to_class, policy, top_k=3):
    """logits: shape (C,) raw model output. Returns a result dict."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.shape[0] != len(idx_to_class):
        raise ValueError(
            f"model has {logits.shape[0]} outputs but {len(idx_to_class)} labels")
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits")
    z = logits / policy.temperature
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    order = np.argsort(-p, kind="stable")[:top_k]
    top = [{"class": idx_to_class[int(i)], "probability": round(float(p[i]), 4)}
           for i in order]
    conf = float(p[order[0]])
    return {
        "status": "accepted" if conf >= policy.threshold else "low_confidence",
        "prediction": top[0]["class"],
        "confidence": round(conf, 4),
        "top_predictions": top,
        "temperature": policy.temperature,
        "threshold": policy.threshold,
        "disclaimer": "Model output, not an expert diagnosis.",
    }


def invalid_result(reason):
    return {"status": "invalid_image", "prediction": None, "confidence": None,
            "top_predictions": [], "reason": str(reason),
            "disclaimer": "Model output, not an expert diagnosis."}
