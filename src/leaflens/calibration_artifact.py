"""Checkpoint-bound calibration artifact (calibration.json next to the checkpoint)."""
import json
from pathlib import Path

from . import config as cfg
from .inference import Policy


class CalibrationError(ValueError):
    pass


def artifact_path(checkpoint_path):
    return Path(checkpoint_path).with_name(cfg.CALIBRATION_NAME)


def write_artifact(path, data):
    Path(path).write_text(json.dumps(data, indent=2))


def load_policy(checkpoint_path, info, require=True):
    """Return (Policy, artifact dict|None). Missing/mismatched artifact raises
    CalibrationError when ``require`` else yields an explicit UNCALIBRATED policy."""
    p = artifact_path(checkpoint_path)
    if not p.exists():
        if require:
            raise CalibrationError(f"calibration artifact missing: {p} (run src/models/calibrate.py)")
        return Policy(threshold=None, temperature=1.0, calibrated=False), None
    try:
        art = json.loads(p.read_text())
        T = float(art["temperature"])
        bound_sha, bound_cls = art["checkpoint_sha256"], art["class_hash"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise CalibrationError(f"calibration artifact unreadable: {e}") from e
    if bound_sha != info["sha256"]:
        raise CalibrationError("calibration artifact belongs to a different checkpoint (sha256 mismatch)")
    if bound_cls != info["class_hash"]:
        raise CalibrationError("calibration artifact class mapping differs from checkpoint")
    if art.get("fitted_on") != "val":
        raise CalibrationError("calibration artifact was not fitted on validation data")
    thr = art.get("threshold")
    try:
        return Policy(threshold=thr, temperature=T, calibrated=True), art
    except ValueError as e:
        raise CalibrationError(str(e)) from e
