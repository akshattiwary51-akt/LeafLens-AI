"""Canonical evaluation (val or the FROZEN test split) for a checkpoint.

  python src/models/evaluate.py --checkpoint <ckpt> --split val
  python src/models/evaluate.py --checkpoint <ckpt> --split test --final

Test guards: the manifest must match TEST_FROZEN.sha256; the calibration
artifact (temperature + threshold, fitted on validation) must exist and match
the checkpoint; ``--final`` is required; an existing test_evaluation.json is
never overwritten (no repeated peeking). Smoke (random-init) checkpoints are
refused for the test split.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from leaflens import calibration as cal
from leaflens import config as cfg
from leaflens import metrics as M
from leaflens.calibration_artifact import artifact_path, load_policy
from leaflens.checkpoint import load_checkpoint
from leaflens.data import load_classes, read_manifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import collect_logits  # noqa: E402


def evaluate(model, info, manifest_dir, split, root, policy, ece_bins=15, n_boot=1000):
    classes, c2i = load_classes(manifest_dir)
    df = read_manifest(manifest_dir, split)
    logits, labels = collect_logits(model, df, c2i, root)
    pred = logits.argmax(1)
    res = M.classification_metrics(labels, pred, classes)
    res.update(M.bootstrap_ci(labels, pred, len(classes), n_boot))
    res["calibration"] = cal.calibration_summary(logits, labels, policy.temperature, ece_bins)
    probs = cal.softmax(logits, policy.temperature)
    conf, wrong = probs.max(1), pred != labels
    if policy.threshold is not None:
        acc_mask = conf >= policy.threshold
        res["selective"] = {"threshold": policy.threshold, "coverage": float(acc_mask.mean()),
                            "accepted": int(acc_mask.sum()),
                            "accepted_error_rate": float(wrong[acc_mask].mean()) if acc_mask.any() else None}
    else:
        res["selective"] = {"threshold": None, "note": "no validated threshold; nothing accepted"}
    res["curve"] = cal.coverage_error_curve(probs, labels, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    return res, df, pred, labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{cfg.BASELINE_DIR}/{cfg.CHECKPOINT_NAME}")
    ap.add_argument("--manifest_dir", default=cfg.MANIFEST_DIR)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required for the test split")
    a = ap.parse_args(argv)
    model, info = load_checkpoint(a.checkpoint)
    policy, art = load_policy(a.checkpoint, info, require=True)
    out_dir = Path(a.checkpoint).parent
    if a.split == "test":
        md = Path(a.manifest_dir)
        frozen = (md / "TEST_FROZEN.sha256").read_text().strip() if (md / "TEST_FROZEN.sha256").exists() else None
        if frozen != hashlib.sha256((md / "test_manifest.csv").read_bytes()).hexdigest():
            sys.exit("REFUSED: test manifest is not the frozen one")
        if not a.final:
            sys.exit("REFUSED: pass --final to evaluate on the frozen test split (single use)")
        if info["meta"].get("random_init"):
            sys.exit("REFUSED: random-init smoke checkpoint must not be evaluated on test")
        if art.get("threshold_policy") is None:
            sys.exit("REFUSED: run select_threshold.py first (threshold must be fixed before test)")
        target = out_dir / "test_evaluation.json"
        if target.exists():
            sys.exit(f"REFUSED: {target} exists; the test split is evaluated once per checkpoint")
    res, df, pred, labels = evaluate(model, info, a.manifest_dir, a.split, a.root, policy)
    classes = info["classes"]
    stem = f"{a.split}_evaluation"
    res["checkpoint_sha256"] = info["sha256"]
    res["split"] = a.split
    res["random_init_checkpoint"] = bool(info["meta"].get("random_init"))
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(res["per_class"]).to_csv(out_dir / f"{stem}_per_class.csv", index=False)
    pd.DataFrame(res["confusion_matrix"], index=classes, columns=classes).to_csv(
        out_dir / f"{stem}_confusion_matrix.csv")
    np.save(out_dir / f"{stem}_predictions.npy", np.stack([labels, pred]))
    print(json.dumps({k: res[k] for k in ("n", "accuracy", "accuracy_ci95", "macro_f1",
                                           "macro_f1_ci95", "weighted_f1", "selective")}, indent=2))


if __name__ == "__main__":
    main()
