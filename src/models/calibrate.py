"""Canonical calibration: fit temperature on VALIDATION logits only.

  python src/models/calibrate.py --checkpoint reports/mobilenetv3_baseline/best_model.pth

Writes next to the checkpoint: calibration.json (bound to the checkpoint
sha256 + class hash), val_logits.npz, reliability_before_after.csv/.png.
Test data is never read.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from leaflens import calibration as cal
from leaflens import config as cfg
from leaflens.calibration_artifact import artifact_path, write_artifact
from leaflens.checkpoint import load_checkpoint
from leaflens.data import ManifestDataset, load_classes, read_manifest


def collect_logits(model, df, c2i, root, batch=64, workers=0):
    ds = ManifestDataset(df, c2i, root, train=False)
    logits, labels = [], []
    with torch.inference_mode():
        for x, y in DataLoader(ds, batch, num_workers=workers):
            logits.append(model(x).numpy().astype(np.float64))
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{cfg.BASELINE_DIR}/{cfg.CHECKPOINT_NAME}")
    ap.add_argument("--manifest_dir", default=cfg.MANIFEST_DIR)
    ap.add_argument("--root", default=".")
    ap.add_argument("--ece_bins", type=int, default=15)
    a = ap.parse_args(argv)
    model, info = load_checkpoint(a.checkpoint)
    classes, c2i = load_classes(a.manifest_dir)
    if classes != info["classes"]:
        sys.exit("manifest classes.json differs from checkpoint classes; refusing to calibrate")
    val = read_manifest(a.manifest_dir, "val")
    logits, labels = collect_logits(model, val, c2i, a.root)
    T = cal.fit_temperature(logits, labels)
    at_bound = cal.temperature_at_bound(T)
    summ = cal.calibration_summary(logits, labels, T, a.ece_bins)
    out = artifact_path(a.checkpoint)
    np.savez(out.with_name("val_logits.npz"), logits=logits, labels=labels)
    rows = []
    for name, t in (("before", 1.0), ("after", T)):
        for r in cal.reliability_table(cal.softmax(logits, t), labels, a.ece_bins):
            rows.append(dict(stage=name, **r))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out.with_name("reliability_before_after.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for name in ("before", "after"):
            r = [x for x in rows if x["stage"] == name and x["count"]]
            ax.plot([x["mean_confidence"] for x in r], [x["accuracy"] for x in r], "o-",
                    label=f"{name} (ECE {summ[name]['ece']:.3f})")
        ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability (validation)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out.with_name("reliability_before_after.png"), dpi=130)
    except Exception as e:  # plotting must never block the artifact
        print(f"warning: reliability plot skipped: {e}", file=sys.stderr)
    art = {"schema_version": 1, "temperature": T, "temperature_at_search_bound": bool(at_bound),
           "fitted_on": "val", "n_val": int(len(labels)),
           "checkpoint_sha256": info["sha256"], "class_hash": info["class_hash"],
           "classes": info["classes"],
           "val_manifest_sha256": hashlib.sha256(
               (Path(a.manifest_dir) / "val_manifest.csv").read_bytes()).hexdigest(),
           "random_init": bool(info["meta"].get("random_init")),
           "metrics": summ, "threshold": None, "threshold_policy": None}
    write_artifact(out, art)
    print(json.dumps({"temperature": T, "at_bound": at_bound, **{k: summ[k] for k in summ}}, indent=2))
    if at_bound:
        print("WARNING: temperature hit the search bound; calibration unreliable", file=sys.stderr)


if __name__ == "__main__":
    main()
