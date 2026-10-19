"""Select the confidence threshold from VALIDATION coverage-vs-error.

Risk policy (documented, adjustable): choose the lowest threshold such that the
Wilson 95% upper bound of the error rate among accepted predictions is
<= --max_error with >= --min_accepted accepted samples. If none qualifies the
threshold stays null and the app must treat every prediction as low-confidence.
Note: T and the threshold are both fitted on the same validation split, so the
reported validation coverage/error is optimistic; the frozen test split is the
only unbiased check (src/models/evaluate.py).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from leaflens import calibration as cal
from leaflens import config as cfg
from leaflens.calibration_artifact import artifact_path, write_artifact


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{cfg.BASELINE_DIR}/{cfg.CHECKPOINT_NAME}")
    ap.add_argument("--max_error", type=float, default=0.10)
    ap.add_argument("--min_accepted", type=int, default=30)
    a = ap.parse_args(argv)
    ap_ = artifact_path(a.checkpoint)
    if not ap_.exists():
        sys.exit("run src/models/calibrate.py first")
    art = json.loads(ap_.read_text())
    z = np.load(ap_.with_name("val_logits.npz"))
    probs = cal.softmax(z["logits"], art["temperature"])
    thr, rows = cal.choose_threshold(probs, z["labels"], a.max_error, a.min_accepted)
    pd.DataFrame(rows).to_csv(ap_.with_name("threshold_curve_val.csv"), index=False)
    art["threshold"] = thr
    art["threshold_policy"] = {"rule": "lowest threshold with Wilson-95 upper error <= max_error",
                               "max_error": a.max_error, "min_accepted": a.min_accepted,
                               "fitted_on": "val"}
    if thr is not None:
        row = next(r for r in rows if abs(r["threshold"] - thr) < 1e-9)
        art["threshold_validation_result"] = {k: row[k] for k in
                                              ("coverage", "accepted", "accepted_error_rate", "error_upper95")}
    write_artifact(ap_, art)
    print(json.dumps({"threshold": thr, "policy": art["threshold_policy"],
                      "validation": art.get("threshold_validation_result")}, indent=2))
    if thr is None:
        print("WARNING: no threshold meets the policy; predictions will never be accepted", file=sys.stderr)


if __name__ == "__main__":
    main()
