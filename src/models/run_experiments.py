"""Controlled experiment runner: standard vs weighted_sampler vs class_weighted_loss.

For each experiment (identical data, seed(s), augmentation, optimiser, schedule,
early stopping; only the one named factor differs): train -> calibrate ->
select threshold (all validation) -> record validation metrics.
The winner is chosen by the predeclared metric (mean val macro-F1; ties -> lower val NLL),
and only that model is evaluated once on the frozen test split.

The 'standard' run is the project baseline and is written to the canonical
reports/mobilenetv3_baseline. If ImageNet weights are unavailable the runner
STOPS before writing anything and prints the next command.

  python src/models/run_experiments.py --seeds 42,43,44
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import calibrate
import evaluate as evaluate_mod
import select_threshold
import train
from leaflens import config as cfg
from leaflens.calibration_artifact import load_policy
from leaflens.checkpoint import load_checkpoint


def run_dir(base, exp, seed, multi):
    if exp == "standard" and not multi:
        return Path(cfg.BASELINE_DIR)
    return Path(base) / (f"{exp}_seed{seed}" if multi else exp)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default=cfg.MANIFEST_DIR)
    ap.add_argument("--root", default=".")
    ap.add_argument("--base_dir", default="reports/experiments")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--allow_random_init", action="store_true", help="pipeline smoke test only")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    multi = len(seeds) > 1
    ok, why = train.pretrained_available()
    if not ok and not a.allow_random_init:
        sys.exit("BLOCKED: pretrained MobileNetV3-Small weights unavailable (" + why + ").\n"
                 "Next: python src/models/run_experiments.py --seeds 42,43,44   (on a machine with "
                 "network access to download.pytorch.org)")
    rows = []
    for exp in cfg.EXPERIMENTS:
        for seed in seeds:
            d = run_dir(a.base_dir, exp, seed, multi)
            args = ["--manifest_dir", a.manifest_dir, "--root", a.root, "--experiment", exp,
                    "--output_dir", str(d), "--seed", str(seed), "--epochs", str(a.epochs)]
            if a.allow_random_init:
                args.append("--allow_random_init")
            train.main(args)
            ck = str(d / cfg.CHECKPOINT_NAME)
            calibrate.main(["--checkpoint", ck, "--manifest_dir", a.manifest_dir, "--root", a.root])
            select_threshold.main(["--checkpoint", ck])
            evaluate_mod.main(["--checkpoint", ck, "--manifest_dir", a.manifest_dir,
                               "--root", a.root, "--split", "val"])
            v = json.loads((d / "val_evaluation.json").read_text())
            meta = json.loads((d / "train_meta.json").read_text())
            art = json.loads((d / cfg.CALIBRATION_NAME).read_text())
            rows.append(dict(experiment=exp, seed=seed, dir=str(d),
                             val_accuracy=v["accuracy"], val_macro_f1=v["macro_f1"],
                             val_weighted_f1=v["weighted_f1"],
                             val_nll_after=v["calibration"]["after"]["nll"],
                             val_ece_after=v["calibration"]["after"]["ece"],
                             temperature=art["temperature"], threshold=art["threshold"],
                             best_epoch=meta["best_epoch"], train_seconds=meta["train_seconds"],
                             smoke_only=meta["random_init"]))
    tab = pd.DataFrame(rows)
    out = Path(a.base_dir); out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "experiment_runs.csv", index=False)
    agg = tab.groupby("experiment").agg(
        val_macro_f1_mean=("val_macro_f1", "mean"), val_macro_f1_sd=("val_macro_f1", "std"),
        val_accuracy_mean=("val_accuracy", "mean"), val_nll_mean=("val_nll_after", "mean"),
        runs=("seed", "count")).reset_index()
    agg.to_csv(out / "experiment_table.csv", index=False)
    best = agg.sort_values(["val_macro_f1_mean", "val_nll_mean"], ascending=[False, True]).iloc[0]
    smoke = bool(tab.smoke_only.any())
    sel = {"selected_experiment": best.experiment, "criterion": cfg.SELECTION_METRIC + " (mean over seeds)",
           "smoke_only": smoke, "test_evaluated": False}
    if smoke:
        sel["note"] = "random-init smoke run: no result, test split NOT evaluated"
    else:
        cand = tab[tab.experiment == best.experiment].sort_values("val_macro_f1", ascending=False).iloc[0]
        evaluate_mod.main(["--checkpoint", str(Path(cand.dir) / cfg.CHECKPOINT_NAME), "--manifest_dir",
                           a.manifest_dir, "--root", a.root, "--split", "test", "--final"])
        sel.update(test_evaluated=True, checkpoint_dir=cand.dir)
    (out / "selection.json").write_text(json.dumps(sel, indent=2))
    print(agg.to_string(index=False)); print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
