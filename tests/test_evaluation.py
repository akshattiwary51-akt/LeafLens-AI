import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score

from leaflens import checkpoint as ck
from leaflens import config as cfg
from leaflens import metrics as M

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
import calibrate as calib_cli  # noqa: E402
import evaluate as eval_cli  # noqa: E402
import run_experiments as runner  # noqa: E402
import select_threshold as thr_cli  # noqa: E402
import train as train_mod  # noqa: E402


def test_metrics_match_sklearn_and_per_class_sums():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 300); p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 4, 300))
    r = M.classification_metrics(y, p, list("abcd"))
    assert r["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert r["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert r["weighted_f1"] == pytest.approx(f1_score(y, p, average="weighted"))
    assert sum(c["support"] for c in r["per_class"]) == 300
    assert np.trace(np.array(r["confusion_matrix"])) == round(r["accuracy"] * 300)
    assert sum(c["errors"] for c in r["per_class"]) == 300 - np.trace(np.array(r["confusion_matrix"]))


def test_metrics_handle_absent_class_without_nan():
    r = M.classification_metrics([0, 0, 1], [0, 0, 0], list("abc"))  # class c never appears
    assert all(np.isfinite(c["f1"]) for c in r["per_class"])


def test_bootstrap_is_seeded_and_contains_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200); p = np.where(rng.random(200) < 0.8, y, 0)
    a = M.bootstrap_ci(y, p, 3, n_boot=300, seed=5)
    b = M.bootstrap_ci(y, p, 3, n_boot=300, seed=5)
    assert a == b
    acc = (y == p).mean()
    assert a["accuracy_ci95"][0] <= acc <= a["accuracy_ci95"][1]
    d = M.paired_bootstrap_diff(y, p, p, 3, n_boot=100)
    assert d["acc_diff_ci95"] == [0.0, 0.0]


@pytest.fixture
def smoke_run(tmp_path, tiny_dataset):
    torch.manual_seed(0)
    out = tmp_path / "run"
    ck.save_checkpoint(out / cfg.CHECKPOINT_NAME, ck.build_model(3, False),
                       tiny_dataset["classes"], {"random_init": False})  # pretend a real run
    c = str(out / cfg.CHECKPOINT_NAME)
    args = ["--manifest_dir", str(tiny_dataset["manifests"]), "--root", str(tiny_dataset["root"])]
    calib_cli.main(["--checkpoint", c] + args)
    thr_cli.main(["--checkpoint", c, "--max_error", "0.99", "--min_accepted", "1"])
    return dict(ckpt=c, args=args, ds=tiny_dataset, dir=out)


def test_val_evaluation_outputs(smoke_run):
    eval_cli.main(["--checkpoint", smoke_run["ckpt"], "--split", "val"] + smoke_run["args"])
    d = smoke_run["dir"]
    r = json.loads((d / "val_evaluation.json").read_text())
    assert r["n"] == 6 and "macro_f1_ci95" in r and "selective" in r
    assert (d / "val_evaluation_per_class.csv").exists() and (d / "val_evaluation_confusion_matrix.csv").exists()


def _freeze(ds):
    import hashlib
    md = ds["manifests"]
    (md / "TEST_FROZEN.sha256").write_text(hashlib.sha256((md / "test_manifest.csv").read_bytes()).hexdigest())


def test_test_split_guards(smoke_run):
    base = ["--checkpoint", smoke_run["ckpt"], "--split", "test"] + smoke_run["args"]
    with pytest.raises(SystemExit, match="frozen"):
        eval_cli.main(base + ["--final"])  # no frozen hash yet
    _freeze(smoke_run["ds"])
    with pytest.raises(SystemExit, match="--final"):
        eval_cli.main(base)
    eval_cli.main(base + ["--final"])
    assert (smoke_run["dir"] / "test_evaluation.json").exists()
    with pytest.raises(SystemExit, match="once per checkpoint"):
        eval_cli.main(base + ["--final"])
    # tampering with the test manifest after freezing is refused
    t = smoke_run["ds"]["manifests"] / "test_manifest.csv"
    pd.read_csv(t).iloc[:-1].to_csv(t, index=False)
    (smoke_run["dir"] / "test_evaluation.json").unlink()
    with pytest.raises(SystemExit, match="frozen"):
        eval_cli.main(base + ["--final"])


def test_test_refused_for_random_init_and_without_threshold_policy(tmp_path, tiny_dataset):
    out = tmp_path / "r"
    ck.save_checkpoint(out / cfg.CHECKPOINT_NAME, ck.build_model(3, False),
                       tiny_dataset["classes"], {"random_init": True})
    c = str(out / cfg.CHECKPOINT_NAME)
    args = ["--manifest_dir", str(tiny_dataset["manifests"]), "--root", str(tiny_dataset["root"])]
    calib_cli.main(["--checkpoint", c] + args)
    _freeze(tiny_dataset)
    with pytest.raises(SystemExit, match="threshold"):  # policy not yet selected
        # make it non-random so we reach the threshold guard
        d = torch.load(c, weights_only=True); d["meta"]["random_init"] = False; torch.save(d, c)
        _, info = ck.load_checkpoint(c)
        calib_cli.main(["--checkpoint", c] + args)
        eval_cli.main(["--checkpoint", c, "--split", "test", "--final"] + args)
    d = torch.load(c, weights_only=True); d["meta"]["random_init"] = True; torch.save(d, c)
    calib_cli.main(["--checkpoint", c] + args)
    with pytest.raises(SystemExit, match="random-init"):
        eval_cli.main(["--checkpoint", c, "--split", "test", "--final"] + args)


def test_runner_blocks_without_pretrained_weights(tmp_path, tiny_dataset, monkeypatch):
    monkeypatch.setattr(train_mod, "pretrained_available", lambda: (False, "offline"))
    with pytest.raises(SystemExit, match="BLOCKED"):
        runner.main(["--manifest_dir", str(tiny_dataset["manifests"]), "--root",
                     str(tiny_dataset["root"]), "--base_dir", str(tmp_path / "exp")])
    assert not (tmp_path / "exp").exists()


def test_runner_smoke_runs_all_three_experiments_and_never_touches_test(tmp_path, tiny_dataset, monkeypatch):
    monkeypatch.setattr(train_mod, "pretrained_available", lambda: (False, "offline"))
    monkeypatch.setattr(train_mod, "build_model", lambda n, pretrained: ck.build_model(n, False))
    monkeypatch.chdir(tmp_path)  # canonical baseline dir is relative
    runner.main(["--manifest_dir", str(tiny_dataset["manifests"]), "--root", str(tiny_dataset["root"]),
                 "--base_dir", str(tmp_path / "exp"), "--seeds", "1,2", "--epochs", "1",
                 "--allow_random_init"])
    t = pd.read_csv(tmp_path / "exp" / "experiment_table.csv")
    assert set(t.experiment) == set(cfg.EXPERIMENTS) and (t.runs == 2).all()
    sel = json.loads((tmp_path / "exp" / "selection.json").read_text())
    assert sel["smoke_only"] is True and sel["test_evaluated"] is False
    assert not list(tmp_path.rglob("test_evaluation.json"))
