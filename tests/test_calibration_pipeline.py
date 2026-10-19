import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from leaflens import calibration as cal
from leaflens import checkpoint as ck
from leaflens import config as cfg
from leaflens.calibration_artifact import CalibrationError, artifact_path, load_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
import calibrate as calib_cli  # noqa: E402
import predict as predict_cli  # noqa: E402
import select_threshold as thr_cli  # noqa: E402


@pytest.fixture
def run(tmp_path, tiny_dataset):
    torch.manual_seed(0)
    m = ck.build_model(3, pretrained=False)
    out = tmp_path / "run"
    ck.save_checkpoint(out / cfg.CHECKPOINT_NAME, m, tiny_dataset["classes"], {"random_init": True})
    return dict(ckpt=out / cfg.CHECKPOINT_NAME, ds=tiny_dataset)


def _calibrate(run):
    calib_cli.main(["--checkpoint", str(run["ckpt"]), "--manifest_dir", str(run["ds"]["manifests"]),
                    "--root", str(run["ds"]["root"])])


def test_temperature_positive_by_construction_and_bounded():
    rng = np.random.default_rng(0)
    for scale in (0.01, 1.0, 100.0):  # extreme under/over-confidence
        lg = rng.normal(size=(200, 4)) * scale
        y = rng.integers(0, 4, 200)
        T = cal.fit_temperature(lg, y)
        assert 0.05 <= T <= 10.0 and T > 0
    assert cal.temperature_at_bound(10.0) and not cal.temperature_at_bound(1.3)


def test_fit_rejects_nonfinite_and_empty():
    with pytest.raises(ValueError):
        cal.fit_temperature(np.array([[np.nan, 1.0]]), np.array([0]))
    with pytest.raises(ValueError):
        cal.fit_temperature(np.zeros((0, 3)), np.zeros(0, int))


def test_calibration_artifact_is_bound_to_checkpoint_and_val_only(run):
    _calibrate(run)
    art = json.loads(artifact_path(run["ckpt"]).read_text())
    assert art["fitted_on"] == "val" and art["n_val"] == 6  # 3 classes x 2 val images
    assert art["temperature"] > 0 and art["random_init"] is True
    for k in ("nll", "ece", "brier"):
        assert k in art["metrics"]["before"] and k in art["metrics"]["after"]
    model, info = ck.load_checkpoint(run["ckpt"])
    pol, _ = load_policy(run["ckpt"], info)
    assert pol.calibrated and pol.temperature == pytest.approx(art["temperature"])
    # fitting never touched the test split
    assert not any("test" in str(p) for p in [art["val_manifest_sha256"]])


def test_mismatched_or_missing_artifact_is_an_error(run, tmp_path):
    _, info = ck.load_checkpoint(run["ckpt"])
    with pytest.raises(CalibrationError, match="missing"):
        load_policy(run["ckpt"], info)
    pol, art = load_policy(run["ckpt"], info, require=False)
    assert art is None and not pol.calibrated and pol.threshold is None
    _calibrate(run)
    p = artifact_path(run["ckpt"])
    good = json.loads(p.read_text())
    for field, value, msg in [("checkpoint_sha256", "0" * 64, "different checkpoint"),
                              ("class_hash", "deadbeef", "class mapping"),
                              ("fitted_on", "test", "validation"),
                              ("temperature", -1.0, "temperature")]:
        bad = {**good, field: value}
        p.write_text(json.dumps(bad))
        with pytest.raises(CalibrationError, match=msg):
            load_policy(run["ckpt"], info)
    p.write_text("{not json")
    with pytest.raises(CalibrationError, match="unreadable"):
        load_policy(run["ckpt"], info)


def test_changed_checkpoint_invalidates_artifact(run):
    _calibrate(run)
    m2 = ck.build_model(3, pretrained=False)  # different random weights, same path
    torch.manual_seed(123)
    m2 = ck.build_model(3, pretrained=False)
    ck.save_checkpoint(run["ckpt"], m2, run["ds"]["classes"], {"random_init": True})
    _, info = ck.load_checkpoint(run["ckpt"])
    with pytest.raises(CalibrationError, match="different checkpoint"):
        load_policy(run["ckpt"], info)


def test_choose_threshold_meets_policy_and_reports_infeasible():
    rng = np.random.default_rng(1)
    n, c = 600, 4
    labels = rng.integers(0, c, n)
    logits = rng.normal(size=(n, c))
    correct = rng.random(n) < 0.9  # confident => right
    boost = np.where(correct, rng.uniform(2, 8, n), rng.uniform(0.2, 1.5, n))  # confidence is informative
    logits[np.arange(n), np.where(correct, labels, (labels + 1) % c)] += boost
    probs = cal.softmax(logits)
    thr, rows = cal.choose_threshold(probs, labels, max_error=0.10, min_accepted=30)
    assert thr is not None
    row = next(r for r in rows if r["threshold"] == pytest.approx(thr))
    assert row["error_upper95"] <= 0.10 and row["accepted"] >= 30
    # lower thresholds that were tried do not meet the policy (it is the LOWEST feasible)
    assert all(not r["meets_policy"] for r in rows if r["threshold"] < thr)
    none_thr, _ = cal.choose_threshold(probs, (labels + 1) % c, max_error=0.01, min_accepted=30)
    assert none_thr is None


def test_wilson_bound_properties():
    assert cal.wilson_upper(0, 0) == 1.0
    assert cal.wilson_upper(0, 100) < 0.05
    assert cal.wilson_upper(5, 100) > 0.05
    assert cal.wilson_upper(100, 100) == pytest.approx(1.0, abs=1e-9)


def test_threshold_cli_uses_validation_logits_and_writes_policy(run):
    _calibrate(run)
    thr_cli.main(["--checkpoint", str(run["ckpt"]), "--max_error", "0.5", "--min_accepted", "1"])
    art = json.loads(artifact_path(run["ckpt"]).read_text())
    assert art["threshold_policy"]["fitted_on"] == "val"
    assert art["threshold"] is None or 0 < art["threshold"] <= 1
    assert artifact_path(run["ckpt"]).with_name("threshold_curve_val.csv").exists()


def test_predict_cli_requires_artifact_then_runs(run, tmp_path, capsys):
    img = run["ds"]["root"] / "img/test/Class A/10.jpg"
    with pytest.raises(SystemExit) as e:
        predict_cli.main(["--image", str(img), "--checkpoint", str(run["ckpt"])])
    assert "calibration artifact missing" in str(e.value)
    predict_cli.main(["--image", str(img), "--checkpoint", str(run["ckpt"]), "--allow_uncalibrated"])
    out = json.loads(capsys.readouterr().out)
    assert out["calibrated"] is False and out["status"] == "low_confidence"  # never accepted uncalibrated
    _calibrate(run)
    thr_cli.main(["--checkpoint", str(run["ckpt"]), "--max_error", "0.99", "--min_accepted", "1"])
    capsys.readouterr()
    predict_cli.main(["--image", str(img), "--checkpoint", str(run["ckpt"])])
    out = json.loads(capsys.readouterr().out)
    assert out["calibrated"] is True
    with pytest.raises(SystemExit):
        predict_cli.main(["--image", str(tmp_path / "missing.png"), "--checkpoint", str(run["ckpt"])])
