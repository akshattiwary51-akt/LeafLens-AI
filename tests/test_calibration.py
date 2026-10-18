import numpy as np
import pytest

from leaflens import calibration as cal


def _synthetic(n=4000, c=5, scale=3.0, seed=0):
    """Overconfident model: true T is `scale`."""
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(size=(n, c))
    probs = cal.softmax(true_logits)
    labels = np.array([rng.choice(c, p=p) for p in probs])
    return true_logits * scale, labels  # inflate logits -> overconfident


def test_fit_temperature_recovers_scale():
    logits, labels = _synthetic(scale=3.0)
    assert cal.fit_temperature(logits, labels) == pytest.approx(3.0, rel=0.1)


def test_temperature_improves_nll_ece_on_heldout():
    lv, yv = _synthetic(seed=1)
    lt, yt = _synthetic(seed=2)
    T = cal.fit_temperature(lv, yv)
    before, after = cal.softmax(lt), cal.softmax(lt, T)
    assert cal.nll(after, yt) < cal.nll(before, yt)
    assert cal.ece(after, yt) < cal.ece(before, yt)
    assert cal.brier(after, yt) < cal.brier(before, yt)


def test_metrics_edge_cases():
    p = np.eye(3)[[0, 1, 2]]
    y = np.array([0, 1, 2])
    assert cal.ece(p, y) == pytest.approx(0.0)
    assert cal.brier(p, y) == pytest.approx(0.0)
    assert cal.nll(p, y) == pytest.approx(0.0, abs=1e-9)
    assert cal.ece(np.full((4, 4), 0.25), np.array([0, 1, 2, 3])) <= 0.25


def test_coverage_error_curve_monotone_coverage():
    lg, y = _synthetic(n=1000, scale=1.0)
    rows = cal.coverage_error_curve(cal.softmax(lg), y, [0.3, 0.5, 0.7, 0.9])
    cov = [r["coverage"] for r in rows]
    assert cov == sorted(cov, reverse=True)
    errs = [r["accepted_error_rate"] for r in rows if r["accepted_error_rate"] is not None]
    assert errs[0] >= errs[-1]
