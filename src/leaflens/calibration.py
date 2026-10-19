"""Calibration metrics and temperature scaling (logits in, never probabilities)."""
import numpy as np


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs, labels):
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def brier(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ece(probs, labels, n_bins=15):
    """Top-label ECE with equal-width bins on (0, 1]."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(val_logits, val_labels, lo=0.05, hi=10.0):
    """Fit T by minimising validation NLL over log T in [log lo, log hi]
    (golden-section; T = exp(u) is positive by construction). Validation
    data only -- never pass test logits. Returns float T; see
    ``temperature_at_bound`` to detect an unreliable fit."""
    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels)
    if val_logits.ndim != 2 or len(val_logits) == 0 or len(val_logits) != len(val_labels):
        raise ValueError("need non-empty (N, C) logits with N labels")
    if not np.isfinite(val_logits).all():
        raise ValueError("non-finite validation logits")
    f = lambda u: nll(softmax(val_logits, np.exp(u)), val_labels)
    g = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(100):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(min(hi, max(lo, np.exp((a + b) / 2))))


def temperature_at_bound(t, lo=0.05, hi=10.0, tol=1e-2):
    return t <= lo * (1 + tol) or t >= hi * (1 - tol)


def reliability_table(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else None,
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows


def calibration_summary(logits, labels, temperature, n_bins=15):
    """NLL / ECE / Brier / accuracy for T=1 (before) and the fitted T (after)."""
    out = {}
    for name, t in (("before", 1.0), ("after", temperature)):
        p = softmax(logits, t)
        out[name] = {"temperature": float(t), "nll": nll(p, labels),
                     "ece": ece(p, labels, n_bins), "brier": brier(p, labels),
                     "accuracy": float((p.argmax(1) == labels).mean()), "ece_bins": n_bins}
    return out


def wilson_upper(errors, n, z=1.96):
    """Upper bound of the Wilson score interval for an error proportion."""
    if n == 0:
        return 1.0
    p = errors / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (centre + half) / den))


def choose_threshold(probs, labels, max_error=0.10, min_accepted=30, z=1.96):
    """Risk policy: the LOWEST threshold (=highest coverage) whose accepted
    predictions have a Wilson upper-bound error <= max_error with at least
    ``min_accepted`` accepted samples. Candidates are the observed
    confidences. Returns (threshold or None, curve rows). None => no
    threshold on this data meets the policy (callers must not accept)."""
    conf = probs.max(axis=1)
    wrong = probs.argmax(axis=1) != labels
    cands = np.unique(np.round(conf, 6))
    rows, best = [], None
    for t in cands:
        m = conf >= t
        n = int(m.sum())
        e = int(wrong[m].sum())
        ub = wilson_upper(e, n, z)
        ok = n >= min_accepted and ub <= max_error
        rows.append({"threshold": float(t), "coverage": n / len(conf), "accepted": n,
                     "accepted_errors": e, "accepted_error_rate": e / n if n else None,
                     "error_upper95": ub, "meets_policy": bool(ok)})
        if ok and best is None:
            best = float(t)
    return best, rows


def coverage_error_curve(probs, labels, thresholds):
    """For each threshold: coverage and error rate among accepted samples."""
    conf = probs.max(axis=1)
    wrong = probs.argmax(axis=1) != labels
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({
            "threshold": float(t),
            "coverage": float(m.mean()),
            "accepted": int(m.sum()),
            "accepted_error_rate": float(wrong[m].mean()) if m.any() else None,
        })
    return rows
