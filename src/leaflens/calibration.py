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
    """Fit T by minimising validation NLL (golden-section search; NLL is
    unimodal in T). Validation data only -- never pass test logits."""
    val_logits = np.asarray(val_logits)
    val_labels = np.asarray(val_labels)
    f = lambda t: nll(softmax(val_logits, t), val_labels)
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(100):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float((a + b) / 2)


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
