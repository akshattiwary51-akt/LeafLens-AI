"""Classification metrics with uncertainty (seeded bootstrap)."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, classes):
    n = len(classes)
    labels = list(range(n))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                   average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = [{"class": classes[i], "precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f[i]), "support": int(s[i]),
                  "errors": int(s[i] - cm[i, i])} for i in labels]
    return {"n": int(len(y_true)), "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
            "macro_f1": float(f.mean()), "weighted_f1": float(f_w),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def _macro_f1(y, p, n):
    _, _, f, _ = precision_recall_fscore_support(y, p, labels=list(range(n)), zero_division=0)
    return float(f.mean())


def bootstrap_ci(y_true, y_pred, n_classes, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CIs for accuracy and macro-F1 (resampling images; images from
    one duplicate group are already kept inside one split, but samples within
    a split are treated as independent -- a simplification, see EVALUATION.md)."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        accs.append((y[i] == p[i]).mean())
        f1s.append(_macro_f1(y[i], p[i], n_classes))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"accuracy_ci95": [float(x) for x in np.percentile(accs, q)],
            "macro_f1_ci95": [float(x) for x in np.percentile(f1s, q)],
            "n_boot": n_boot, "seed": seed}


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_classes, n_boot=1000, seed=0):
    """CI of metric(A) - metric(B) on the same samples."""
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    d_acc, d_f1 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d_acc.append((a[i] == y[i]).mean() - (b[i] == y[i]).mean())
        d_f1.append(_macro_f1(y[i], a[i], n_classes) - _macro_f1(y[i], b[i], n_classes))
    q = [2.5, 97.5]
    return {"acc_diff_ci95": [float(x) for x in np.percentile(d_acc, q)],
            "macro_f1_diff_ci95": [float(x) for x in np.percentile(d_f1, q)]}
