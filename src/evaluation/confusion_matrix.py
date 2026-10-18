"""Confusion matrix plotting — flags the specific confusions that matter
most for this use case: healthy-predicted-as-diseased (false alarm) and,
worse, diseased-predicted-as-healthy (missed disease, the costlier error
for a farmer)."""
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(class_names)), max(5, 0.5 * len(class_names))))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return cm


def missed_disease_rate(y_true, y_pred, healthy_label: str) -> float:
    """Fraction of truly-diseased samples the model called healthy — the
    highest-cost error type for a farmer relying on this tool."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diseased_mask = y_true != healthy_label
    if diseased_mask.sum() == 0:
        return float("nan")
    missed = (y_pred[diseased_mask] == healthy_label).sum()
    return missed / diseased_mask.sum()
