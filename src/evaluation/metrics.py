"""Evaluation metrics helpers — used by both the classical baseline and the
future CNN evaluation, so numbers are computed the same way everywhere."""
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)


def compute_metrics(y_true, y_pred, class_names=None) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }
