"""Classification metrics for validation / test."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_labels: int = 3,
    label_names: tuple[str, ...] = ("negative", "neutral", "positive"),
) -> dict[str, float]:
    """
    Flat dict of scalar metrics suitable for Hugging Face ``Trainer.compute_metrics``
    and JSON manifests (confusion cells as ``cm_true_pred`` = count).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    labels = list(range(num_labels))

    # Calculate the accuracy
    acc = float(accuracy_score(y_true, y_pred))
    # Calculate the macro F1 score
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    # Calculate the weighted F1 score
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    
    # Calculate the precision, recall, and F1 score for each label
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    out: dict[str, float] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
    }
    # For each label, add the precision, recall, and F1 score to the output
    for i in range(num_labels):
        name = label_names[i] if i < len(label_names) else str(i)
        out[f"precision_{name}"] = float(prec[i])
        out[f"recall_{name}"] = float(rec[i])
        out[f"f1_{name}"] = float(f1[i])

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # For each label, add the confusion matrix cell to the output
    for i in range(num_labels):
        for j in range(num_labels):
            out[f"cm_{i}_{j}"] = float(cm[i, j])
    return out


def metrics_from_eval_pred(eval_pred: Any) -> dict[str, float]:
    """Adapter for ``Trainer.compute_metrics`` (logits + labels)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return classification_metrics(labels, preds)


def confusion_matrix_list(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_labels: int = 3,
) -> list[list[int]]:
    """Nested confusion matrix rows = true class, columns = predicted class."""
    cm = confusion_matrix(
        np.asarray(y_true).astype(int).ravel(),
        np.asarray(y_pred).astype(int).ravel(),
        labels=list(range(num_labels)),
    )
    return [[int(x) for x in row] for row in cm]
