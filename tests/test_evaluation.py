"""Tests for classification metric helpers."""

from __future__ import annotations

import numpy as np

from training.evaluation import classification_metrics, confusion_matrix_list, metrics_from_eval_pred


def test_classification_metrics_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["cm_0_0"] == 2.0
    assert m["cm_1_1"] == 2.0
    assert m["cm_2_2"] == 2.0


def test_confusion_matrix_list_order():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 2, 2, 1])
    cm = confusion_matrix_list(y_true, y_pred)
    assert cm[0][0] == 1
    assert cm[0][1] == 1
    assert cm[1][2] == 1
    assert cm[2][2] == 1


def test_metrics_from_eval_pred():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1])
    m = metrics_from_eval_pred((logits, labels))
    assert m["accuracy"] == 1.0
    assert "macro_f1" in m
