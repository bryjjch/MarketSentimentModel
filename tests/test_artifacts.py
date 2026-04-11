"""Tests for training manifest helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from training.artifacts import (
    args_namespace_to_dict,
    build_training_manifest,
    write_training_manifest,
)


def test_args_namespace_to_dict_paths(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("x", encoding="utf-8")
    ns = Namespace(output_dir=p, seed=3, flag=True)
    d = args_namespace_to_dict(ns)
    assert d["seed"] == 3
    assert d["flag"] is True
    assert "f.txt" in d["output_dir"]


def test_write_training_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = build_training_manifest(
        run_configuration={"output_dir": str(tmp_path / "out")},
        validation_metrics={"eval_macro_f1": 0.5},
        test_metrics={"test_accuracy": 0.9},
        validation_confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        test_confusion_matrix=None,
        best_metric_name="eval_macro_f1",
        best_metric_value=0.5,
    )
    path = write_training_manifest(tmp_path, manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["validation_confusion_matrix"][0][0] == 1
    assert data["test_confusion_matrix"] is None
