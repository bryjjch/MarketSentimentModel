"""Training run metadata written next to saved weights."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_MANIFEST_NAME = "training_manifest.json"


def _git_revision(cwd: Path | None = None) -> str | None:
    """Get the Git revision of the repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _library_versions() -> dict[str, str]:
    """Get the versions of the libraries used in the training run."""
    import importlib.metadata

    import torch
    import transformers

    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    for pkg in ("datasets", "accelerate", "numpy", "pandas", "scikit-learn"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def build_training_manifest(
    *,
    run_configuration: dict[str, Any],
    validation_metrics: Mapping[str, Any] | None = None,
    test_metrics: Mapping[str, Any] | None = None,
    validation_confusion_matrix: list[list[int]] | None = None,
    test_confusion_matrix: list[list[int]] | None = None,
    best_model_checkpoint: str | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    data_summary: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serializable manifest dict."""
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(cwd=root),
        "inference_api_version": None,
        "library_versions": _library_versions(),
        "run_configuration": run_configuration,
        "data_summary": data_summary or {},
        "validation_metrics": dict(validation_metrics or {}),
        "test_metrics": dict(test_metrics or {}),
        "validation_confusion_matrix": validation_confusion_matrix,
        "test_confusion_matrix": test_confusion_matrix,
        "best_model_checkpoint": best_model_checkpoint,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    try:
        from .inference import INFERENCE_API_VERSION

        manifest["inference_api_version"] = INFERENCE_API_VERSION
    except ImportError:
        pass
    return manifest


def write_training_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``training_manifest.json`` under ``output_dir``."""
    out = Path(output_dir) / _MANIFEST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return out


def args_namespace_to_dict(ns: Any) -> dict[str, Any]:
    """Convert argparse Namespace (or dict) to JSON-friendly primitives."""
    if isinstance(ns, dict):
        raw = ns
    else:
        raw = vars(ns).copy()
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, Path):
            out[k] = str(v.resolve()) if v.exists() or v.is_absolute() else str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = repr(v)
    return out
