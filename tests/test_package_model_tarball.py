"""Tests for infra/scripts/package_model_tarball.py."""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


def _load_package_module(repo_root: Path):
    path = repo_root / "infra" / "scripts" / "package_model_tarball.py"
    spec = importlib.util.spec_from_file_location("package_model_tarball", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def fake_hf_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "hf_model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["BertForSequenceClassification"]}), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"not-a-real-checkpoint")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (d / "checkpoint-999").mkdir()
    (d / "checkpoint-999" / "config.json").write_text("{}", encoding="utf-8")
    return d


def test_package_excludes_checkpoints_and_includes_code(
    fake_hf_model_dir: Path, tmp_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_package_module(repo_root)
    serving = repo_root / "infra" / "sagemaker" / "serving" / "code"
    out = tmp_path / "model.tar.gz"

    mod.package_model(fake_hf_model_dir, out, serving_code_dir=serving)

    assert out.is_file()
    with tarfile.open(out, "r:gz") as tf:
        names = {m.name.replace("\\", "/") for m in tf.getmembers() if m.isfile()}

    assert "code/inference.py" in names
    assert "config.json" in names
    assert not any(n.startswith("checkpoint-") for n in names)
