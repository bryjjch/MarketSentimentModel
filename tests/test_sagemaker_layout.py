"""Tests for SageMaker-aware path resolution and artifact layout helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.common import (
    copy_serving_code_into,
    default_serving_code_dir,
    sagemaker_channel_dir,
    sagemaker_model_dir,
    sagemaker_output_data_dir,
)
from training.train_classifier import parse_args as classifier_parse_args
from training.train_mlm import parse_args as mlm_parse_args


def test_default_serving_code_dir_points_at_repo_inference_py():
    d = default_serving_code_dir()
    assert (d / "inference.py").is_file()


def test_sagemaker_env_helpers_honor_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    monkeypatch.delenv("SM_OUTPUT_DATA_DIR", raising=False)
    monkeypatch.delenv("SM_CHANNEL_INPUT", raising=False)
    assert sagemaker_model_dir() is None
    assert sagemaker_output_data_dir() is None
    assert sagemaker_channel_dir("input") is None

    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    channel_dir = tmp_path / "channel"
    monkeypatch.setenv("SM_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(output_dir))
    monkeypatch.setenv("SM_CHANNEL_INPUT", str(channel_dir))
    assert sagemaker_model_dir() == model_dir
    assert sagemaker_output_data_dir() == output_dir
    assert sagemaker_channel_dir("input") == channel_dir


def test_copy_serving_code_into_writes_inference_py(tmp_path: Path):
    target = tmp_path / "model_artifact"
    target.mkdir()

    copied = copy_serving_code_into(target, default_serving_code_dir())

    assert copied == target / "code"
    assert (target / "code" / "inference.py").is_file()


def test_copy_serving_code_into_replaces_existing_code_dir(tmp_path: Path):
    target = tmp_path / "model_artifact"
    target.mkdir()
    stale = target / "code"
    stale.mkdir()
    (stale / "stale.py").write_text("# stale", encoding="utf-8")

    copy_serving_code_into(target, default_serving_code_dir())

    assert not (target / "code" / "stale.py").exists()
    assert (target / "code" / "inference.py").is_file()


def test_copy_serving_code_into_errors_on_missing_source(tmp_path: Path):
    target = tmp_path / "model_artifact"
    target.mkdir()
    with pytest.raises(FileNotFoundError):
        copy_serving_code_into(target, tmp_path / "does_not_exist")


def test_classifier_parse_args_defaults_output_dir_to_sm_model_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "sm_model"))
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(tmp_path / "sm_out"))
    args = classifier_parse_args([])
    assert Path(args.output_dir) == tmp_path / "sm_model"
    assert Path(args.checkpoint_dir) == tmp_path / "sm_out" / "checkpoints"


def test_classifier_parse_args_requires_output_dir_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    monkeypatch.delenv("SM_OUTPUT_DATA_DIR", raising=False)
    with pytest.raises(SystemExit):
        classifier_parse_args([])


def test_classifier_parse_args_cli_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "sm_model"))
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(tmp_path / "sm_out"))
    override = tmp_path / "cli_override"
    args = classifier_parse_args(["--output_dir", str(override)])
    assert Path(args.output_dir) == override


def test_mlm_parse_args_defaults_from_sagemaker_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    channel = tmp_path / "train_channel"
    channel.mkdir()
    a = channel / "a.jsonl"
    b = channel / "b.txt"
    noise = channel / "ignored.csv"
    a.write_text(json.dumps({"text": "hi"}) + "\n", encoding="utf-8")
    b.write_text("hello world\n", encoding="utf-8")
    noise.write_text("skip\n", encoding="utf-8")

    monkeypatch.setenv("SM_CHANNEL_TRAIN", str(channel))
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "sm_model"))
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(tmp_path / "sm_out"))

    args = mlm_parse_args([])
    assert sorted(args.train_files) == sorted([str(a), str(b)])
    assert Path(args.output_dir) == tmp_path / "sm_model"
    assert Path(args.checkpoint_dir) == tmp_path / "sm_out" / "checkpoints"


def test_mlm_parse_args_requires_files_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SM_CHANNEL_TRAIN", raising=False)
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    monkeypatch.delenv("SM_OUTPUT_DATA_DIR", raising=False)
    with pytest.raises(SystemExit):
        mlm_parse_args([])


def test_mlm_parse_args_cli_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    channel = tmp_path / "train_channel"
    channel.mkdir()
    (channel / "a.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("SM_CHANNEL_TRAIN", str(channel))
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "sm_model"))
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(tmp_path / "sm_out"))

    cli_file = tmp_path / "explicit.jsonl"
    cli_file.write_text("{}\n", encoding="utf-8")
    cli_out = tmp_path / "cli_out"

    args = mlm_parse_args(
        ["--train_files", str(cli_file), "--output_dir", str(cli_out)]
    )
    assert args.train_files == [str(cli_file)]
    assert Path(args.output_dir) == cli_out
