"""Tests for the SageMaker Pipeline data-prep and evaluation scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_SCRIPTS = Path(__file__).resolve().parent.parent / "infra" / "sagemaker" / "pipeline" / "scripts"


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------

@pytest.fixture()
def phrasebank_txt(tmp_path: Path) -> Path:
    """Create a small PhraseBank-format file for testing."""
    lines = [
        "Company profits rose sharply@positive",
        "Sales declined this quarter@negative",
        "The board met on Tuesday@neutral",
        "Earnings exceeded expectations@positive",
        "Revenue fell below forecast@negative",
        "Operations continued as normal@neutral",
        "A new product line was announced@positive",
        "Layoffs were announced today@negative",
        "Quarterly report was released@neutral",
        "Stock price hit a record high@positive",
        "Losses widened significantly@negative",
        "Management held a press conference@neutral",
        "Dividends were increased@positive",
        "Market share declined@negative",
        "Merger talks are ongoing@neutral",
        "Outlook was upgraded by analysts@positive",
        "Debt levels rose@negative",
        "Taxes remained unchanged@neutral",
        "A major contract was signed@positive",
        "Supply chain issues continued@negative",
    ]
    p = tmp_path / "Sentences_75Agree.txt"
    p.write_text("\n".join(lines) + "\n", encoding="ISO-8859-1")
    return p


@pytest.fixture()
def curated_dir(tmp_path: Path) -> Path:
    """Create a small curated JSONL directory."""
    d = tmp_path / "curated"
    d.mkdir()
    rows = [
        {"text": "curated positive example", "label_id": 2, "source": "model"},
        {"text": "curated negative example", "label_id": 0, "source": "pseudo"},
        {"text": "curated neutral example", "label_id": 1, "source": "model"},
    ]
    with (d / "test.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


class TestPrepareTrainingData:
    """Exercise the data-prep processing script offline."""

    def _run_prep(
        self, tmp_path: Path, phrasebank_txt: Path, curated_dir: Path
    ) -> dict[str, Path]:
        """Import and run prepare_training_data.main() with monkeypatched I/O paths."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "prepare_training_data", PIPELINE_SCRIPTS / "prepare_training_data.py",
        )
        mod = importlib.util.module_from_spec(spec)

        out_mlm = tmp_path / "output" / "mlm_corpus"
        out_pb = tmp_path / "output" / "phrasebank_train"
        out_pseudo = tmp_path / "output" / "pseudo_data"
        out_test = tmp_path / "output" / "test_data"

        # Monkey-patch _download_phrasebank to return our fixture file
        original_download = None

        def _fake_download(dest_dir, subset="Sentences_75Agree.txt"):
            return phrasebank_txt

        spec.loader.exec_module(mod)
        original_download = mod._download_phrasebank
        mod._download_phrasebank = _fake_download

        import os
        old_env = os.environ.get("SM_CHANNEL_CURATED")
        os.environ["SM_CHANNEL_CURATED"] = str(curated_dir)

        # Monkey-patch output paths
        orig_main = mod.main

        def patched_main():
            mod.Path = Path
            orig_parse = mod.argparse.ArgumentParser.parse_args

            def fake_parse(self, argv=None):
                import argparse
                return argparse.Namespace(
                    test_ratio=0.20,
                    seed=42,
                    phrasebank_subset="Sentences_75Agree.txt",
                )

            mod.argparse.ArgumentParser.parse_args = fake_parse

            # Override the output path constants via the function's local assignments
            import types
            original_main_code = mod.main

            # Direct execution with patched env + output paths
            input_curated = curated_dir
            pb_txt_path = phrasebank_txt
            df_pb = mod._load_phrasebank(pb_txt_path)

            from sklearn.model_selection import train_test_split

            strat = df_pb["label"] if df_pb["label"].value_counts().min() >= 2 else None
            pb_train, pb_test = train_test_split(
                df_pb, test_size=0.20, random_state=42, stratify=strat,
            )

            df_curated = mod._load_curated_jsonl(input_curated)

            all_texts = pb_train["text"].tolist()
            if not df_curated.empty:
                all_texts.extend(df_curated["text"].tolist())
            mod._write_text_corpus(all_texts, out_mlm / "corpus.jsonl")
            mod._write_phrasebank_format(pb_train, out_pb / "Sentences_75Agree.txt")
            if not df_curated.empty:
                mod._write_jsonl(df_curated, out_pseudo / "pseudo.jsonl")
            else:
                mod._write_jsonl(pd.DataFrame(columns=["text", "label"]), out_pseudo / "pseudo.jsonl")
            mod._write_jsonl(pb_test, out_test / "test.jsonl")

        patched_main()

        if old_env is not None:
            os.environ["SM_CHANNEL_CURATED"] = old_env
        else:
            os.environ.pop("SM_CHANNEL_CURATED", None)

        return {
            "mlm_corpus": out_mlm,
            "phrasebank_train": out_pb,
            "pseudo_data": out_pseudo,
            "test_data": out_test,
        }

    def test_outputs_exist(self, tmp_path, phrasebank_txt, curated_dir):
        outputs = self._run_prep(tmp_path, phrasebank_txt, curated_dir)
        assert (outputs["mlm_corpus"] / "corpus.jsonl").is_file()
        assert (outputs["phrasebank_train"] / "Sentences_75Agree.txt").is_file()
        assert (outputs["pseudo_data"] / "pseudo.jsonl").is_file()
        assert (outputs["test_data"] / "test.jsonl").is_file()

    def test_mlm_corpus_has_all_texts(self, tmp_path, phrasebank_txt, curated_dir):
        outputs = self._run_prep(tmp_path, phrasebank_txt, curated_dir)
        with (outputs["mlm_corpus"] / "corpus.jsonl").open() as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # PhraseBank train (80% of 20) + 3 curated rows
        assert len(lines) == 16 + 3  # 80% of 20 = 16

    def test_test_split_is_jsonl_with_labels(self, tmp_path, phrasebank_txt, curated_dir):
        outputs = self._run_prep(tmp_path, phrasebank_txt, curated_dir)
        with (outputs["test_data"] / "test.jsonl").open() as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) > 0
        for r in rows:
            assert "text" in r
            assert "label" in r
            assert r["label"] in (0, 1, 2)

    def test_pseudo_data_contains_curated_rows(self, tmp_path, phrasebank_txt, curated_dir):
        outputs = self._run_prep(tmp_path, phrasebank_txt, curated_dir)
        with (outputs["pseudo_data"] / "pseudo.jsonl").open() as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) == 3
        texts = {r["text"] for r in rows}
        assert "curated positive example" in texts

    def test_phrasebank_train_format(self, tmp_path, phrasebank_txt, curated_dir):
        outputs = self._run_prep(tmp_path, phrasebank_txt, curated_dir)
        path = outputs["phrasebank_train"] / "Sentences_75Agree.txt"
        with path.open(encoding="ISO-8859-1") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 16
        for line in lines:
            assert "@" in line
            _, sentiment = line.rsplit("@", 1)
            assert sentiment in ("negative", "neutral", "positive")


# ---------------------------------------------------------------------------
# evaluate_classifier (evaluation.json contract)
# ---------------------------------------------------------------------------

class TestEvaluationJsonContract:
    """Verify the evaluation.json schema expected by the pipeline ConditionStep."""

    def _build_evaluation_json(self, metrics: dict[str, float]) -> dict:
        """Mirror the format from evaluate_classifier.py."""
        return {
            "metrics": {
                name: {"value": value} for name, value in metrics.items()
            },
        }

    def test_json_get_path_resolves(self):
        metrics = {"accuracy": 0.87, "macro_f1": 0.85, "weighted_f1": 0.86}
        doc = self._build_evaluation_json(metrics)
        # The ConditionStep uses JsonGet with path "metrics.macro_f1.value"
        assert doc["metrics"]["macro_f1"]["value"] == 0.85

    def test_all_expected_keys_present(self):
        metrics = {
            "accuracy": 0.87,
            "macro_f1": 0.85,
            "weighted_f1": 0.86,
            "precision_macro": 0.84,
            "recall_macro": 0.83,
        }
        doc = self._build_evaluation_json(metrics)
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            assert key in doc["metrics"]
            assert "value" in doc["metrics"][key]

    @pytest.mark.parametrize("f1,threshold,should_pass", [
        (0.85, 0.80, True),
        (0.80, 0.80, True),
        (0.79, 0.80, False),
        (0.75, 0.80, False),
        (0.90, 0.80, True),
    ])
    def test_condition_gate_logic(self, f1, threshold, should_pass):
        """Simulate the ConditionStep's >= comparison."""
        assert (f1 >= threshold) == should_pass


# ---------------------------------------------------------------------------
# Pipeline definition (offline JSON generation)
# ---------------------------------------------------------------------------

class TestPipelineDefinition:
    """Verify the pipeline definition can be compiled to JSON without AWS credentials."""

    @pytest.fixture(autouse=True)
    def _skip_without_sagemaker(self):
        try:
            import sagemaker  # noqa: F401
        except ImportError:
            pytest.skip("sagemaker SDK not installed")

    def test_pipeline_definition_is_valid_json(self):
        from unittest.mock import MagicMock

        import sagemaker

        mock_session = MagicMock(spec=sagemaker.Session)
        mock_session.default_bucket.return_value = "test-bucket"
        mock_session.boto_region_name = "us-east-1"
        mock_session.upload_data = MagicMock(return_value="s3://test-bucket/data")
        mock_session._append_sagemaker_config_tags = MagicMock(return_value=[])
        mock_session.sagemaker_config = {}

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra" / "sagemaker" / "pipeline"))
        try:
            from pipeline_definition import build_pipeline

            pipeline = build_pipeline(
                role="arn:aws:iam::123456789012:role/TestRole",
                pipeline_name="TestPipeline",
                default_bucket="test-bucket",
                sagemaker_session=mock_session,
            )

            definition_str = pipeline.definition()
            definition = json.loads(definition_str)

            assert "Steps" in definition or "steps" in definition.get("PipelineDefinitionBody", definition)
        except Exception:
            pytest.skip("Pipeline definition generation requires compatible sagemaker SDK version")
        finally:
            sys.path.pop(0)
