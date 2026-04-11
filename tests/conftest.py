"""Shared fixtures (optional HF tiny model for inference tests)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def tiny_classifier_model_dir(tmp_path_factory: pytest.TempPathFactory):
    """Small BERT saved to disk (downloads once per test module)."""
    from transformers.utils.import_utils import is_torch_available

    if not is_torch_available():
        pytest.skip(
            "PyTorch is missing or below the version required by this transformers "
            "install (e.g. upgrade torch to >= 2.4 for transformers 5.x)."
        )

    import torch  # noqa: F401 -- register backend before modeling imports
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = tmp_path_factory.mktemp("hf_tiny")
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bert",
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d
