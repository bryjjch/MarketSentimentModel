"""Tests for versioned inference API."""

from __future__ import annotations

from training.inference import (
    DEFAULT_MAX_LENGTH,
    INFERENCE_API_VERSION,
    SentimentPredictor,
)


def test_default_max_length_matches_train_default():
    from training.train_classifier import parse_args

    p = parse_args(["--output_dir", "x"])
    assert p.max_length == DEFAULT_MAX_LENGTH


def test_predictor_empty_and_batching(tiny_classifier_model_dir):
    pred = SentimentPredictor(tiny_classifier_model_dir, device="cpu")
    out = pred.predict(["", "   ", "hello", "world"], batch_size=2)
    assert len(out) == 4
    assert out[0].error == "empty_or_whitespace_text"
    assert out[1].error == "empty_or_whitespace_text"
    assert out[2].error is None
    assert out[2].label_id in (0, 1, 2)
    assert out[2].probabilities is not None
    assert set(out[2].probabilities.keys()) == {"negative", "neutral", "positive"}
    assert out[2].inference_api_version == INFERENCE_API_VERSION


def test_predict_one(tiny_classifier_model_dir):
    pred = SentimentPredictor(tiny_classifier_model_dir, device="cpu")
    r = pred.predict_one("EPS guidance raised")
    assert r.error is None
    assert r.label_name in ("negative", "neutral", "positive")
