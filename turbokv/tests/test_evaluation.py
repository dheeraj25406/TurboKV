"""Unit tests for evaluation metrics: exact match, normalized match, and distribution divergence."""

import pytest
import numpy as np
import mlx.core as mx
from turbokv.evaluation.retrieval import (
    normalize_text,
    compute_exact_match,
    compute_normalized_match,
    evaluate_retrieval,
)
from turbokv.datasets.retrieval_dataset import RetrievalItem
from turbokv.evaluation.distribution_metrics import compute_distribution_metrics


def test_normalize_text():
    raw = "  Hello, World! 123 \n\t "
    norm = normalize_text(raw)
    assert norm == "hello world 123"


def test_compute_exact_match():
    assert compute_exact_match("The password is Delta-99.", "Delta-99") is True
    assert compute_exact_match("Delta-99", "Delta-99") is True
    assert compute_exact_match("Something completely different", "Delta-99") is False


def test_compute_normalized_match():
    # Case differences and trailing punctuation
    assert compute_normalized_match("delta-99.", "Delta-99") is True
    assert compute_normalized_match("Coordinates: Latitude 45.12!", "latitude 45.12") is True


def test_evaluate_retrieval_aggregation():
    items = [
        RetrievalItem(
            item_id="item1",
            target_fact="",
            question="",
            ground_truth_answer="433.92 megahertz",
            context_tokens_target=1024,
            actual_prompt_tokens=1024,
            fact_position_fraction=0.1,
            position_bucket="beginning",
            full_prompt="",
            split="calibration",
        ),
        RetrievalItem(
            item_id="item2",
            target_fact="",
            question="",
            ground_truth_answer="Delta-99-Gamma",
            context_tokens_target=1024,
            actual_prompt_tokens=1024,
            fact_position_fraction=0.9,
            position_bucket="end",
            full_prompt="",
            split="calibration",
        ),
    ]

    preds = ["The frequency is 433.92 megahertz.", "Incorrect Answer"]
    report = evaluate_retrieval(items, preds)

    assert report.total_samples == 2
    assert report.overall_accuracy == 0.5
    assert report.accuracy_by_position["beginning"] == 1.0
    assert report.accuracy_by_position["end"] == 0.0


def test_compute_distribution_metrics():
    # Identical distributions should yield KL ~ 0.0 and Top-1 agreement = 1.0
    lp1 = mx.array([-1.0, -0.2, -3.0])
    lp2 = mx.array([-1.0, -0.2, -3.0])

    report = compute_distribution_metrics([lp1], [lp2])
    assert report.total_steps == 1
    assert report.top1_agreement_rate == 1.0
    assert report.mean_kl_divergence < 1e-5

    # Divergent distributions
    lp_divergent = mx.array([-0.1, -2.5, -4.0])
    report_div = compute_distribution_metrics([lp1], [lp_divergent])
    assert report_div.top1_agreement_rate == 0.0
    assert report_div.mean_kl_divergence > 0.1
