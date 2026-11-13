"""Unit tests for adaptive KV cache policy selection."""

import pytest
from turbokv.config.policy_config import PolicyConstraints, CacheConfig
from turbokv.optimization.adaptive_policy import AdaptiveKVPolicy


@pytest.fixture
def mock_calibration_data():
    return {
        "version": "1.0.0",
        "by_context_length": {
            "1024": {
                "fp16": {
                    "measured_cache_mb": 28.0,
                    "retrieval_accuracy": 1.0,
                    "decode_tokens_per_sec": 75.0,
                },
                "int8_g64_s0": {
                    "measured_cache_mb": 15.0,
                    "retrieval_accuracy": 1.0,
                    "decode_tokens_per_sec": 74.0,
                },
                "int4_g64_s0": {
                    "measured_cache_mb": 8.0,
                    "retrieval_accuracy": 0.80,  # 20% drop
                    "decode_tokens_per_sec": 72.0,
                },
                "int4_g64_s512": {
                    "measured_cache_mb": 18.0,
                    "retrieval_accuracy": 1.0,  # Preserves accuracy with onset delay
                    "decode_tokens_per_sec": 73.0,
                },
            },
            "4096": {
                "fp16": {
                    "measured_cache_mb": 112.0,
                    "retrieval_accuracy": 1.0,
                    "decode_tokens_per_sec": 70.0,
                },
                "int8_g64_s0": {
                    "measured_cache_mb": 60.0,
                    "retrieval_accuracy": 1.0,
                    "decode_tokens_per_sec": 68.0,
                },
                "int4_g64_s0": {
                    "measured_cache_mb": 32.0,
                    "retrieval_accuracy": 0.70,
                    "decode_tokens_per_sec": 65.0,
                },
                "int4_g64_s512": {
                    "measured_cache_mb": 45.0,
                    "retrieval_accuracy": 0.98,
                    "decode_tokens_per_sec": 66.0,
                },
            },
        },
    }


def test_policy_selects_int8_when_budget_tight(mock_calibration_data):
    policy = AdaptiveKVPolicy(calibration_data=mock_calibration_data)
    # At ctx 1024, memory budget 20 MB, max acc drop 0.05
    # FP16 = 28 MB (exceeds budget)
    # INT4 s0 = 8 MB (accuracy drop is 0.20, exceeds allowed 0.05)
    # INT8 s0 = 15 MB (fits 20 MB budget, accuracy drop is 0.0, satisfies all!)
    constraints = PolicyConstraints(max_memory_budget_mb=20.0, max_accuracy_drop=0.05)
    decision = policy.select(1024, constraints)

    assert decision.is_feasible is True
    assert decision.fallback_to_fp16 is False
    assert decision.chosen_config is not None
    assert decision.chosen_config.precision == "int8"
    assert "Selected optimal configuration" in decision.decision_reason


def test_policy_selects_onset_delayed_int4_when_accuracy_preserved(mock_calibration_data):
    policy = AdaptiveKVPolicy(calibration_data=mock_calibration_data)
    # At ctx 4096, budget 50 MB, max acc drop 0.05
    # FP16 = 112 MB (exceeds)
    # INT8 = 60 MB (exceeds 50 MB)
    # INT4 s0 = 32 MB (drop is 0.30, exceeds)
    # INT4 s512 = 45 MB (fits 50 MB, drop is 0.02 <= 0.05) -> Should be chosen!
    constraints = PolicyConstraints(max_memory_budget_mb=50.0, max_accuracy_drop=0.05)
    decision = policy.select(4096, constraints)

    assert decision.is_feasible is True
    assert decision.chosen_config.key == "int4_g64_s512"


def test_policy_fallback_to_fp16_when_quantized_fails_quality(mock_calibration_data):
    policy = AdaptiveKVPolicy(calibration_data=mock_calibration_data)
    # Only INT4 s0 is evaluated (say candidate list only has INT4 s0 and FP16)
    # INT4 drop 0.20 fails max drop 0.05
    # FP16 fits generous 100 MB budget -> fallback to FP16!
    cands = [CacheConfig("fp16"), CacheConfig("int4", group_size=64, quantized_start=0)]
    constraints = PolicyConstraints(max_memory_budget_mb=100.0, max_accuracy_drop=0.05)
    decision = policy.select(1024, constraints, candidate_configs=cands)

    assert decision.is_feasible is True
    assert decision.fallback_to_fp16 is True
    assert decision.chosen_config.precision == "fp16"
    assert "Fallback to FP16" in decision.decision_reason


def test_policy_infeasible_when_budget_too_small(mock_calibration_data):
    policy = AdaptiveKVPolicy(calibration_data=mock_calibration_data)
    # Budget 5 MB (no config fits 8 MB minimum)
    constraints = PolicyConstraints(max_memory_budget_mb=5.0)
    decision = policy.select(1024, constraints)

    assert decision.is_feasible is False
    assert decision.chosen_config is None
    assert "Workload infeasible" in decision.decision_reason
