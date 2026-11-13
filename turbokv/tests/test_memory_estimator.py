"""Unit tests for KV cache theoretical memory estimator."""

import pytest
from turbokv.config.model_config import ModelConfig
from turbokv.config.policy_config import CacheConfig
from turbokv.optimization.memory_estimator import MemoryEstimator


def test_estimator_fp16():
    model_cfg = ModelConfig(num_layers=28, num_kv_heads=2, head_dim=128)
    estimator = MemoryEstimator(model_cfg)

    # 1 token elements = 2 * 28 * 2 * 128 = 14336 elements
    # FP16 = 2 bytes per element = 28672 bytes per token
    # For seq_len = 256 (matches step chunk exactly)
    est = estimator.estimate(256, CacheConfig("fp16"), account_for_step_padding=True)
    expected_bytes = 256 * 28672
    assert est.total_bytes == expected_bytes
    assert est.padded_sequence_length == 256
    assert est.unquantized_tokens == 256
    assert est.quantized_tokens == 0


def test_estimator_step_padding():
    estimator = MemoryEstimator()
    # 100 tokens should be padded to 256
    assert estimator.pad_to_step(100) == 256
    # 256 tokens should remain 256
    assert estimator.pad_to_step(256) == 256
    # 257 tokens should pad to 512
    assert estimator.pad_to_step(257) == 512


def test_estimator_int8_vs_int4_reduction():
    estimator = MemoryEstimator()
    est_fp16 = estimator.estimate(1024, CacheConfig("fp16"))
    est_int8 = estimator.estimate(1024, CacheConfig("int8", group_size=64, quantized_start=0))
    est_int4 = estimator.estimate(1024, CacheConfig("int4", group_size=64, quantized_start=0))

    # INT8 with group 64 metadata should be ~53% of FP16
    ratio_int8 = est_int8.total_bytes / est_fp16.total_bytes
    assert 0.50 < ratio_int8 < 0.55

    # INT4 with group 64 metadata should be ~28% of FP16
    ratio_int4 = est_int4.total_bytes / est_fp16.total_bytes
    assert 0.25 < ratio_int4 < 0.30

    # INT4 memory must be strictly less than INT8 memory
    assert est_int4.total_bytes < est_int8.total_bytes


def test_estimator_quantized_onset():
    estimator = MemoryEstimator()
    # At seq_len 1024 with quantized_start = 512:
    # First 512 tokens are FP16, remaining 512 are INT4
    est = estimator.estimate(1024, CacheConfig("int4", group_size=64, quantized_start=512))
    assert est.unquantized_tokens == 512
    assert est.quantized_tokens == 512
    assert est.unquantized_bytes > est.quantized_data_bytes
