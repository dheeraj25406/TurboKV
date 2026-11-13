"""Unit tests for configuration classes."""

import pytest
from turbokv.config.model_config import ModelConfig
from turbokv.config.benchmark_config import BenchmarkConfig
from turbokv.config.policy_config import CacheConfig, PolicyConstraints


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.num_layers == 28
    assert cfg.num_kv_heads == 2
    assert cfg.head_dim == 128
    assert cfg.hidden_size == 1536
    assert cfg.weight_bits == 4


def test_cache_config_precisions():
    c_fp16 = CacheConfig("fp16")
    assert c_fp16.bits is None
    assert c_fp16.key == "fp16"

    c_int8 = CacheConfig("int8", group_size=64, quantized_start=0)
    assert c_int8.bits == 8
    assert c_int8.key == "int8_g64_s0"

    c_int4 = CacheConfig("int4", group_size=32, quantized_start=512)
    assert c_int4.bits == 4
    assert c_int4.key == "int4_g32_s512"

    with pytest.raises(ValueError):
        CacheConfig("int2")


def test_cache_config_serialization():
    cfg = CacheConfig("int8", group_size=32, quantized_start=256)
    d = cfg.to_dict()
    assert d["precision"] == "int8"
    assert d["bits"] == 8
    assert d["group_size"] == 32
    assert d["quantized_start"] == 256

    cfg_restored = CacheConfig.from_dict(d)
    assert cfg_restored.key == cfg.key
    assert cfg_restored.bits == cfg.bits


def test_benchmark_config_defaults():
    cfg = BenchmarkConfig()
    assert 1024 in cfg.context_lengths
    assert cfg.generation_tokens == 128
    assert cfg.num_repetitions == 3
