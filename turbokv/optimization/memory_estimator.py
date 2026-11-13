"""Theoretical KV cache memory estimation and modeling."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from turbokv.config.model_config import ModelConfig
from turbokv.config.policy_config import CacheConfig


@dataclass
class MemoryEstimate:
    """Detailed breakdown of theoretical KV cache memory."""

    sequence_length: int
    padded_sequence_length: int
    precision: str
    bits: Optional[int]
    group_size: int
    quantized_start: int

    unquantized_tokens: int
    quantized_tokens: int

    unquantized_bytes: int
    quantized_data_bytes: int
    quantized_scale_bytes: int
    quantized_bias_bytes: int
    total_bytes: int
    total_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "padded_sequence_length": self.padded_sequence_length,
            "precision": self.precision,
            "bits": self.bits,
            "group_size": self.group_size,
            "quantized_start": self.quantized_start,
            "unquantized_tokens": self.unquantized_tokens,
            "quantized_tokens": self.quantized_tokens,
            "unquantized_bytes": self.unquantized_bytes,
            "quantized_data_bytes": self.quantized_data_bytes,
            "quantized_scale_bytes": self.quantized_scale_bytes,
            "quantized_bias_bytes": self.quantized_bias_bytes,
            "total_bytes": self.total_bytes,
            "total_mb": self.total_mb,
        }


class MemoryEstimator:
    """
    Theoretical KV cache memory estimator.
    Computes exact byte footprint including:
    - 2 (Key and Value) * layers * seq_len * kv_heads * head_dim * bytes_per_element
    - MLX Quantize pack format (uint32 packing)
    - FP16 scale and bias metadata per quantization group
    - Quantization onset threshold (quantized_start)
    - Cache step-growth allocation padding (step = 256)
    """

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.config = model_config or ModelConfig()
        self.step_chunk_size = 256  # MLX KVCache and QuantizedKVCache step chunk

    def pad_to_step(self, seq_len: int) -> int:
        """MLX KVCache grows in steps of step_chunk_size (256)."""
        if seq_len <= 0:
            return 0
        return ((seq_len + self.step_chunk_size - 1) // self.step_chunk_size) * self.step_chunk_size

    def estimate_cache_memory_mb(self, sequence_length: int, cache_config: CacheConfig) -> float:
        """Convenience method returning theoretical cache memory in MB."""
        return self.estimate(sequence_length, cache_config).total_mb

    def estimate(
        self,
        sequence_length: int,
        cache_config: CacheConfig,
        account_for_step_padding: bool = True,
    ) -> MemoryEstimate:
        effective_len = (
            self.pad_to_step(sequence_length)
            if account_for_step_padding
            else sequence_length
        )

        elements_per_token = 2 * self.config.num_layers * self.config.num_kv_heads * self.config.head_dim

        # FP16 elements are 2 bytes each (float16)
        fp16_bytes_per_token = elements_per_token * 2

        if cache_config.bits is None or cache_config.precision.lower() == "fp16":
            total_bytes = effective_len * fp16_bytes_per_token
            return MemoryEstimate(
                sequence_length=sequence_length,
                padded_sequence_length=effective_len,
                precision="fp16",
                bits=None,
                group_size=cache_config.group_size,
                quantized_start=0,
                unquantized_tokens=effective_len,
                quantized_tokens=0,
                unquantized_bytes=total_bytes,
                quantized_data_bytes=0,
                quantized_scale_bytes=0,
                quantized_bias_bytes=0,
                total_bytes=total_bytes,
                total_mb=total_bytes / (1024.0 * 1024.0),
            )

        bits = cache_config.bits
        group_size = max(1, cache_config.group_size)
        start = cache_config.quantized_start

        # If effective_len < start, cache is entirely unquantized
        if effective_len <= start:
            unquantized_tokens = effective_len
            quantized_tokens = 0
        else:
            # If start == 0, entire cache is quantized
            if start == 0:
                unquantized_tokens = 0
                quantized_tokens = effective_len
            else:
                unquantized_tokens = start
                quantized_tokens = effective_len - start

        unquantized_bytes = unquantized_tokens * fp16_bytes_per_token

        # Quantized data: bits / 8 bytes per element
        # In MLX, quantized values are stored packed in uint32
        data_bytes_per_token = int(elements_per_token * (bits / 8.0))
        # Scales and biases: FP16 (2 bytes each) per group
        scales_bytes_per_token = int(elements_per_token * (2.0 / group_size))
        biases_bytes_per_token = int(elements_per_token * (2.0 / group_size))

        quant_data_bytes = quantized_tokens * data_bytes_per_token
        quant_scale_bytes = quantized_tokens * scales_bytes_per_token
        quant_bias_bytes = quantized_tokens * biases_bytes_per_token

        total_bytes = unquantized_bytes + quant_data_bytes + quant_scale_bytes + quant_bias_bytes

        return MemoryEstimate(
            sequence_length=sequence_length,
            padded_sequence_length=effective_len,
            precision=cache_config.precision,
            bits=bits,
            group_size=group_size,
            quantized_start=start,
            unquantized_tokens=unquantized_tokens,
            quantized_tokens=quantized_tokens,
            unquantized_bytes=unquantized_bytes,
            quantized_data_bytes=quant_data_bytes,
            quantized_scale_bytes=quant_scale_bytes,
            quantized_bias_bytes=quant_bias_bytes,
            total_bytes=total_bytes,
            total_mb=total_bytes / (1024.0 * 1024.0),
        )
