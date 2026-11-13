"""Model architecture and loading configuration."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for LLM model and its architectural parameters."""

    model_path: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    fallback_model_path: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    model_name: str = "Qwen2.5-1.5B-Instruct-4bit"
    max_context_length: int = 32768
    num_layers: int = 28
    num_kv_heads: int = 2
    head_dim: int = 128
    hidden_size: int = 1536
    num_attention_heads: int = 12
    vocab_size: int = 151936
    weight_bits: int = 4
    trust_remote_code: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def theoretical_kv_bytes_per_token(self, bits_per_element: float = 16.0, group_size: int = 64) -> float:
        """
        Theoretical KV cache size per token in bytes.
        KV cache bytes per token = 2 * num_layers * num_kv_heads * head_dim * (bits / 8).
        If quantized, accounts for scale and bias overhead:
        For group quantization with group_size:
        scale (FP16 = 2 bytes) and bias (FP16 = 2 bytes) per group of elements:
        4 bytes / group_size per element.
        """
        num_elements_per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim
        base_bytes = num_elements_per_token * (bits_per_element / 8.0)
        if bits_per_element < 16.0 and group_size > 0:
            # MLX quantize stores scales and biases (FP16) per group
            scale_bias_overhead = num_elements_per_token * (4.0 / group_size)
            return base_bytes + scale_bias_overhead
        return base_bytes

    def theoretical_kv_cache_bytes(
        self, sequence_length: int, bits_per_element: float = 16.0, group_size: int = 64
    ) -> int:
        """Calculate total theoretical KV cache memory for a given sequence length."""
        per_token = self.theoretical_kv_bytes_per_token(bits_per_element=bits_per_element, group_size=group_size)
        return int(per_token * sequence_length)
