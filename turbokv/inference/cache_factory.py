"""KV Cache factory and management for FP16, INT8, and INT4 precision."""

from typing import List, Optional, Any, Dict
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache
from turbokv.config.policy_config import CacheConfig


class CacheFactory:
    """Creates and configures KV caches for MLX models."""

    @staticmethod
    def create_cache(
        model: nn.Module,
        cache_config: CacheConfig,
    ) -> List[Any]:
        """
        Instantiate the appropriate cache list for the model layers.
        If cache_config has quantized_start == 0 and bits is not None:
            pre-allocates QuantizedKVCache directly.
        If quantized_start > 0 and bits is not None:
            starts as KVCache, and converts via to_quantized once offset >= quantized_start.
        If bits is None:
            allocates standard FP16 KVCache.
        """
        num_layers = len(model.layers)

        if cache_config.bits is None or cache_config.precision.lower() == "fp16":
            return [KVCache() for _ in range(num_layers)]

        # Quantized cache
        if cache_config.quantized_start == 0:
            return [
                QuantizedKVCache(
                    group_size=cache_config.group_size,
                    bits=cache_config.bits,
                )
                for _ in range(num_layers)
            ]
        else:
            # Delayed quantization onset starts in FP16
            return [KVCache() for _ in range(num_layers)]


class CacheInspector:
    """Inspects live KV cache state and memory allocations across transformer layers."""

    @staticmethod
    def inspect_cache(prompt_cache: Optional[List[Any]]) -> Dict[str, Any]:
        if prompt_cache is None:
            return {"num_layers": 0, "total_bytes": 0, "total_mb": 0.0, "layers": []}

        total_bytes = 0
        layer_details = []
        for i, layer in enumerate(prompt_cache):
            layer_type = type(layer).__name__
            offset = getattr(layer, "offset", 0)
            nbytes = getattr(layer, "nbytes", 0)
            total_bytes += nbytes

            info = {
                "layer_idx": i,
                "type": layer_type,
                "offset": offset,
                "nbytes": nbytes,
                "mb": nbytes / (1024.0 * 1024.0),
            }
            if hasattr(layer, "bits"):
                info["bits"] = layer.bits
            if hasattr(layer, "group_size"):
                info["group_size"] = layer.group_size

            layer_details.append(info)

        return {
            "num_layers": len(prompt_cache),
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024.0 * 1024.0),
            "layers": layer_details,
        }
