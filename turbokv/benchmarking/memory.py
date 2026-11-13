"""Memory profiling tools for process RSS and Apple Silicon Unified Memory (Metal)."""

import os
import psutil
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import mlx.core as mx


@dataclass
class MemorySnapshot:
    """Snapshot of process and Metal memory metrics."""

    timestamp_s: float
    process_rss_mb: float
    metal_active_mb: float
    metal_peak_mb: float
    metal_cache_mb: float
    kv_cache_measured_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryProfiler:
    """
    Tracks and isolates memory usage across process RSS and Apple Silicon Metal allocations.
    Distinguishes process-level memory, Metal active memory, and direct KV cache tensor allocations.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_process_rss_mb(self) -> float:
        """Process resident set size in megabytes."""
        return self.process.memory_info().rss / (1024.0 * 1024.0)

    def get_metal_active_mb(self) -> float:
        """Metal unified memory currently actively allocated in megabytes."""
        if hasattr(mx, "get_active_memory"):
            return mx.get_active_memory() / (1024.0 * 1024.0)
        elif hasattr(mx.metal, "get_active_memory"):
            return mx.metal.get_active_memory() / (1024.0 * 1024.0)
        return 0.0

    def get_metal_peak_mb(self) -> float:
        """Metal unified memory peak allocation in megabytes."""
        if hasattr(mx, "get_peak_memory"):
            return mx.get_peak_memory() / (1024.0 * 1024.0)
        elif hasattr(mx.metal, "get_peak_memory"):
            return mx.metal.get_peak_memory() / (1024.0 * 1024.0)
        return 0.0

    def get_metal_cache_mb(self) -> float:
        """Metal memory pool cache in megabytes."""
        if hasattr(mx, "get_cache_memory"):
            return mx.get_cache_memory() / (1024.0 * 1024.0)
        elif hasattr(mx.metal, "get_cache_memory"):
            return mx.metal.get_cache_memory() / (1024.0 * 1024.0)
        return 0.0

    def reset_peak_memory(self):
        """Reset Metal peak memory counter."""
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

    def clear_metal_cache(self):
        """Request Metal allocator to release cached unused memory buffers."""
        if hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

    @staticmethod
    def measure_cache_bytes(prompt_cache: Optional[List[Any]]) -> int:
        """
        Directly measures the exact byte storage of the KV cache buffers in MLX.
        Reads .nbytes from each layer's cache object.
        """
        if prompt_cache is None:
            return 0
        total_bytes = 0
        for layer_cache in prompt_cache:
            if hasattr(layer_cache, "nbytes"):
                total_bytes += layer_cache.nbytes
            elif hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                # Fallback for arbitrary cache structures
                if isinstance(layer_cache.keys, (tuple, list)):
                    total_bytes += sum(k.nbytes for k in layer_cache.keys if hasattr(k, "nbytes"))
                    total_bytes += sum(v.nbytes for v in layer_cache.values if hasattr(v, "nbytes"))
                elif hasattr(layer_cache.keys, "nbytes"):
                    total_bytes += layer_cache.keys.nbytes + layer_cache.values.nbytes
        return total_bytes

    def capture_snapshot(self, prompt_cache: Optional[List[Any]] = None) -> MemorySnapshot:
        """Capture comprehensive memory state snapshot."""
        import time
        kv_bytes = self.measure_cache_bytes(prompt_cache)
        return MemorySnapshot(
            timestamp_s=time.time(),
            process_rss_mb=self.get_process_rss_mb(),
            metal_active_mb=self.get_metal_active_mb(),
            metal_peak_mb=self.get_metal_peak_mb(),
            metal_cache_mb=self.get_metal_cache_mb(),
            kv_cache_measured_mb=kv_bytes / (1024.0 * 1024.0),
        )
