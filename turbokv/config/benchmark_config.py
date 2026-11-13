"""Benchmark configuration and execution parameters."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Benchmark suite configuration."""

    context_lengths: List[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192, 16384])
    generation_tokens: int = 128
    warmup_runs: int = 1
    num_repetitions: int = 3
    random_seed: int = 42
    temperature: float = 0.0  # Greedy decoding for deterministic evaluation
    prefill_step_size: int = 2048

    # Quantization configurations to evaluate
    supported_precisions: List[str] = field(default_factory=lambda: ["fp16", "int8", "int4"])
    supported_group_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    supported_quantized_starts: List[int] = field(default_factory=lambda: [0, 512, 1024, 2048])

    # Memory limits (safety thresholds for 16GB unified memory machine)
    max_process_rss_mb: float = 12000.0  # 12 GB safety threshold
    output_dir: str = "results"
