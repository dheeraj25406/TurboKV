"""TurboKV adaptive policy configuration and constraint definitions."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class PolicyConstraints:
    """Constraints for adaptive KV cache selection."""

    max_memory_budget_mb: float = 1000.0  # Max permitted KV cache memory in MB
    max_accuracy_drop: float = 0.05  # Maximum allowed retrieval accuracy drop (e.g. 5% relative or absolute)
    max_latency_regression_pct: float = 20.0  # Maximum allowed % decode latency slowdown vs FP16
    min_tokens_per_second: Optional[float] = None  # Absolute throughput floor if specified
    prefer_higher_precision_on_tie: bool = True


@dataclass
class CacheConfig:
    """Specific cache configuration."""

    precision: str  # 'fp16', 'int8', 'int4'
    bits: Optional[int] = None  # None for fp16, 8 for int8, 4 for int4
    group_size: int = 64
    quantized_start: int = 0  # Token onset for quantization

    def __post_init__(self):
        if self.precision.lower() == "fp16":
            self.bits = None
        elif self.precision.lower() == "int8":
            self.bits = 8
        elif self.precision.lower() == "int4":
            self.bits = 4
        else:
            raise ValueError(f"Unsupported cache precision: {self.precision}")

    @property
    def key(self) -> str:
        if self.bits is None:
            return "fp16"
        return f"{self.precision}_g{self.group_size}_s{self.quantized_start}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "bits": self.bits,
            "group_size": self.group_size,
            "quantized_start": self.quantized_start,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        return cls(
            precision=data["precision"],
            bits=data.get("bits"),
            group_size=data.get("group_size", 64),
            quantized_start=data.get("quantized_start", 0),
        )
