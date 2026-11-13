"""Constraint validation and candidate filtering for KV cache selection."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from turbokv.config.policy_config import PolicyConstraints, CacheConfig


@dataclass
class CandidateEvaluation:
    """Evaluation of a specific cache configuration against user constraints."""

    cache_config: CacheConfig
    estimated_memory_mb: float
    measured_memory_mb: Optional[float]
    retrieval_accuracy: Optional[float]
    decode_tokens_per_sec: Optional[float]
    accuracy_drop: float
    latency_regression_pct: float
    fits_memory_budget: bool
    satisfies_quality_constraint: bool
    satisfies_latency_constraint: bool
    is_feasible: bool
    rejection_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_config": self.cache_config.to_dict(),
            "estimated_memory_mb": self.estimated_memory_mb,
            "measured_memory_mb": self.measured_memory_mb,
            "retrieval_accuracy": self.retrieval_accuracy,
            "decode_tokens_per_sec": self.decode_tokens_per_sec,
            "accuracy_drop": self.accuracy_drop,
            "latency_regression_pct": self.latency_regression_pct,
            "fits_memory_budget": self.fits_memory_budget,
            "satisfies_quality_constraint": self.satisfies_quality_constraint,
            "satisfies_latency_constraint": self.satisfies_latency_constraint,
            "is_feasible": self.is_feasible,
            "rejection_reasons": self.rejection_reasons,
        }
