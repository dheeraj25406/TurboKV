"""Quality report combining retrieval, perplexity, and distribution divergence."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from turbokv.evaluation.retrieval import RetrievalReport
from turbokv.evaluation.distribution_metrics import DistributionDivergenceReport


@dataclass
class QualityReport:
    """Comprehensive quality evaluation report for a cache configuration."""

    cache_config_key: str
    retrieval_report: RetrievalReport
    perplexity: Optional[float] = None
    perplexity_loss: Optional[float] = None
    distribution_divergence: Optional[DistributionDivergenceReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_config_key": self.cache_config_key,
            "overall_accuracy": self.retrieval_report.overall_accuracy,
            "exact_match_accuracy": self.retrieval_report.exact_match_accuracy,
            "accuracy_by_context_length": self.retrieval_report.accuracy_by_context_length,
            "accuracy_by_position": self.retrieval_report.accuracy_by_position,
            "perplexity": self.perplexity,
            "perplexity_loss": self.perplexity_loss,
            "distribution_divergence": (
                self.distribution_divergence.to_dict()
                if self.distribution_divergence
                else None
            ),
        }
