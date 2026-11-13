"""Retrieval evaluation metrics: Exact Match, Normalized Exact Match, and stratified accuracy."""

import re
import string
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from turbokv.datasets.retrieval_dataset import RetrievalItem


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and collapsing whitespace."""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Collapse whitespace
    tokens = text.strip().split()
    return " ".join(tokens)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Strict or substring exact match."""
    norm_pred = prediction.strip()
    norm_gt = ground_truth.strip()
    return norm_gt in norm_pred or norm_pred == norm_gt


def compute_normalized_match(prediction: str, ground_truth: str) -> bool:
    """Normalized exact match after lowercase, punctuation removal and whitespace normalization."""
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)
    if not norm_gt:
        return False
    return norm_gt in norm_pred or norm_pred == norm_gt


@dataclass
class RetrievalResult:
    """Evaluation result for a single retrieval instance."""

    item_id: str
    context_tokens: int
    fact_position: float
    position_bucket: str
    prediction: str
    ground_truth: str
    exact_match: bool
    normalized_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "context_tokens": self.context_tokens,
            "fact_position": self.fact_position,
            "position_bucket": self.position_bucket,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "exact_match": self.exact_match,
            "normalized_match": self.normalized_match,
        }


@dataclass
class RetrievalReport:
    """Aggregated retrieval evaluation report with stratification by context length and position."""

    overall_accuracy: float
    exact_match_accuracy: float
    total_samples: int
    accuracy_by_context_length: Dict[int, float] = field(default_factory=dict)
    accuracy_by_position: Dict[str, float] = field(default_factory=dict)
    detailed_results: List[RetrievalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_accuracy": self.overall_accuracy,
            "exact_match_accuracy": self.exact_match_accuracy,
            "total_samples": self.total_samples,
            "accuracy_by_context_length": {str(k): v for k, v in self.accuracy_by_context_length.items()},
            "accuracy_by_position": self.accuracy_by_position,
            "detailed_results": [r.to_dict() for r in self.detailed_results],
        }


def evaluate_retrieval(
    items: List[RetrievalItem],
    predictions: List[str],
) -> RetrievalReport:
    """Aggregates retrieval accuracy across items and predictions."""
    if len(items) != len(predictions):
        raise ValueError(f"Mismatch between items count ({len(items)}) and predictions count ({len(predictions)})")

    results: List[RetrievalResult] = []
    by_ctx: Dict[int, List[bool]] = {}
    by_pos: Dict[str, List[bool]] = {}

    for item, pred in zip(items, predictions):
        em = compute_exact_match(pred, item.ground_truth_answer)
        nm = compute_normalized_match(pred, item.ground_truth_answer)

        res = RetrievalResult(
            item_id=item.item_id,
            context_tokens=item.context_tokens_target,
            fact_position=item.fact_position_fraction,
            position_bucket=item.position_bucket,
            prediction=pred,
            ground_truth=item.ground_truth_answer,
            exact_match=em,
            normalized_match=nm,
        )
        results.append(res)

        by_ctx.setdefault(item.context_tokens_target, []).append(nm)
        by_pos.setdefault(item.position_bucket, []).append(nm)

    total = len(results)
    if total == 0:
        return RetrievalReport(0.0, 0.0, 0)

    overall_acc = sum(r.normalized_match for r in results) / total
    em_acc = sum(r.exact_match for r in results) / total

    acc_by_ctx = {ctx: sum(matches) / len(matches) for ctx, matches in by_ctx.items()}
    acc_by_pos = {pos: sum(matches) / len(matches) for pos, matches in by_pos.items()}

    return RetrievalReport(
        overall_accuracy=overall_acc,
        exact_match_accuracy=em_acc,
        total_samples=total,
        accuracy_by_context_length=acc_by_ctx,
        accuracy_by_position=acc_by_pos,
        detailed_results=results,
    )
