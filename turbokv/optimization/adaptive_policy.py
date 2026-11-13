"""TurboKV workload-aware adaptive KV cache policy selector."""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from turbokv.config.policy_config import PolicyConstraints, CacheConfig
from turbokv.optimization.memory_estimator import MemoryEstimator
from turbokv.optimization.constraints import CandidateEvaluation


@dataclass
class PolicyDecision:
    """Explainable result of an adaptive policy evaluation."""

    context_length: int
    memory_budget_mb: float
    chosen_config: Optional[CacheConfig]
    is_feasible: bool
    fallback_to_fp16: bool
    decision_reason: str
    expected_memory_mb: float
    expected_accuracy: Optional[float]
    expected_tokens_per_sec: Optional[float]
    evaluated_candidates: List[CandidateEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_length": self.context_length,
            "memory_budget_mb": self.memory_budget_mb,
            "chosen_config": self.chosen_config.to_dict() if self.chosen_config else None,
            "is_feasible": self.is_feasible,
            "fallback_to_fp16": self.fallback_to_fp16,
            "decision_reason": self.decision_reason,
            "expected_memory_mb": self.expected_memory_mb,
            "expected_accuracy": self.expected_accuracy,
            "expected_tokens_per_sec": self.expected_tokens_per_sec,
            "evaluated_candidates": [c.to_dict() for c in self.evaluated_candidates],
        }


class AdaptiveKVPolicy:
    """
    Constrained memory-adaptive KV cache selection engine.
    Uses empirical calibration data and theoretical memory bounds to select
    the optimal cache configuration for any given context length and memory budget.
    """

    def __init__(
        self,
        calibration_data: Optional[Dict[str, Any]] = None,
        memory_estimator: Optional[MemoryEstimator] = None,
    ):
        self.calibration_data = calibration_data or {}
        self.estimator = memory_estimator or MemoryEstimator()

    @classmethod
    def from_calibration_file(cls, filepath: str) -> "AdaptiveKVPolicy":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(calibration_data=data)

    def _get_calibration_metrics(
        self, context_length: int, config_key: str
    ) -> Dict[str, Any]:
        """Lookup measured calibration results for a context length and configuration key."""
        # Check direct match
        ctx_str = str(context_length)
        by_ctx = self.calibration_data.get("by_context_length", {})
        if ctx_str in by_ctx and config_key in by_ctx[ctx_str]:
            return by_ctx[ctx_str][config_key]

        # Nearest context length fallback in calibration set
        if by_ctx:
            available_ctxs = [int(k) for k in by_ctx.keys() if k.isdigit()]
            if available_ctxs:
                closest_ctx = min(available_ctxs, key=lambda x: abs(x - context_length))
                return by_ctx[str(closest_ctx)].get(config_key, {})

        return {}

    def select(
        self,
        context_length: int,
        constraints: PolicyConstraints,
        candidate_configs: Optional[List[CacheConfig]] = None,
    ) -> PolicyDecision:
        """
        Evaluate candidate cache configurations against constraints and choose
        the configuration that maximizes memory savings while strictly satisfying
        quality and latency requirements.
        """
        if candidate_configs is None:
            # Default candidate search space
            candidate_configs = [
                CacheConfig("fp16"),
                CacheConfig("int8", group_size=64, quantized_start=0),
                CacheConfig("int8", group_size=32, quantized_start=0),
                CacheConfig("int8", group_size=64, quantized_start=512),
                CacheConfig("int4", group_size=64, quantized_start=0),
                CacheConfig("int4", group_size=32, quantized_start=0),
                CacheConfig("int4", group_size=64, quantized_start=512),
            ]

        # Get baseline FP16 metrics for calibration comparison
        fp16_calib = self._get_calibration_metrics(context_length, "fp16")
        baseline_acc = fp16_calib.get("retrieval_accuracy", 1.0)
        baseline_tps = fp16_calib.get("decode_tokens_per_sec", 40.0)

        evaluations: List[CandidateEvaluation] = []
        feasible_candidates: List[CandidateEvaluation] = []

        for cand in candidate_configs:
            # 1. Theoretical memory calculation
            mem_est = self.estimator.estimate(context_length, cand)
            calib = self._get_calibration_metrics(context_length, cand.key)

            measured_mem = calib.get("measured_cache_mb")
            effective_mem = measured_mem if measured_mem is not None else mem_est.total_mb

            cand_acc = calib.get("retrieval_accuracy")
            cand_tps = calib.get("decode_tokens_per_sec")

            rejection_reasons = []

            # Check memory budget
            fits_memory = effective_mem <= constraints.max_memory_budget_mb
            if not fits_memory:
                rejection_reasons.append(
                    f"Memory {effective_mem:.1f} MB exceeds budget {constraints.max_memory_budget_mb:.1f} MB"
                )

            # Check quality constraint
            acc_drop = 0.0
            satisfies_quality = True
            if cand_acc is not None:
                acc_drop = max(0.0, baseline_acc - cand_acc)
                if acc_drop > constraints.max_accuracy_drop:
                    satisfies_quality = False
                    rejection_reasons.append(
                        f"Accuracy drop {acc_drop:.3f} exceeds maximum allowed drop {constraints.max_accuracy_drop:.3f}"
                    )
            elif cand.precision.lower() != "fp16" and self.calibration_data:
                # If calibration data exists but this quantized candidate is uncalibrated, reject it as unverified
                satisfies_quality = False
                rejection_reasons.append("Uncalibrated quantized configuration; quality preservation unverified")

            # Check latency constraint
            latency_regr_pct = 0.0
            satisfies_latency = True
            if cand_tps is not None and baseline_tps > 0:
                if cand_tps < baseline_tps:
                    latency_regr_pct = ((baseline_tps - cand_tps) / baseline_tps) * 100.0
                if latency_regr_pct > constraints.max_latency_regression_pct:
                    satisfies_latency = False
                    rejection_reasons.append(
                        f"Latency slowdown {latency_regr_pct:.1f}% exceeds max regression {constraints.max_latency_regression_pct:.1f}%"
                    )

            if constraints.min_tokens_per_second is not None and cand_tps is not None:
                if cand_tps < constraints.min_tokens_per_second:
                    satisfies_latency = False
                    rejection_reasons.append(
                        f"Throughput {cand_tps:.1f} tok/s below minimum required {constraints.min_tokens_per_second:.1f}"
                    )

            is_feas = fits_memory and satisfies_quality and satisfies_latency

            eval_res = CandidateEvaluation(
                cache_config=cand,
                estimated_memory_mb=mem_est.total_mb,
                measured_memory_mb=measured_mem,
                retrieval_accuracy=cand_acc,
                decode_tokens_per_sec=cand_tps,
                accuracy_drop=acc_drop,
                latency_regression_pct=latency_regr_pct,
                fits_memory_budget=fits_memory,
                satisfies_quality_constraint=satisfies_quality,
                satisfies_latency_constraint=satisfies_latency,
                is_feasible=is_feas,
                rejection_reasons=rejection_reasons,
            )
            evaluations.append(eval_res)
            if is_feas:
                feasible_candidates.append(eval_res)

        # Selection logic
        # 1. First check feasible QUANTIZED candidates (objective is memory reduction)
        quantized_feasible = [c for c in feasible_candidates if c.cache_config.precision.lower() != "fp16"]

        if quantized_feasible:
            # Sort primarily by lowest memory consumption (maximizing savings)
            # Secondarily by highest accuracy, then highest tokens per second
            def sort_key(c: CandidateEvaluation):
                mem = c.measured_memory_mb if c.measured_memory_mb is not None else c.estimated_memory_mb
                acc = c.retrieval_accuracy if c.retrieval_accuracy is not None else 0.0
                tps = c.decode_tokens_per_sec if c.decode_tokens_per_sec is not None else 0.0
                return (round(mem, 1), -round(acc, 3), -round(tps, 1))

            quantized_feasible.sort(key=sort_key)
            best = quantized_feasible[0]

            reason = (
                f"Selected optimal configuration {best.cache_config.key}: "
                f"Fits budget with {best.estimated_memory_mb:.1f} MB (vs budget {constraints.max_memory_budget_mb:.1f} MB), "
                f"accuracy drop {best.accuracy_drop:.3f} <= {constraints.max_accuracy_drop:.3f}, "
                f"latency regression {best.latency_regression_pct:.1f}% <= {constraints.max_latency_regression_pct:.1f}%."
            )
            return PolicyDecision(
                context_length=context_length,
                memory_budget_mb=constraints.max_memory_budget_mb,
                chosen_config=best.cache_config,
                is_feasible=True,
                fallback_to_fp16=False,
                decision_reason=reason,
                expected_memory_mb=best.estimated_memory_mb,
                expected_accuracy=best.retrieval_accuracy,
                expected_tokens_per_sec=best.decode_tokens_per_sec,
                evaluated_candidates=evaluations,
            )

        # 2. If no quantized candidate was feasible:
        # Check if FP16 is feasible within memory budget
        fp16_cand = next((c for c in evaluations if c.cache_config.precision.lower() == "fp16"), None)
        if fp16_cand and fp16_cand.fits_memory_budget:
            reason = (
                f"Fallback to FP16: No quantized configuration satisfied quality/latency constraints, "
                f"but FP16 fits within memory budget ({fp16_cand.estimated_memory_mb:.1f} MB <= {constraints.max_memory_budget_mb:.1f} MB)."
            )
            return PolicyDecision(
                context_length=context_length,
                memory_budget_mb=constraints.max_memory_budget_mb,
                chosen_config=fp16_cand.cache_config,
                is_feasible=True,
                fallback_to_fp16=True,
                decision_reason=reason,
                expected_memory_mb=fp16_cand.estimated_memory_mb,
                expected_accuracy=fp16_cand.retrieval_accuracy,
                expected_tokens_per_sec=fp16_cand.decode_tokens_per_sec,
                evaluated_candidates=evaluations,
            )

        # Workload is strictly infeasible
        reasons_summary = "; ".join(
            [f"{e.cache_config.key}: {', '.join(e.rejection_reasons)}" for e in evaluations]
        )
        reason = (
            f"Workload infeasible: No cache configuration satisfied all constraints within memory budget "
            f"of {constraints.max_memory_budget_mb:.1f} MB. Details: {reasons_summary}"
        )
        return PolicyDecision(
            context_length=context_length,
            memory_budget_mb=constraints.max_memory_budget_mb,
            chosen_config=None,
            is_feasible=False,
            fallback_to_fp16=False,
            decision_reason=reason,
            expected_memory_mb=0.0,
            expected_accuracy=None,
            expected_tokens_per_sec=None,
            evaluated_candidates=evaluations,
        )
