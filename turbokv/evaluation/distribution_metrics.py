"""Next-token distribution divergence (KL divergence) and top-1 agreement analysis."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import mlx.core as mx


@dataclass
class TokenStepDivergence:
    """Metrics for a single token decoding step."""

    step: int
    kl_divergence: float
    top1_agreement: bool
    reference_top1_token: int
    candidate_top1_token: int


@dataclass
class DistributionDivergenceReport:
    """Aggregated distribution divergence report over generated sequence."""

    mean_kl_divergence: float
    max_kl_divergence: float
    top1_agreement_rate: float
    total_steps: int
    step_metrics: List[TokenStepDivergence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_kl_divergence": self.mean_kl_divergence,
            "max_kl_divergence": self.max_kl_divergence,
            "top1_agreement_rate": self.top1_agreement_rate,
            "total_steps": self.total_steps,
            "step_metrics": [
                {
                    "step": s.step,
                    "kl_divergence": s.kl_divergence,
                    "top1_agreement": s.top1_agreement,
                    "reference_top1_token": s.reference_top1_token,
                    "candidate_top1_token": s.candidate_top1_token,
                }
                for s in self.step_metrics
            ],
        }


def compute_distribution_metrics(
    reference_logprobs: List[mx.array],
    candidate_logprobs: List[mx.array],
) -> DistributionDivergenceReport:
    """
    Computes step-by-step KL divergence D_KL(P_ref || P_cand) and top-1 token agreement.
    P_ref is the baseline FP16 distribution; P_cand is the quantized cache distribution.
    """
    num_steps = min(len(reference_logprobs), len(candidate_logprobs))
    if num_steps == 0:
        return DistributionDivergenceReport(0.0, 0.0, 0.0, 0)

    step_metrics: List[TokenStepDivergence] = []
    kl_values: List[float] = []
    top1_matches: List[bool] = []

    for step in range(num_steps):
        # Convert to numpy for stable scipy/numpy calculations
        ref_lp = np.array(reference_logprobs[step].tolist(), dtype=np.float64)
        cand_lp = np.array(candidate_logprobs[step].tolist(), dtype=np.float64)

        # Normalize logprobs to probabilities
        ref_p = np.exp(ref_lp - np.max(ref_lp))
        ref_p = ref_p / np.sum(ref_p)

        cand_p = np.exp(cand_lp - np.max(cand_lp))
        cand_p = cand_p / np.sum(cand_p)

        # Safe KL divergence: sum(P * (log(P) - log(Q)))
        # Add epsilon to prevent log(0)
        eps = 1e-12
        ref_p_safe = np.clip(ref_p, eps, 1.0)
        cand_p_safe = np.clip(cand_p, eps, 1.0)
        kl = float(np.sum(ref_p_safe * (np.log(ref_p_safe) - np.log(cand_p_safe))))
        # Clamp negative numerical precision jitter to 0
        kl = max(0.0, kl)

        ref_top1 = int(np.argmax(ref_lp))
        cand_top1 = int(np.argmax(cand_lp))
        agreement = (ref_top1 == cand_top1)

        kl_values.append(kl)
        top1_matches.append(agreement)

        step_metrics.append(
            TokenStepDivergence(
                step=step,
                kl_divergence=kl,
                top1_agreement=agreement,
                reference_top1_token=ref_top1,
                candidate_top1_token=cand_top1,
            )
        )

    mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
    max_kl = float(np.max(kl_values)) if kl_values else 0.0
    top1_rate = float(np.mean(top1_matches)) if top1_matches else 0.0

    return DistributionDivergenceReport(
        mean_kl_divergence=mean_kl,
        max_kl_divergence=max_kl,
        top1_agreement_rate=top1_rate,
        total_steps=num_steps,
        step_metrics=step_metrics,
    )
