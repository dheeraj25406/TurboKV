"""Statistical aggregation utilities for latency, memory, and quality benchmarks."""

import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class MetricSummary:
    """Statistical summary of repeated measurements."""

    count: int
    mean: float
    median: float
    p95: float
    p99: float
    min: float
    max: float
    std_dev: float
    iqr: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_statistics(values: List[float]) -> MetricSummary:
    """Computes comprehensive statistical metrics on a sample of values."""
    if not values:
        return MetricSummary(
            count=0,
            mean=0.0,
            median=0.0,
            p95=0.0,
            p99=0.0,
            min=0.0,
            max=0.0,
            std_dev=0.0,
            iqr=0.0,
            ci95_low=0.0,
            ci95_high=0.0,
        )

    arr = np.array(values, dtype=float)
    count = int(len(arr))
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    std_dev = float(np.std(arr, ddof=1)) if count > 1 else 0.0
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)

    # 95% confidence interval for mean (t-distribution approximation or normal)
    if count > 1 and std_dev > 0:
        # Standard error
        se = std_dev / math.sqrt(count)
        # Use 1.96 approximation for CI
        margin = 1.96 * se
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = mean
        ci_high = mean

    return MetricSummary(
        count=count,
        mean=mean,
        median=median,
        p95=p95,
        p99=p99,
        min=min_val,
        max=max_val,
        std_dev=std_dev,
        iqr=iqr,
        ci95_low=ci_low,
        ci95_high=ci_high,
    )
