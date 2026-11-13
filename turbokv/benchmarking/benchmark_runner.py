"""Comprehensive benchmark runner for KV cache performance and memory evaluation."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from turbokv.config.model_config import ModelConfig
from turbokv.config.benchmark_config import BenchmarkConfig
from turbokv.config.policy_config import CacheConfig
from turbokv.inference.generation import GenerationEngine, GenerationResult
from turbokv.optimization.memory_estimator import MemoryEstimator
from turbokv.benchmarking.statistics import compute_statistics, MetricSummary


@dataclass
class AggregatedBenchmarkResult:
    """Aggregated benchmark metrics across multiple repetitions."""

    context_length: int
    cache_config: CacheConfig
    repetitions: int
    prompt_tokens: int
    generation_tokens: int

    # Latencies (ms)
    ttft_summary: MetricSummary
    decode_ms_summary: MetricSummary
    total_ms_summary: MetricSummary

    # Throughput (tokens/sec)
    decode_tps_summary: MetricSummary
    total_tps_summary: MetricSummary

    # Memory (MB)
    measured_cache_mb_summary: MetricSummary
    peak_rss_mb_summary: MetricSummary
    metal_active_mb_summary: MetricSummary
    theoretical_cache_mb: float

    # Raw per-run measurements
    raw_runs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_length": self.context_length,
            "cache_config": self.cache_config.to_dict(),
            "repetitions": self.repetitions,
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.generation_tokens,
            "ttft_ms": self.ttft_summary.to_dict(),
            "decode_ms": self.decode_ms_summary.to_dict(),
            "total_ms": self.total_ms_summary.to_dict(),
            "decode_tps": self.decode_tps_summary.to_dict(),
            "total_tps": self.total_tps_summary.to_dict(),
            "measured_cache_mb": self.measured_cache_mb_summary.to_dict(),
            "peak_rss_mb": self.peak_rss_mb_summary.to_dict(),
            "metal_active_mb": self.metal_active_mb_summary.to_dict(),
            "theoretical_cache_mb": self.theoretical_cache_mb,
            "raw_runs": self.raw_runs,
        }


class BenchmarkRunner:
    """Executes repeatable, rigorous benchmarks across context lengths and cache precisions."""

    def __init__(
        self,
        model,
        tokenizer,
        model_config: Optional[ModelConfig] = None,
        benchmark_config: Optional[BenchmarkConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config or ModelConfig()
        self.config = benchmark_config or BenchmarkConfig()
        self.engine = GenerationEngine(self.model, self.tokenizer)
        self.estimator = MemoryEstimator(self.model_config)

    def _generate_synthetic_prompt(self, target_tokens: int) -> str:
        """Generates deterministic prompt of exact target token length."""
        # Use simple repetitive sentence pattern
        base = "The quick brown fox jumps over the lazy dog near the river bank. "
        prompt = ""
        while True:
            prompt += base
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            if len(ids) >= target_tokens:
                # Trim tokens exactly to target_tokens
                trimmed_ids = ids[:target_tokens]
                detok = self.tokenizer.detokenizer
                detok.reset()
                for t in trimmed_ids:
                    detok.add_token(t)
                detok.finalize()
                return detok.text

    def run_benchmark(
        self,
        context_lengths: Optional[List[int]] = None,
        cache_configs: Optional[List[CacheConfig]] = None,
        num_repetitions: Optional[int] = None,
        generation_tokens: Optional[int] = None,
    ) -> List[AggregatedBenchmarkResult]:
        ctx_lens = context_lengths or self.config.context_lengths
        configs = cache_configs or [
            CacheConfig("fp16"),
            CacheConfig("int8", group_size=64, quantized_start=0),
            CacheConfig("int4", group_size=64, quantized_start=0),
        ]
        reps = num_repetitions or self.config.num_repetitions
        gen_toks = generation_tokens or self.config.generation_tokens

        results: List[AggregatedBenchmarkResult] = []

        # Warmup model once
        print("[BenchmarkRunner] Executing initial GPU warmup...")
        warmup_prompt = "Hello world warmup"
        self.engine.generate(warmup_prompt, CacheConfig("fp16"), max_tokens=16)

        for ctx_len in ctx_lens:
            print(f"\n==================================================")
            print(f"[BenchmarkRunner] Benchmarking Context Length: {ctx_len}")
            print(f"==================================================")
            prompt = self._generate_synthetic_prompt(ctx_len)
            actual_prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=True))

            for cfg in configs:
                print(f"[BenchmarkRunner] Configuration: {cfg.key} (reps={reps})...")
                # Warmup run for this configuration
                self.engine.generate(prompt, cfg, max_tokens=16)

                raw_runs: List[Dict[str, Any]] = []
                ttft_vals: List[float] = []
                decode_ms_vals: List[float] = []
                total_ms_vals: List[float] = []
                decode_tps_vals: List[float] = []
                total_tps_vals: List[float] = []
                cache_mb_vals: List[float] = []
                peak_rss_vals: List[float] = []
                metal_active_vals: List[float] = []

                for r in range(reps):
                    res = self.engine.generate(
                        prompt=prompt,
                        cache_config=cfg,
                        max_tokens=gen_toks,
                        capture_logprobs=False,
                    )
                    raw_runs.append(res.to_dict(include_tensors=False))
                    ttft_vals.append(res.ttft_ms)
                    decode_ms_vals.append(res.decode_ms)
                    total_ms_vals.append(res.total_latency_ms)
                    decode_tps_vals.append(res.decode_tokens_per_sec)
                    total_tps_vals.append(res.total_tokens_per_sec)
                    cache_mb_vals.append(res.kv_cache_measured_mb)
                    peak_rss_vals.append(res.peak_rss_mb)
                    metal_active_vals.append(res.metal_active_mb)

                mem_est = self.estimator.estimate(ctx_len + gen_toks, cfg)

                agg = AggregatedBenchmarkResult(
                    context_length=ctx_len,
                    cache_config=cfg,
                    repetitions=reps,
                    prompt_tokens=actual_prompt_tokens,
                    generation_tokens=gen_toks,
                    ttft_summary=compute_statistics(ttft_vals),
                    decode_ms_summary=compute_statistics(decode_ms_vals),
                    total_ms_summary=compute_statistics(total_ms_vals),
                    decode_tps_summary=compute_statistics(decode_tps_vals),
                    total_tps_summary=compute_statistics(total_tps_vals),
                    measured_cache_mb_summary=compute_statistics(cache_mb_vals),
                    peak_rss_mb_summary=compute_statistics(peak_rss_vals),
                    metal_active_mb_summary=compute_statistics(metal_active_vals),
                    theoretical_cache_mb=mem_est.total_mb,
                    raw_runs=raw_runs,
                )
                results.append(agg)
                print(
                    f"  -> Cache: {agg.measured_cache_mb_summary.median:.2f} MB "
                    f"(Theo: {mem_est.total_mb:.2f} MB) | "
                    f"Decode TPS: {agg.decode_tps_summary.median:.1f} | "
                    f"TTFT: {agg.ttft_summary.median:.1f} ms"
                )

        return results

    @staticmethod
    def save_results(results: List[AggregatedBenchmarkResult], output_prefix: str):
        """Save results to JSON and summary CSV."""
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        json_path = f"{output_prefix}.json"
        csv_path = f"{output_prefix}.csv"

        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        rows = []
        for r in results:
            rows.append({
                "context_length": r.context_length,
                "precision": r.cache_config.precision,
                "group_size": r.cache_config.group_size,
                "quantized_start": r.cache_config.quantized_start,
                "config_key": r.cache_config.key,
                "cache_memory_mb_median": r.measured_cache_mb_summary.median,
                "cache_memory_mb_mean": r.measured_cache_mb_summary.mean,
                "theoretical_cache_mb": r.theoretical_cache_mb,
                "peak_rss_mb_median": r.peak_rss_mb_summary.median,
                "decode_tps_median": r.decode_tps_summary.median,
                "decode_tps_p95": r.decode_tps_summary.p95,
                "ttft_ms_median": r.ttft_summary.median,
                "total_latency_ms_median": r.total_ms_summary.median,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[BenchmarkRunner] Saved benchmark results to {json_path} and {csv_path}")
