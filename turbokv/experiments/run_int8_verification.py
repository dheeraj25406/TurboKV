"""
Repetition and verification suite for FP16 vs INT8 at context length 8,192.
Conducts 5 independent repetitions to evaluate median, IQR, min, max, std dev,
and 95% confidence intervals for TTFT, decode throughput, and memory metrics.
Verifies whether the throughput acceleration is reproducible under strict controls.
"""

import os
import json
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from turbokv.config.model_config import ModelConfig
from turbokv.config.policy_config import CacheConfig
from turbokv.inference.model_loader import ModelLoader
from turbokv.inference.generation import GenerationEngine
from turbokv.benchmarking.statistics import compute_statistics
from turbokv.optimization.memory_estimator import MemoryEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Verify INT8 vs FP16 at 8192 context")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="results/int8_verification_8k.json")
    parser.add_argument("--output-csv", type=str, default="results/int8_verification_8k.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("TurboKV Verification: INT8 vs FP16 at 8,192 Context")
    print(f"Repetitions: {args.repetitions} independent runs")
    print(f"Generation tokens: {args.generation_tokens}")
    print("==================================================")

    loader = ModelLoader()
    model, tokenizer, model_config = loader.load_model()
    engine = GenerationEngine(model, tokenizer)
    estimator = MemoryEstimator(model_config)

    # Standardized repeatable prompt padded deterministically to 8,192 tokens
    base_text = "The accelerated inference of autoregressive transformer architectures requires optimizing key-value tensor allocations across high-bandwidth memory hierarchies. "
    base_tokens = tokenizer.encode(base_text)
    repeats_needed = (args.context_length // len(base_tokens)) + 1
    long_tokens = (base_tokens * repeats_needed)[: args.context_length]
    prompt_str = tokenizer.decode(long_tokens)
    print(f"Created controlled prompt with exactly {len(long_tokens)} tokens.")

    configs = {
        "FP16": CacheConfig(precision="fp16"),
        "INT8": CacheConfig(precision="int8", bits=8, group_size=64, quantized_start=0),
    }

    results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for name, cfg in configs.items():
        print(f"\nEvaluating {name} ({args.repetitions} runs)...")
        # Warmup run to eliminate Metal shader compilation jitter
        print("  Running warmup iteration...")
        _ = engine.generate(prompt_str, cache_config=cfg, max_tokens=16)

        ttft_runs = []
        decode_tps_runs = []
        total_latency_runs = []
        measured_cache_runs = []
        peak_rss_runs = []
        metal_active_runs = []

        for rep in range(args.repetitions):
            res = engine.generate(prompt_str, cache_config=cfg, max_tokens=args.generation_tokens)
            ttft_runs.append(res.ttft_ms)
            decode_tps_runs.append(res.decode_tokens_per_sec)
            total_latency_runs.append(res.total_latency_ms)
            measured_cache_runs.append(res.kv_cache_measured_mb)
            peak_rss_runs.append(res.peak_rss_mb)
            metal_active_runs.append(res.metal_active_mb)
            print(f"  Run {rep+1}/{args.repetitions}: TTFT={res.ttft_ms:.1f}ms | TPS={res.decode_tokens_per_sec:.2f} | Cache={res.kv_cache_measured_mb:.2f}MB | PeakRSS={res.peak_rss_mb:.1f}MB")

        theo_mem = estimator.estimate_cache_memory_mb(args.context_length + args.generation_tokens, cfg)

        ttft_stats = compute_statistics(ttft_runs).to_dict()
        tps_stats = compute_statistics(decode_tps_runs).to_dict()
        lat_stats = compute_statistics(total_latency_runs).to_dict()
        cache_stats = compute_statistics(measured_cache_runs).to_dict()
        rss_stats = compute_statistics(peak_rss_runs).to_dict()
        metal_stats = compute_statistics(metal_active_runs).to_dict()

        results[name] = {
            "config": cfg.to_dict(),
            "theoretical_cache_mb": theo_mem,
            "measured_cache_mb": cache_stats,
            "peak_rss_mb": rss_stats,
            "metal_active_mb": metal_stats,
            "ttft_ms": ttft_stats,
            "decode_tokens_per_sec": tps_stats,
            "total_latency_ms": lat_stats,
            "raw_runs": {
                "ttft_ms": ttft_runs,
                "decode_tokens_per_sec": decode_tps_runs,
                "total_latency_ms": total_latency_runs,
                "measured_cache_mb": measured_cache_runs,
                "peak_rss_mb": peak_rss_runs,
            },
        }

        summary_rows.append({
            "configuration": name,
            "context_length": args.context_length,
            "generation_tokens": args.generation_tokens,
            "repetitions": args.repetitions,
            "theoretical_cache_mb": theo_mem,
            "measured_cache_mb_median": cache_stats["median"],
            "measured_cache_mb_iqr": cache_stats["iqr"],
            "peak_rss_mb_median": rss_stats["median"],
            "ttft_ms_median": ttft_stats["median"],
            "ttft_ms_iqr": ttft_stats["iqr"],
            "decode_tps_median": tps_stats["median"],
            "decode_tps_iqr": tps_stats["iqr"],
            "decode_tps_std": tps_stats["std_dev"],
            "decode_tps_min": tps_stats["min"],
            "decode_tps_max": tps_stats["max"],
            "total_latency_ms_median": lat_stats["median"],
        })

    # Throughput comparison
    fp16_tps = results["FP16"]["decode_tokens_per_sec"]["median"]
    int8_tps = results["INT8"]["decode_tokens_per_sec"]["median"]
    tps_improvement_pct = ((int8_tps - fp16_tps) / fp16_tps) * 100.0

    fp16_mem = results["FP16"]["measured_cache_mb"]["median"]
    int8_mem = results["INT8"]["measured_cache_mb"]["median"]
    mem_reduction_pct = ((fp16_mem - int8_mem) / fp16_mem) * 100.0

    print("\n==================================================")
    print("VERIFICATION SUMMARY (8,192 Context)")
    print(f"FP16 Median Decode TPS: {fp16_tps:.2f} tok/s")
    print(f"INT8 Median Decode TPS: {int8_tps:.2f} tok/s")
    print(f"Throughput Improvement: {tps_improvement_pct:+.2f}%")
    print(f"Memory Reduction:       {mem_reduction_pct:.2f}% (FP16: {fp16_mem:.1f}MB -> INT8: {int8_mem:.1f}MB)")
    print("==================================================")

    results["comparison"] = {
        "context_length": args.context_length,
        "fp16_median_tps": fp16_tps,
        "int8_median_tps": int8_tps,
        "throughput_improvement_pct": tps_improvement_pct,
        "fp16_median_cache_mb": fp16_mem,
        "int8_median_cache_mb": int8_mem,
        "memory_reduction_pct": mem_reduction_pct,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(summary_rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved artifacts to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
