"""CLI runner for baseline benchmarks (FP16, INT8, INT4)."""

import argparse
import os
import json
from turbokv.config.model_config import ModelConfig
from turbokv.config.benchmark_config import BenchmarkConfig
from turbokv.config.policy_config import CacheConfig
from turbokv.inference.model_loader import ModelLoader
from turbokv.benchmarking.benchmark_runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run TurboKV Baseline Benchmarks")
    parser.add_argument("--model-path", type=str, default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--num-repetitions", type=int, default=3)
    parser.add_argument("--output-prefix", type=str, default="results/baselines_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig(model_path=args.model_path)
    bench_cfg = BenchmarkConfig(
        context_lengths=args.context_lengths,
        generation_tokens=args.generation_tokens,
        num_repetitions=args.num_repetitions,
    )

    model, tokenizer, model_cfg = ModelLoader.load_model(model_cfg)
    runner = BenchmarkRunner(model, tokenizer, model_config=model_cfg, benchmark_config=bench_cfg)

    configs = [
        CacheConfig("fp16"),
        CacheConfig("int8", group_size=64, quantized_start=0),
        CacheConfig("int4", group_size=64, quantized_start=0),
    ]

    results = runner.run_benchmark(
        context_lengths=args.context_lengths,
        cache_configs=configs,
        num_repetitions=args.num_repetitions,
        generation_tokens=args.generation_tokens,
    )

    BenchmarkRunner.save_results(results, args.output_prefix)


if __name__ == "__main__":
    main()
