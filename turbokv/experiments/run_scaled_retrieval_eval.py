"""
Scaled, high-statistical-power held-out retrieval evaluation.
Evaluates 15 distinct held-out retrieval items per context length across
7 normalized positions: [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95].
Compares FP16, INT8 (g64, s0), INT4 (g64, s0), and Delayed INT4 (g64, s512).
Calculates exact-match accuracy, normalized accuracy, sample counts,
and 95% Wilson score binomial confidence intervals.
"""

import os
import json
import argparse
import math
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from turbokv.config.policy_config import CacheConfig
from turbokv.inference.model_loader import ModelLoader
from turbokv.inference.generation import GenerationEngine
from turbokv.datasets.retrieval_dataset import RetrievalDatasetGenerator
from turbokv.evaluation.retrieval import evaluate_retrieval
from turbokv.evaluation.distribution_metrics import compute_distribution_metrics


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    spread = (z * math.sqrt((p * (1 - p) / total) + ((z**2) / (4 * (total**2))))) / denom
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return round(low, 4), round(high, 4)


def parse_args():
    parser = argparse.ArgumentParser(description="Scaled Held-Out Retrieval Benchmark")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output-json", type=str, default="results/scaled_retrieval_results.json")
    parser.add_argument("--output-csv", type=str, default="results/scaled_retrieval_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("TurboKV Evaluation: Scaled Disjoint Retrieval Suite")
    print(f"Context lengths: {args.context_lengths}")
    print("Positions: [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]")
    print("==================================================")

    loader = ModelLoader()
    model, tokenizer, model_config = loader.load_model()
    engine = GenerationEngine(model, tokenizer)
    dataset_gen = RetrievalDatasetGenerator(tokenizer)

    positions = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]

    configs_to_test = {
        "FP16": CacheConfig(precision="fp16"),
        "INT8 (g64, s0)": CacheConfig(precision="int8", bits=8, group_size=64, quantized_start=0),
        "INT4 (g64, s0)": CacheConfig(precision="int4", bits=4, group_size=64, quantized_start=0),
        "Delayed INT4 (g64, s512)": CacheConfig(precision="int4", bits=4, group_size=64, quantized_start=512),
    }

    all_results: List[Dict[str, Any]] = []

    for ctx in args.context_lengths:
        print(f"\n==========================================")
        print(f"Context Length: {ctx} Tokens")
        print(f"==========================================")

        # Generate suite for this context length across the 7 positions
        items = dataset_gen.generate_suite(
            context_lengths=[ctx],
            split="held_out",
            positions=positions,
        )
        print(f"Generated {len(items)} strictly disjoint held-out test items.")

        # Reference logprobs from FP16
        ref_logprobs_by_item: List[Any] = []

        for name, cfg in configs_to_test.items():
            print(f"  Evaluating {name}...")
            predictions = []
            item_top1_agreements = []

            for idx, item in enumerate(items):
                res = engine.generate(
                    item.full_prompt,
                    cache_config=cfg,
                    max_tokens=args.max_tokens,
                    capture_logprobs=True,
                )
                predictions.append(res.generated_text)

                if name == "FP16":
                    ref_logprobs_by_item.append(res.step_logprobs)
                    item_top1_agreements.append(1.0)
                else:
                    ref_lps = ref_logprobs_by_item[idx] if idx < len(ref_logprobs_by_item) else []
                    dist_rep = compute_distribution_metrics(ref_lps, res.step_logprobs)
                    item_top1_agreements.append(dist_rep.top1_agreement_rate)

            report = evaluate_retrieval(items, predictions)
            success_count = int(report.exact_match_accuracy * report.total_samples)
            ci_low, ci_high = wilson_score_interval(success_count, report.total_samples)
            mean_top1 = float(np.mean(item_top1_agreements)) if item_top1_agreements else 0.0

            rec = {
                "context_length": ctx,
                "configuration": name,
                "total_samples": report.total_samples,
                "exact_match_successes": success_count,
                "exact_match_accuracy": report.exact_match_accuracy,
                "normalized_accuracy": report.overall_accuracy,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "top1_agreement": round(mean_top1, 4),
                "accuracy_by_position": report.accuracy_by_position,
            }
            all_results.append(rec)
            print(f"    -> Acc: {report.overall_accuracy*100:.1f}% (EM: {report.exact_match_accuracy*100:.1f}% [{ci_low*100:.1f}%, {ci_high*100:.1f}%]) | Top-1: {mean_top1*100:.1f}% | Positions: {report.accuracy_by_position}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n[Scaled Retrieval] Saved results to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
