"""
Comprehensive investigation into delayed INT4 KV-cache quantization onset.
Evaluates quantized_start in [0, 256, 512, 1024] against the FP16 reference baseline.
Measures: actual cache memory (.nbytes), peak memory, TTFT, decode TPS,
perplexity, retrieval accuracy (EM/NEM), and next-token top-1 agreement.
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
from turbokv.datasets.retrieval_dataset import RetrievalDatasetGenerator
from turbokv.evaluation.retrieval import evaluate_retrieval
from turbokv.evaluation.perplexity import compute_cross_entropy_perplexity
from turbokv.evaluation.distribution_metrics import compute_distribution_metrics
from turbokv.optimization.memory_estimator import MemoryEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Investigate Delayed INT4 Quantization")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--quantized-starts", type=int, nargs="+", default=[0, 256, 512, 1024])
    parser.add_argument("--max-gen-tokens", type=int, default=32)
    parser.add_argument("--num-retrieval-items", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="results/delayed_int4_study.json")
    parser.add_argument("--output-csv", type=str, default="results/delayed_int4_study.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("TurboKV Research: Delayed INT4 Quantization Study")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Quantized starts: {args.quantized_starts}")
    print("==================================================")

    loader = ModelLoader()
    model, tokenizer, model_config = loader.load_model()
    engine = GenerationEngine(model, tokenizer)
    dataset_gen = RetrievalDatasetGenerator(tokenizer)
    estimator = MemoryEstimator(model_config)

    records: List[Dict[str, Any]] = []

    for ctx in args.context_lengths:
        print(f"\n--- Context Length: {ctx} ---")
        
        # 1. Generate deterministic retrieval items for this context length
        retrieval_items = dataset_gen.generate_suite(
            context_lengths=[ctx],
            split="held_out",
            positions=[0.1, 0.3, 0.5, 0.7, 0.9],
        )
        print(f"Generated {len(retrieval_items)} held-out retrieval items.")

        # 2. First, evaluate FP16 Ground Truth Reference
        print("  Evaluating FP16 (Reference)...")
        fp16_config = CacheConfig(precision="fp16")
        
        # Generate on first prompt to get reference distribution for KL / top-1
        fp16_sample_res = engine.generate(
            retrieval_items[0].full_prompt,
            cache_config=fp16_config,
            max_tokens=args.max_gen_tokens,
            capture_logprobs=True,
        )
        fp16_logprobs = fp16_sample_res.step_logprobs

        # Measure FP16 retrieval
        fp16_retrieval_preds = []
        for item in retrieval_items:
            res = engine.generate(
                item.full_prompt,
                cache_config=fp16_config,
                max_tokens=args.max_gen_tokens,
                capture_logprobs=False,
            )
            fp16_retrieval_preds.append(res.generated_text)
        fp16_ret_report = evaluate_retrieval(retrieval_items, fp16_retrieval_preds)

        # Measure FP16 perplexity
        fp16_ppl_text = " ".join([it.full_prompt[:512] for it in retrieval_items[:2]])
        fp16_ppl, _ = compute_cross_entropy_perplexity(model, tokenizer, fp16_ppl_text, fp16_config)

        fp16_mem = fp16_sample_res.kv_cache_measured_mb
        fp16_rec = {
            "context_length": ctx,
            "precision": "fp16",
            "quantized_start": None,
            "config_label": "FP16 (Reference)",
            "measured_cache_mb": fp16_mem,
            "theoretical_cache_mb": estimator.estimate_cache_memory_mb(ctx + args.max_gen_tokens, fp16_config),
            "peak_rss_mb": fp16_sample_res.peak_rss_mb,
            "memory_reduction_pct": 0.0,
            "ttft_ms": fp16_sample_res.ttft_ms,
            "decode_tps": fp16_sample_res.decode_tokens_per_sec,
            "retrieval_accuracy": fp16_ret_report.overall_accuracy,
            "retrieval_em_accuracy": fp16_ret_report.exact_match_accuracy,
            "perplexity": fp16_ppl,
            "mean_kl_divergence": 0.0,
            "top1_agreement": 1.0,
        }
        records.append(fp16_rec)
        print(f"    [FP16] Cache: {fp16_mem:.1f} MB | TPS: {fp16_sample_res.decode_tokens_per_sec:.1f} | Acc: {fp16_ret_report.overall_accuracy*100:.1f}% | PPL: {fp16_ppl:.2f}")

        # 3. Evaluate INT4 with varying quantized_start
        for qs in args.quantized_starts:
            print(f"  Evaluating INT4 (quantized_start={qs})...")
            int4_cfg = CacheConfig(precision="int4", bits=4, group_size=64, quantized_start=qs)

            # Sample run for distribution metrics
            int4_sample_res = engine.generate(
                retrieval_items[0].full_prompt,
                cache_config=int4_cfg,
                max_tokens=args.max_gen_tokens,
                capture_logprobs=True,
            )
            dist_metrics = compute_distribution_metrics(fp16_logprobs, int4_sample_res.step_logprobs)

            # Retrieval evaluation
            int4_retrieval_preds = []
            for item in retrieval_items:
                res = engine.generate(
                    item.full_prompt,
                    cache_config=int4_cfg,
                    max_tokens=args.max_gen_tokens,
                    capture_logprobs=False,
                )
                int4_retrieval_preds.append(res.generated_text)
            int4_ret_report = evaluate_retrieval(retrieval_items, int4_retrieval_preds)

            # Perplexity evaluation
            int4_ppl, _ = compute_cross_entropy_perplexity(model, tokenizer, fp16_ppl_text, int4_cfg)

            int4_mem = int4_sample_res.kv_cache_measured_mb
            mem_red = ((fp16_mem - int4_mem) / fp16_mem) * 100.0 if fp16_mem > 0 else 0.0

            rec = {
                "context_length": ctx,
                "precision": "int4",
                "quantized_start": qs,
                "config_label": f"INT4 (qs={qs})",
                "measured_cache_mb": int4_mem,
                "theoretical_cache_mb": estimator.estimate_cache_memory_mb(ctx + args.max_gen_tokens, int4_cfg),
                "peak_rss_mb": int4_sample_res.peak_rss_mb,
                "memory_reduction_pct": mem_red,
                "ttft_ms": int4_sample_res.ttft_ms,
                "decode_tps": int4_sample_res.decode_tokens_per_sec,
                "retrieval_accuracy": int4_ret_report.overall_accuracy,
                "retrieval_em_accuracy": int4_ret_report.exact_match_accuracy,
                "perplexity": int4_ppl,
                "mean_kl_divergence": dist_metrics.mean_kl_divergence,
                "top1_agreement": dist_metrics.top1_agreement_rate,
            }
            records.append(rec)
            print(f"    [INT4 qs={qs:4d}] Cache: {int4_mem:.1f} MB (-{mem_red:.1f}%) | TPS: {int4_sample_res.decode_tokens_per_sec:.1f} | Acc: {int4_ret_report.overall_accuracy*100:.1f}% | PPL: {int4_ppl:.2f} | Top1: {dist_metrics.top1_agreement_rate*100:.1f}%")

    # Save results
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"\n[Delayed INT4 Study] Saved results to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
