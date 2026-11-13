"""Final held-out comparison evaluating FP16, INT8, INT4, and TurboKV adaptive policy."""

import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from turbokv.config.model_config import ModelConfig
from turbokv.config.benchmark_config import BenchmarkConfig
from turbokv.config.policy_config import CacheConfig, PolicyConstraints
from turbokv.inference.model_loader import ModelLoader
from turbokv.inference.generation import GenerationEngine, GenerationResult
from turbokv.optimization.adaptive_policy import AdaptiveKVPolicy, PolicyDecision
from turbokv.optimization.memory_estimator import MemoryEstimator
from turbokv.datasets.retrieval_dataset import RetrievalDatasetGenerator, RetrievalItem
from turbokv.evaluation.retrieval import evaluate_retrieval, RetrievalReport
from turbokv.evaluation.perplexity import compute_cross_entropy_perplexity
from turbokv.evaluation.distribution_metrics import compute_distribution_metrics
from turbokv.benchmarking.statistics import compute_statistics


@dataclass
class FinalComparisonRecord:
    """Record of performance and quality for a method at a specific context length."""

    context_length: int
    method_name: str  # "FP16", "INT8", "INT4", "TurboKV Adaptive"
    config_used: str

    # Memory
    kv_cache_measured_mb: float
    theoretical_cache_mb: float
    peak_rss_mb: float
    memory_reduction_vs_fp16_pct: float

    # Latency & Throughput
    ttft_ms: float
    decode_tokens_per_sec: float
    total_latency_ms: float
    latency_change_vs_fp16_pct: float

    # Quality on Held-out Suite
    retrieval_accuracy: float
    accuracy_diff_vs_fp16: float
    perplexity: float
    mean_kl_divergence: float
    top1_agreement_rate: float

    # Policy metadata
    policy_selection_reason: Optional[str] = None
    is_feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_length": self.context_length,
            "method_name": self.method_name,
            "config_used": self.config_used,
            "kv_cache_measured_mb": self.kv_cache_measured_mb,
            "theoretical_cache_mb": self.theoretical_cache_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "memory_reduction_vs_fp16_pct": self.memory_reduction_vs_fp16_pct,
            "ttft_ms": self.ttft_ms,
            "decode_tokens_per_sec": self.decode_tokens_per_sec,
            "total_latency_ms": self.total_latency_ms,
            "latency_change_vs_fp16_pct": self.latency_change_vs_fp16_pct,
            "retrieval_accuracy": self.retrieval_accuracy,
            "accuracy_diff_vs_fp16": self.accuracy_diff_vs_fp16,
            "perplexity": self.perplexity,
            "mean_kl_divergence": self.mean_kl_divergence,
            "top1_agreement_rate": self.top1_agreement_rate,
            "policy_selection_reason": self.policy_selection_reason,
            "is_feasible": self.is_feasible,
        }


class FinalComparisonRunner:
    """Orchestrates head-to-head empirical comparison across held-out test workloads."""

    def __init__(
        self,
        model,
        tokenizer,
        calibration_path: str = "results/calibration_profile.json",
        model_config: Optional[ModelConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config or ModelConfig()
        self.engine = GenerationEngine(self.model, self.tokenizer)
        self.dataset_gen = RetrievalDatasetGenerator(self.tokenizer)
        self.estimator = MemoryEstimator(self.model_config)

        if os.path.exists(calibration_path):
            self.policy = AdaptiveKVPolicy.from_calibration_file(calibration_path)
        else:
            self.policy = AdaptiveKVPolicy()

    def run_comparison(
        self,
        context_lengths: Optional[List[int]] = None,
        memory_budget_mb: float = 80.0,  # e.g. 80 MB cache budget to test constrained adaptation
        max_tokens: int = 32,
        output_dir: str = "results",
    ) -> List[FinalComparisonRecord]:
        ctx_lens = context_lengths or [1024, 2048, 4096]
        os.makedirs(output_dir, exist_ok=True)

        print("\n==================================================")
        print("[Final Comparison] Running Head-to-Head Evaluation")
        print(f"Held-out Context Lengths: {ctx_lens}")
        print(f"Cache Memory Budget: {memory_budget_mb} MB")
        print("==================================================")

        # Generate held-out dataset (strictly distinct facts and questions from calibration)
        heldout_items = self.dataset_gen.generate_suite(
            context_lengths=ctx_lens,
            split="held_out",
            positions=[0.1, 0.5, 0.9],
        )
        print(f"[Final Comparison] Generated {len(heldout_items)} held-out retrieval items.")

        items_by_ctx: Dict[int, List[RetrievalItem]] = {}
        for it in heldout_items:
            items_by_ctx.setdefault(it.context_tokens_target, []).append(it)

        all_records: List[FinalComparisonRecord] = []
        policy_selections: Dict[str, int] = {}

        for ctx in ctx_lens:
            ctx_items = items_by_ctx.get(ctx, [])
            if not ctx_items:
                continue

            print(f"\n==========================================")
            print(f"Context Length: {ctx} ({len(ctx_items)} held-out test items)")
            print(f"==========================================")

            # Determine adaptive policy choice
            constraints = PolicyConstraints(
                max_memory_budget_mb=memory_budget_mb,
                max_accuracy_drop=0.10,
                max_latency_regression_pct=25.0,
            )
            decision = self.policy.select(context_length=ctx, constraints=constraints)
            chosen_cfg = decision.chosen_config or CacheConfig("fp16")
            print(f"[Policy Decision for ctx={ctx}]: {decision.decision_reason}")
            print(f"  -> Selected: {chosen_cfg.key} (feasible={decision.is_feasible})")
            policy_selections[chosen_cfg.key] = policy_selections.get(chosen_cfg.key, 0) + 1

            methods = [
                ("FP16 (Reference)", CacheConfig("fp16")),
                ("INT8 (Baseline)", CacheConfig("int8", group_size=64, quantized_start=0)),
                ("INT4 (Baseline)", CacheConfig("int4", group_size=64, quantized_start=0)),
                ("TurboKV Adaptive", chosen_cfg),
            ]

            fp16_metrics = {}

            for method_name, cfg in methods:
                print(f"\n  Evaluating {method_name} [{cfg.key}]...")
                preds = []
                ttfts = []
                tpss = []
                total_lats = []
                cache_mbs = []
                peak_rsss = []
                all_logprobs = []

                for item in ctx_items:
                    res = self.engine.generate(
                        prompt=item.full_prompt,
                        cache_config=cfg,
                        max_tokens=max_tokens,
                        capture_logprobs=True,
                    )
                    preds.append(res.generated_text)
                    ttfts.append(res.ttft_ms)
                    tpss.append(res.decode_tokens_per_sec)
                    total_lats.append(res.total_latency_ms)
                    cache_mbs.append(res.kv_cache_measured_mb)
                    peak_rsss.append(res.peak_rss_mb)
                    all_logprobs.append(res.step_logprobs)

                # Quality evaluation
                ret_rep = evaluate_retrieval(ctx_items, preds)
                ppl, _ = compute_cross_entropy_perplexity(
                    self.model, self.tokenizer, cache_config=cfg, max_length=256
                )

                med_cache = float(np.median(cache_mbs))
                med_ttft = float(np.median(ttfts))
                med_tps = float(np.median(tpss))
                med_total_lat = float(np.median(total_lats))
                med_peak_rss = float(np.median(peak_rsss))

                theo_mem = self.estimator.estimate(ctx + max_tokens, cfg).total_mb

                if method_name == "FP16 (Reference)":
                    fp16_metrics[ctx] = {
                        "cache_mb": med_cache,
                        "ttft_ms": med_ttft,
                        "tps": med_tps,
                        "total_lat": med_total_lat,
                        "acc": ret_rep.overall_accuracy,
                        "logprobs": all_logprobs,
                    }
                    mem_red = 0.0
                    lat_change = 0.0
                    acc_diff = 0.0
                    kl_div = 0.0
                    top1_agree = 1.0
                else:
                    fp16_base = fp16_metrics.get(ctx, {})
                    base_cache = fp16_base.get("cache_mb", med_cache)
                    base_lat = fp16_base.get("total_lat", med_total_lat)
                    base_acc = fp16_base.get("acc", ret_rep.overall_accuracy)

                    mem_red = ((base_cache - med_cache) / base_cache * 100.0) if base_cache > 0 else 0.0
                    lat_change = ((med_total_lat - base_lat) / base_lat * 100.0) if base_lat > 0 else 0.0
                    acc_diff = ret_rep.overall_accuracy - base_acc

                    # KL divergence vs FP16
                    kls = []
                    agrees = []
                    ref_lps = fp16_base.get("logprobs", [])
                    for ref_lp, cand_lp in zip(ref_lps, all_logprobs):
                        div_rep = compute_distribution_metrics(ref_lp, cand_lp)
                        kls.append(div_rep.mean_kl_divergence)
                        agrees.append(div_rep.top1_agreement_rate)
                    kl_div = float(np.mean(kls)) if kls else 0.0
                    top1_agree = float(np.mean(agrees)) if agrees else 1.0

                record = FinalComparisonRecord(
                    context_length=ctx,
                    method_name=method_name,
                    config_used=cfg.key,
                    kv_cache_measured_mb=med_cache,
                    theoretical_cache_mb=theo_mem,
                    peak_rss_mb=med_peak_rss,
                    memory_reduction_vs_fp16_pct=mem_red,
                    ttft_ms=med_ttft,
                    decode_tokens_per_sec=med_tps,
                    total_latency_ms=med_total_lat,
                    latency_change_vs_fp16_pct=lat_change,
                    retrieval_accuracy=ret_rep.overall_accuracy,
                    accuracy_diff_vs_fp16=acc_diff,
                    perplexity=ppl,
                    mean_kl_divergence=kl_div,
                    top1_agreement_rate=top1_agree,
                    policy_selection_reason=(
                        decision.decision_reason if method_name == "TurboKV Adaptive" else None
                    ),
                    is_feasible=decision.is_feasible if method_name == "TurboKV Adaptive" else True,
                )
                all_records.append(record)

                print(
                    f"    [{method_name} - {cfg.key}] "
                    f"Cache: {med_cache:.1f} MB ({mem_red:.1f}% red) | "
                    f"TPS: {med_tps:.1f} | "
                    f"Acc: {ret_rep.overall_accuracy * 100:.1f}% (diff: {acc_diff * 100:+.1f}%) | "
                    f"KL: {kl_div:.4f} | PPL: {ppl:.2f}"
                )

        # Save artifacts
        json_file = os.path.join(output_dir, "final_comparison_results.json")
        csv_file = os.path.join(output_dir, "final_comparison_results.csv")
        policy_file = os.path.join(output_dir, "policy_selections.json")

        with open(json_file, "w") as f:
            json.dump([r.to_dict() for r in all_records], f, indent=2)

        with open(policy_file, "w") as f:
            json.dump(policy_selections, f, indent=2)

        df = pd.DataFrame([r.to_dict() for r in all_records])
        df.to_csv(csv_file, index=False)
        print(f"\n[Final Comparison] Saved full evaluation artifacts to {json_file} and {csv_file}")

        return all_records


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run TurboKV Final Held-Out Comparison")
    parser.add_argument("--model-path", type=str, default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--calibration-path", type=str, default="results/calibration_profile.json")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--memory-budget-mb", type=float, default=80.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig(model_path=args.model_path)
    model, tokenizer, model_cfg = ModelLoader.load_model(model_cfg)
    runner = FinalComparisonRunner(
        model,
        tokenizer,
        calibration_path=args.calibration_path,
        model_config=model_cfg,
    )
    runner.run_comparison(
        context_lengths=args.context_lengths,
        memory_budget_mb=args.memory_budget_mb,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
