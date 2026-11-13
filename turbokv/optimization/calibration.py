"""Empirical calibration runner evaluating configuration grid on calibration set."""

import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from turbokv.config.model_config import ModelConfig
from turbokv.config.benchmark_config import BenchmarkConfig
from turbokv.config.policy_config import CacheConfig
from turbokv.datasets.retrieval_dataset import RetrievalDatasetGenerator, RetrievalItem
from turbokv.inference.generation import GenerationEngine
from turbokv.evaluation.retrieval import evaluate_retrieval, RetrievalReport
from turbokv.evaluation.perplexity import compute_cross_entropy_perplexity
from turbokv.evaluation.distribution_metrics import compute_distribution_metrics, DistributionDivergenceReport
from turbokv.benchmarking.statistics import compute_statistics


@dataclass
class CalibrationProfile:
    """Versioned calibration profile capturing empirical trade-offs across configurations."""

    version: str
    model_name: str
    hardware: str
    timestamp: str
    context_lengths: List[int]
    by_context_length: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary_table: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
            "context_lengths": self.context_lengths,
            "by_context_length": self.by_context_length,
            "summary_table": self.summary_table,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Calibration] Saved profile to {filepath}")


class CalibrationRunner:
    """Executes calibration across precision, group size, and onset threshold search space."""

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
        self.dataset_gen = RetrievalDatasetGenerator(self.tokenizer)

    def run_calibration(
        self,
        context_lengths: Optional[List[int]] = None,
        candidate_configs: Optional[List[CacheConfig]] = None,
        output_path: str = "results/calibration_profile.json",
    ) -> CalibrationProfile:
        ctx_lens = context_lengths or [1024, 2048, 4096]
        configs = candidate_configs or [
            CacheConfig("fp16"),
            CacheConfig("int8", group_size=64, quantized_start=0),
            CacheConfig("int8", group_size=32, quantized_start=0),
            CacheConfig("int8", group_size=64, quantized_start=512),
            CacheConfig("int4", group_size=64, quantized_start=0),
            CacheConfig("int4", group_size=32, quantized_start=0),
            CacheConfig("int4", group_size=64, quantized_start=512),
        ]

        print("\n==================================================")
        print("[Calibration] Starting Empirical Calibration Suite")
        print(f"Context lengths: {ctx_lens}")
        print(f"Configurations ({len(configs)}): {[c.key for c in configs]}")
        print("==================================================")

        # Generate deterministic calibration dataset
        calib_items = self.dataset_gen.generate_suite(
            context_lengths=ctx_lens,
            split="calibration",
            positions=[0.1, 0.5, 0.9],
        )
        print(f"[Calibration] Generated {len(calib_items)} calibration retrieval instances.")

        by_ctx_data: Dict[str, Dict[str, Any]] = {}
        summary_rows: List[Dict[str, Any]] = []

        # Group items by context length
        items_by_ctx: Dict[int, List[RetrievalItem]] = {}
        for it in calib_items:
            items_by_ctx.setdefault(it.context_tokens_target, []).append(it)

        # Baseline FP16 predictions and logprobs for KL divergence reference
        fp16_refs: Dict[str, Any] = {}

        for ctx in ctx_lens:
            ctx_items = items_by_ctx.get(ctx, [])
            if not ctx_items:
                continue

            by_ctx_data[str(ctx)] = {}
            print(f"\n--- Calibrating Context Length: {ctx} ({len(ctx_items)} items) ---")

            for cfg in configs:
                print(f"  Evaluating {cfg.key}...")
                preds = []
                ttfts = []
                tpss = []
                cache_mbs = []
                all_logprobs = []

                for item in ctx_items:
                    res = self.engine.generate(
                        prompt=item.full_prompt,
                        cache_config=cfg,
                        max_tokens=32,
                        capture_logprobs=True,
                    )
                    preds.append(res.generated_text)
                    ttfts.append(res.ttft_ms)
                    tpss.append(res.decode_tokens_per_sec)
                    cache_mbs.append(res.kv_cache_measured_mb)
                    all_logprobs.append(res.step_logprobs)

                # Evaluate retrieval accuracy
                ret_report = evaluate_retrieval(ctx_items, preds)

                # Store FP16 reference for this context length
                if cfg.precision.lower() == "fp16":
                    fp16_refs[ctx] = {
                        "logprobs": all_logprobs,
                        "preds": preds,
                        "accuracy": ret_report.overall_accuracy,
                    }

                # Compute KL divergence vs FP16 if available
                kl_div = 0.0
                top1_agree = 1.0
                if ctx in fp16_refs and cfg.precision.lower() != "fp16":
                    ref_lps = fp16_refs[ctx]["logprobs"]
                    kls = []
                    agrees = []
                    for ref_lp, cand_lp in zip(ref_lps, all_logprobs):
                        div_rep = compute_distribution_metrics(ref_lp, cand_lp)
                        kls.append(div_rep.mean_kl_divergence)
                        agrees.append(div_rep.top1_agreement_rate)
                    kl_div = float(compute_statistics(kls).mean) if kls else 0.0
                    top1_agree = float(compute_statistics(agrees).mean) if agrees else 1.0

                # Compute Perplexity on held-out snippet
                ppl, ppl_loss = compute_cross_entropy_perplexity(
                    self.model, self.tokenizer, cache_config=cfg, max_length=256
                )

                stat_ttft = compute_statistics(ttfts)
                stat_tps = compute_statistics(tpss)
                stat_cache = compute_statistics(cache_mbs)

                entry = {
                    "config_key": cfg.key,
                    "precision": cfg.precision,
                    "group_size": cfg.group_size,
                    "quantized_start": cfg.quantized_start,
                    "measured_cache_mb": stat_cache.median,
                    "ttft_ms": stat_ttft.median,
                    "decode_tokens_per_sec": stat_tps.median,
                    "retrieval_accuracy": ret_report.overall_accuracy,
                    "retrieval_em_accuracy": ret_report.exact_match_accuracy,
                    "accuracy_by_position": ret_report.accuracy_by_position,
                    "mean_kl_divergence": kl_div,
                    "top1_agreement_rate": top1_agree,
                    "perplexity": ppl,
                }
                by_ctx_data[str(ctx)][cfg.key] = entry

                summary_rows.append({
                    "context_length": ctx,
                    "config_key": cfg.key,
                    "measured_cache_mb": stat_cache.median,
                    "decode_tps": stat_tps.median,
                    "retrieval_accuracy": ret_report.overall_accuracy,
                    "mean_kl_divergence": kl_div,
                    "perplexity": ppl,
                })

                print(
                    f"    Result: Cache: {stat_cache.median:.1f} MB | "
                    f"TPS: {stat_tps.median:.1f} | "
                    f"Retrieval Acc: {ret_report.overall_accuracy * 100:.1f}% | "
                    f"KL Div: {kl_div:.4f} | PPL: {ppl:.2f}"
                )

        profile = CalibrationProfile(
            version="1.0.0",
            model_name=self.model_config.model_name,
            hardware="Apple M4 (16 GB Unified Memory)",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            context_lengths=ctx_lens,
            by_context_length=by_ctx_data,
            summary_table=summary_rows,
        )
        profile.save(output_path)
        return profile
