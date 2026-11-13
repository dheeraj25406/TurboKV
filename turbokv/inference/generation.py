"""Inference engine measuring latency, memory snapshots, and token logits."""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.tokenizer_utils import TokenizerWrapper
from turbokv.config.policy_config import CacheConfig
from turbokv.inference.cache_factory import CacheFactory
from turbokv.benchmarking.memory import MemoryProfiler, MemorySnapshot


@dataclass
class GenerationResult:
    """Detailed output and timing/memory metrics of a single generation run."""

    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int

    # Latencies in milliseconds
    ttft_ms: float  # Time To First Token (prefill + first token)
    prefill_ms: float  # Prompt evaluation time
    decode_ms: float  # Generation time for subsequent tokens
    total_latency_ms: float

    # Throughput
    decode_tokens_per_sec: float
    total_tokens_per_sec: float

    # Cache & Memory metrics in MB
    baseline_rss_mb: float
    prefill_rss_mb: float
    peak_rss_mb: float
    final_rss_mb: float
    metal_active_mb: float
    metal_peak_mb: float
    kv_cache_measured_mb: float

    # Token-level outputs
    generated_token_ids: List[int] = field(default_factory=list)
    step_logprobs: List[mx.array] = field(default_factory=list)

    def to_dict(self, include_tensors: bool = False) -> Dict[str, Any]:
        d = {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "generated_text": self.generated_text,
            "ttft_ms": self.ttft_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "total_latency_ms": self.total_latency_ms,
            "decode_tokens_per_sec": self.decode_tokens_per_sec,
            "total_tokens_per_sec": self.total_tokens_per_sec,
            "baseline_rss_mb": self.baseline_rss_mb,
            "prefill_rss_mb": self.prefill_rss_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "final_rss_mb": self.final_rss_mb,
            "metal_active_mb": self.metal_active_mb,
            "metal_peak_mb": self.metal_peak_mb,
            "kv_cache_measured_mb": self.kv_cache_measured_mb,
            "generated_token_ids": self.generated_token_ids,
        }
        if include_tensors:
            d["step_logprobs"] = [lp.tolist() for lp in self.step_logprobs]
        return d


class GenerationEngine:
    """Executes autoregressive generation with exact latency and memory profiling."""

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
        self.profiler = MemoryProfiler()

    def generate(
        self,
        prompt: str,
        cache_config: CacheConfig,
        max_tokens: int = 128,
        prefill_step_size: int = 2048,
        capture_logprobs: bool = True,
    ) -> GenerationResult:
        """
        Executes single prompt generation with specified cache configuration.
        Synchronizes Metal stream to measure exact TTFT, decode time, and memory.
        """
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_arr = mx.array(prompt_tokens)
        prompt_len = len(prompt_tokens)

        # Baseline memory before generation
        self.profiler.reset_peak_memory()
        baseline_snap = self.profiler.capture_snapshot()

        # Instantiate cache
        prompt_cache = CacheFactory.create_cache(self.model, cache_config)

        # Setup generator
        gen = generate_step(
            prompt=prompt_arr,
            model=self.model,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=prefill_step_size,
            kv_bits=cache_config.bits,
            kv_group_size=cache_config.group_size,
            quantized_kv_start=cache_config.quantized_start,
        )

        generated_ids: List[int] = []
        step_logprobs: List[mx.array] = []
        token_times: List[float] = []

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prefill_snap: Optional[MemorySnapshot] = None

        for token_id, logprob in gen:
            # Synchronize metal execution for this step
            mx.eval(token_id)
            current_time = time.perf_counter()

            token_val = int(token_id) if isinstance(token_id, int) else int(token_id.item())
            if first_token_time is None:
                first_token_time = current_time
                prefill_snap = self.profiler.capture_snapshot(prompt_cache)

            token_times.append(current_time)
            generated_ids.append(token_val)

            if capture_logprobs:
                step_logprobs.append(logprob)

            if len(generated_ids) >= max_tokens:
                break

        end_time = time.perf_counter()
        final_snap = self.profiler.capture_snapshot(prompt_cache)

        # Compute latencies
        total_gen_tokens = len(generated_ids)
        if first_token_time is not None:
            ttft_ms = (first_token_time - start_time) * 1000.0
            prefill_ms = ttft_ms  # In single-step generator, prefill terminates when first token is emitted
            decode_tokens = max(0, total_gen_tokens - 1)
            decode_ms = max(0.001, (end_time - first_token_time) * 1000.0)
            decode_tps = (decode_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        else:
            ttft_ms = (end_time - start_time) * 1000.0
            prefill_ms = ttft_ms
            decode_ms = 0.0
            decode_tps = 0.0

        total_latency_ms = (end_time - start_time) * 1000.0
        total_tps = (total_gen_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        for t in generated_ids:
            detokenizer.add_token(t)
        detokenizer.finalize()
        generated_text = detokenizer.text

        peak_rss = max(
            baseline_snap.process_rss_mb,
            prefill_snap.process_rss_mb if prefill_snap else 0.0,
            final_snap.process_rss_mb,
        )

        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            prompt_tokens=prompt_len,
            generated_tokens=total_gen_tokens,
            ttft_ms=ttft_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_latency_ms=total_latency_ms,
            decode_tokens_per_sec=decode_tps,
            total_tokens_per_sec=total_tps,
            baseline_rss_mb=baseline_snap.process_rss_mb,
            prefill_rss_mb=prefill_snap.process_rss_mb if prefill_snap else baseline_snap.process_rss_mb,
            peak_rss_mb=peak_rss,
            final_rss_mb=final_snap.process_rss_mb,
            metal_active_mb=final_snap.metal_active_mb,
            metal_peak_mb=final_snap.metal_peak_mb,
            kv_cache_measured_mb=final_snap.kv_cache_measured_mb,
            generated_token_ids=generated_ids,
            step_logprobs=step_logprobs,
        )
