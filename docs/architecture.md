# TurboKV Architecture and Engineering Design

TurboKV is an empirical, research-grade framework engineered for memory-adaptive Key-Value (KV) cache quantization during autoregressive LLM inference on Apple Silicon using Apple MLX.

## 1. System Overview

```
                      +-----------------------------+
                      |      Client Request /       |
                      |      Inference Workload     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Model & Config Loader    |
                      |   (turbokv.inference)       |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+-------------------------------+              +---------------------------------+
|      Inference Engine         |              |    Workload-Aware Adaptive      |
|  - GenerationEngine           |              |    Quantization Policy          |
|  - CacheFactory               | <----------+ |  - turbokv.optimization.policy  |
|  - MemoryProfiler             |              |  - MemoryEstimator (theoretical)|
+---------------+---------------+              +----------------+----------------+
                |                                               ^
                v                                               |
+-------------------------------+              +----------------+----------------+
|    Hardware Runtime (MLX)     |              |     Empirical Calibration       |
|  - Apple Metal Unified Memory | -----------> |     Profile Database            |
|  - Apple M4 GPU Exec          |              |  - Perplexity (Held-Out)        |
+-------------------------------+              |  - Needle-in-a-Haystack Acc     |
                                               |  - Next-Token KL Divergence     |
                                               +---------------------------------+
```

## 2. Core Modules

### 2.1 `turbokv.config`
- **`model_config.py`**: Encapsulates model architecture specifications (number of layers, attention heads, key-value heads, hidden dimensions, head dimensions) and provides the base theoretical analytical calculation for uncompressed FP16 KV-caches.
- **`benchmark_config.py`**: Configures sequence lengths (1024, 2048, 4096, 8192, 16384), generation token budgets, warm-up iterations, and repetitions.
- **`policy_config.py`**: Strongly-typed dataclasses for cache configurations (`precision`: FP16, INT8, INT4; `group_size`: 32, 64; `quantized_start`: 0, 512) and user policy constraints (`max_memory_mb`, `max_ppl_increase`, `min_retrieval_acc`, `max_latency_regression_pct`).

### 2.2 `turbokv.inference`
- **`model_loader.py`**: Robust model discovery and loading targeting `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (with automated fallback to `Qwen2.5-0.5B-Instruct-4bit` under memory pressure). Introspects model architecture dynamically from `model.args`.
- **`cache_factory.py`**: Seamless abstraction wrapping MLX's `mlx_lm.models.cache.KVCache` and `mlx_lm.models.cache.QuantizedKVCache`. Standardizes configuration parameters across precisions.
- **`cache_inspector.py`**: Hardware-level introspection of KV cache tensors. Traverses cache objects across all 28 transformer layers, inspecting underlying buffer layouts, `.nbytes`, and quantization group metadata.
- **`generation.py`**: Low-level generation engine driving `mlx_lm.generate.generate_step`. Accurately isolates:
  - Time To First Token (TTFT) / Prompt prefill latency
  - Autoregressive decode token throughput (tokens/sec)
  - Inter-token latency distributions
  - Exact Metal active memory allocations and process RSS deltas.

### 2.3 `turbokv.optimization`
- **`memory_estimator.py`**: Hardware-accurate analytical memory estimator that models the exact buffer allocation behavior of Apple MLX. Rather than using naive formulas ($2 \times L \times S \times H \times D \times B$), it incorporates:
  1. Chunked step allocation (MLX preallocates in increments of 256 tokens).
  2. `uint32` word packing for 4-bit and 8-bit quantized values.
  3. FP16 scale and bias metadata buffers per quantization group.
- **`calibration.py`**: Multi-dimensional calibration pipeline assessing accuracy, perplexity, and latency across discrete configuration tuples.
- **`adaptive_policy.py`**: Constrained selection engine that evaluates all candidate configurations against hard user constraints, optimizes for minimal memory footprint, and provides transparent, human-readable decision explanations.

### 2.4 `turbokv.evaluation`
- **`retrieval.py`**: Multi-needle long-context retrieval evaluator. Generates deterministic synthetic contexts with needle placement at Beginning (10%), Middle (50%), and End (90%) of the context window.
- **`distribution_metrics.py`**: Evaluates step-by-step next-token divergence between candidate quantized caches and the full FP16 baseline, reporting Kullback-Leibler divergence $D_{KL}(P_{\text{ref}} \parallel P_{\text{cand}})$ and Top-1 token agreement.
- **`perplexity.py`**: Evaluates cross-entropy loss and exponential perplexity on held-out linguistic corpora with causal token masking.

### 2.5 `turbokv.benchmarking`
- **`memory.py`**: Metal unified memory and process RSS profiler utilizing `mlx.core.metal` and `psutil`.
- **`statistics.py`**: Robust statistical aggregation computing sample medians, means, standard deviations, min, max, p95 percentiles, and 95% confidence intervals.
- **`benchmark_runner.py`**: Automated test harness orchestrating multi-repetition benchmark runs across context lengths and precision modes.
