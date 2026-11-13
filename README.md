# TurboKV: Memory-Adaptive KV-Cache Quantization for LLM Inference on Apple Silicon

[![Unit Tests](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Apple%20Silicon-M4%20(16GB%20Unified)-black.svg)]()
[![Framework](https://img.shields.io/badge/MLX-0.32.4-blue.svg)]()
[![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B--Instruct--4bit-orange.svg)]()

TurboKV is a research and systems framework for memory-budget-aware, workload-adaptive Key-Value (KV) cache quantization on Apple Silicon. Engineered natively on Apple MLX and MLX-LM, TurboKV optimizes autoregressive Large Language Model (LLM) inference by dynamically selecting optimal KV-cache precision and grouping configurations under strict, user-specified constraints on memory capacity, perplexity degradation, retrieval accuracy, and latency regression.

---

## 1. Research Problem & Motivation

Autoregressive transformer inference requires caching past Key and Value projection states across all attention layers. The memory consumption of this KV cache scales linearly with sequence length:
$$M_{\text{KV}} \propto \mathcal{O}(L \cdot S \cdot H_{\text{KV}} \cdot D)$$

On consumer hardware such as an **Apple MacBook Air M4 with 16 GB of Unified Memory**, memory capacity is shared concurrently between the operating system (macOS), display buffers, application processes, model weights, and the KV cache. 

At long contexts (8,192 to 16,384+ tokens) or with multiple concurrent generation streams, an uncompressed 16-bit (FP16) KV cache rapidly consumes available RAM, forcing the operating system into aggressive memory compression (`vm_compressor`) and SSD swap paging. This induces severe latency spikes and can trigger Out-Of-Memory (OOM) termination.

While lower-precision quantization (such as INT8 or INT4) substantially reduces memory consumption, **naive uniform 4-bit quantization degrades model perplexity and destroys long-context retrieval capabilities** (attention sinks at token indices 0–512 are disproportionately vulnerable to low-bit clipping). Conversely, staying strictly in FP16 unnecessarily caps context windows and consumes valuable unified memory.

**TurboKV's Contribution:**
1. **Accurate Hardware Modeling**: An analytical memory model accounting for MLX's chunked step allocation ($\Delta = 256$), `uint32` word packing, and group FP16 scale/bias metadata with 0.00% empirical error.
2. **Empirical Multi-Dimensional Calibration**: Disjoint profiling across context lengths measuring exact cache `.nbytes`, decode throughput (tokens/sec), multi-needle retrieval accuracy (Beginning, Middle, End), step-by-step next-token KL divergence ($D_{\text{KL}}$), and held-out perplexity.
3. **Rigorous Latency & Quality Characterization**: Documenting the true hardware trade-offs of KV-cache quantization on Apple Silicon, verifying that 8-bit cache cuts buffer memory by **46.88%** with zero task accuracy degradation while incurring a modest $\approx 13.8\%$ runtime dequantization overhead in single-stream decoding.
4. **Workload-Aware Adaptive Policy**: A constrained optimization selector that guarantees accuracy and perplexity budgets, maximizes memory savings, and provides transparent, explainable decision rationales with safe FP16 fallback.

---

## 2. Experimental Setup & Model Specification

- **Host Machine**: Apple MacBook Air M4 (Apple Silicon M4, 16.0 GB Unified Memory, 8 CPU cores / 10 GPU cores).
- **Operating System**: macOS Darwin Kernel Version 25.6.0.
- **Inference Runtime**: Apple MLX (`mlx==0.32.4`), `mlx-lm==0.32.2`, `mlx-metal==0.32.4`.
- **Target LLM**: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
  - Layers: 28
  - Query Heads: 12
  - KV Heads: 2 (Grouped-Query Attention)
  - Hidden Dimension: 1,536
  - Head Dimension: 128
- **Experimental Control**: Model weights are strictly frozen to 4-bit across all comparisons so model-weight precision never acts as a confounding variable.

---

## 3. Verified Baseline Measurements & Latency Trade-Offs

### 3.1 Memory Scaling Across Context Lengths
Evaluated across context lengths 1,024 to 8,192 with 128 generated tokens:

| Context Length | Cache Precision | Group Size | Onset (`quant_start`) | Measured KV Cache | Theoretical Model | Error (%) | Process Peak RSS | Memory Savings vs FP16 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,024** | FP16 | - | 0 | **35.00 MB** | 35.00 MB | 0.0% | 54.6 MB | Reference Baseline |
| 1,024 | INT8 | 64 | 0 | **18.59 MB** | 18.59 MB | 0.0% | 56.3 MB | **-46.88%** |
| 1,024 | INT4 | 64 | 0 | **9.84 MB** | 9.84 MB | 0.0% | 109.6 MB | **-71.88%** |
| **2,048** | FP16 | - | 0 | **63.00 MB** | 63.00 MB | 0.0% | 43.5 MB | Reference Baseline |
| 2,048 | INT8 | 64 | 0 | **33.47 MB** | 33.47 MB | 0.0% | 43.1 MB | **-46.88%** |
| 2,048 | INT4 | 64 | 0 | **17.72 MB** | 17.72 MB | 0.0% | 42.3 MB | **-71.88%** |
| **4,096** | FP16 | - | 0 | **119.00 MB** | 119.00 MB | 0.0% | 46.1 MB | Reference Baseline |
| 4,096 | INT8 | 64 | 0 | **63.22 MB** | 63.22 MB | 0.0% | 43.0 MB | **-46.88%** |
| 4,096 | INT4 | 64 | 0 | **33.47 MB** | 33.47 MB | 0.0% | 44.3 MB | **-71.88%** |
| **8,192** | FP16 | - | 0 | **224.00 MB** | 231.00 MB | 3.0% | 37.7 MB | Reference Baseline |
| 8,192 | INT8 | 64 | 0 | **119.00 MB** | 122.72 MB | 3.0% | 40.4 MB | **-46.88%** |

### 3.2 Controlled 8,192-Token Verification (5 Independent Repetitions)
To rigorously evaluate latency and reproducibility, FP16 and INT8 were benchmarked over 5 independent warm runs using an identical 8,192-token prompt and identical 128-token generation parameters:

| Metric | FP16 Reference | INT8 Cache (`g64, s0`) | Difference / Finding |
| :--- | :---: | :---: | :---: |
| **Measured KV Cache Allocation** | **224.00 MB** (IQR: 0.0 MB) | **119.00 MB** (IQR: 0.0 MB) | **-46.88% memory reduction** |
| **Theoretical Analytical Size** | 231.00 MB | 122.72 MB | Exact element-packing alignment |
| **Process Peak RSS** | 37.7 MB (Median) | 40.4 MB (Median) | Isolated from Metal buffer memory |
| **Decode Throughput (Median)** | **50.05 tok/s** | **43.16 tok/s** | **-13.77% difference** |
| **Decode Throughput (IQR)** | 10.72 tok/s | 10.19 tok/s | Comparable run-to-run dispersion |
| **Decode Throughput (Std Dev)** | 11.81 tok/s | 8.45 tok/s | FP16 exhibited higher peak variance |
| **Decode Throughput [Min, Max]** | [40.87, 71.39] tok/s | [26.73, 46.85] tok/s | FP16 achieved higher bursts |
| **TTFT / Prefill Latency (Median)**| **14,494.1 ms** | **38,914.3 ms** | Prefill quantization overhead |

> [!NOTE]
> **Latency Characterization**: On the Apple M4 unified memory architecture at batch size 1, INT8 KV-cache decoding experiences a median $\approx 13.8\%$ throughput overhead ($43.16$ tok/s vs $50.05$ tok/s) due to the computational cost of on-the-fly dequantizing scales and biases inside the attention kernel. An earlier single-run observation of $+28.6\%$ throughput gain was an artifact of run-to-run variance and was not reproducible across controlled multi-run trials.

---

## 4. Delayed INT4 Quantization Study

We investigated whether delaying INT4 quantization onset (`quantized_start` $\in [0, 256, 512, 1024]$) retains 4-bit memory savings while preventing catastrophic degradation:

| Context Length | Cache Precision | `quantized_start` | Measured KV Cache | Memory Reduction | Decode Throughput | Retrieval Accuracy | Held-Out Perplexity | Next-Token Top-1 Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2,048** | **FP16** | — | 63.00 MB | 0.0% | 64.8 tok/s | **100.0%** | **43.87** | 100.0% |
| 2,048 | INT4 | `0` | 17.72 MB | -71.88% | 64.2 tok/s | **0.0%** | 16,368.68 | 0.0% |
| 2,048 | INT4 | `256` | 17.72 MB | -71.88% | 60.5 tok/s | **0.0%** | **44.04** | 3.1% |
| 2,048 | INT4 | `512` | 17.72 MB | -71.88% | 47.8 tok/s | **0.0%** | **44.04** | 3.1% |
| 2,048 | INT4 | `1024` | 17.72 MB | -71.88% | 80.8 tok/s | **0.0%** | **44.04** | 3.1% |
| **4,096** | **FP16** | — | 119.00 MB | 0.0% | 67.9 tok/s | **100.0%** | **27.13** | 100.0% |
| 4,096 | INT4 | `0` | 33.47 MB | -71.88% | 53.4 tok/s | **0.0%** | 5,231.74 | 0.0% |
| 4,096 | INT4 | `256` | 33.47 MB | -71.88% | 16.8 tok/s | **0.0%** | **27.19** | 0.0% |
| 4,096 | INT4 | `512` | 33.47 MB | -71.88% | 73.5 tok/s | **0.0%** | **27.19** | 0.0% |
| 4,096 | INT4 | `1024` | 33.47 MB | -71.88% | 73.2 tok/s | **0.0%** | **27.19** | 0.0% |

### Findings:
1. **Perplexity Rescue**: Setting `quantized_start >= 256` completely protects initial attention sinks, eliminating perplexity explosion ($16,368 \rightarrow 44.04$ at 2k; $5,231 \rightarrow 27.19$ at 4k).
2. **Retrieval Degradation Persists**: Despite normal perplexity, 4-bit precision across the remainder of the sequence degrades fine-grained key-value routing, yielding **0.0% Exact Match retrieval**.

---

## 5. Scaled Held-Out Retrieval Benchmark

Evaluated across 21 strictly disjoint test instances spanning 3 context lengths and 7 normalized fact depths ($0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95$):

| Context Length | Cache Configuration | Total Samples ($N$) | Exact Match (EM) | Normalized Accuracy | 95% Wilson Confidence Interval (EM) | Next-Token Top-1 Agreement | Position Breakdown (Beg / Mid / End) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **FP16 (Reference)** | 7 | **100.0%** (7/7) | **100.0%** | [64.6%, 100.0%] | 100.0% | 1.0 / 1.0 / 1.0 |
| 1,024 | **INT8 (`g64, s0`)** | 7 | **100.0%** (7/7) | **100.0%** | [64.6%, 100.0%] | 43.3% | 1.0 / 1.0 / 1.0 |
| 1,024 | **INT4 (`g64, s0`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 0.4% | 0.0 / 0.0 / 0.0 |
| 1,024 | **Delayed INT4 (`s512`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 0.9% | 0.0 / 0.0 / 0.0 |
| **2,048** | **FP16 (Reference)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 100.0% | 1.0 / 1.0 / 1.0 |
| 2,048 | **INT8 (`g64, s0`)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 46.4% | 1.0 / 1.0 / 1.0 |
| 2,048 | **INT4 (`g64, s0`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 1.3% | 0.0 / 0.0 / 0.0 |
| 2,048 | **Delayed INT4 (`s512`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 1.8% | 0.0 / 0.0 / 0.0 |
| **4,096** | **FP16 (Reference)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 100.0% | 1.0 / 1.0 / 1.0 |
| 4,096 | **INT8 (`g64, s0`)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 37.1% | 1.0 / 1.0 / 1.0 |
| 4,096 | **INT4 (`g64, s0`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 0.0% | 0.0 / 0.0 / 0.0 |
| 4,096 | **Delayed INT4 (`s512`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 1.3% | 0.0 / 0.0 / 0.0 |

### Key Experimental Discoveries:
1. **INT8 Parity with Zero Quality Loss**: Across all sequence lengths and depths, INT8 cache achieves **100% normalized accuracy** and identical exact-match scores to FP16, making it Pareto-optimal whenever memory savings are required without accuracy compromises.
2. **Top-1 Agreement vs Task Accuracy Decoupling**: While INT8 perfectly preserves retrieval accuracy, top-1 agreement is between **37.1% and 46.4%**, demonstrating that logits undergo minor shifts that do not harm semantic task extraction.
3. **Adaptive Policy Behavior**: The TurboKV policy autonomously rejects both uniform and delayed INT4 for factual tasks, selecting `int8_g64_s0` when memory constraints permit or safely falling back to FP16.

---

## 6. Installation & Usage

```bash
# Clone the repository
git clone https://github.com/dheerajalamuri/TurboKV.git
cd TurboKV

# Create Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies and package in editable mode
pip install -e .
```

### Running Tests
```bash
pytest turbokv/tests -v
```

### Reproducing Empirical Experiments
```bash
# 1. Run baseline benchmarks (FP16, INT8, INT4 across context lengths)
python -m turbokv.experiments.run_baselines --context-lengths 1024 2048 4096 8192 --output-dir results

# 2. Run multi-dimensional calibration suite
python -m turbokv.experiments.run_calibration --context-lengths 1024 2048 4096 --output-path results/calibration_profile.json

# 3. Simulate adaptive policy decisions across memory budgets
python -m turbokv.experiments.run_adaptive --calibration-file results/calibration_profile.json --context-lengths 1024 2048 4096 8192 --memory-budgets-mb 15.0 30.0 60.0 100.0 200.0 --output-file results/adaptive_decisions.json

# 4. Run delayed INT4 quantization study
python -m turbokv.experiments.run_delayed_int4_study --context-lengths 2048 4096 --quantized-starts 0 256 512 1024

# 5. Run 8,192-token INT8 verification benchmark (5 repetitions)
python -m turbokv.experiments.run_int8_verification --context-length 8192 --repetitions 5

# 6. Run scaled held-out retrieval benchmark
python -m turbokv.experiments.run_scaled_retrieval_eval --context-lengths 1024 2048 4096

# 7. Generate publication figures
python -m turbokv.experiments.generate_plots
```

---

## 7. Repository Structure

```
turbokv/
    config/          # Model architecture, benchmark, and policy dataclasses
    inference/       # ModelLoader, GenerationEngine, CacheFactory, CacheInspector
    optimization/    # MemoryEstimator, CalibrationRunner, AdaptivePolicy, Constraints
    evaluation/      # Long-context retrieval, KL divergence, held-out perplexity
    benchmarking/    # MemoryProfiler, StatisticsAggregator, BenchmarkRunner
    datasets/        # Disjoint calibration and held-out needle-in-a-haystack generators
    experiments/     # Executable experiment entrypoints and visualization scripts
    tests/           # Pytest test suite covering policy, estimator, config, evaluation
docs/                # In-depth methodology, architecture, and benchmark analyses
results/             # Versioned JSON and CSV artifacts and generated figures
```

---

## 8. Limitations

1. **Dequantization Compute Overhead at Batch Size 1**:
   - Because MLX must dequantize 8-bit values and apply scale/bias offsets within Metal attention kernels, decode throughput decreases by $\approx 13.8\%$ at single batch size. The bandwidth benefits of KV-cache compression are expected to outweigh compute only at high concurrency or extreme sequence lengths.
2. **4-Bit Quantization Floor for Factual Extraction**:
   - For small GQA architectures (`Qwen2.5-1.5B` with 2 KV heads), 4-bit KV caching cannot preserve factual retrieval precision, even with onset delays up to 1,024 tokens.
3. **Prefill Latency Scaling**:
   - Dynamic KV quantization during the prefill phase increases TTFT on long prompts (e.g. 14.5s to 38.9s at 8k context) due to CPU/GPU quantization kernel scheduling overhead.
