# TurboKV Baseline Benchmark Results

All empirical measurements recorded on Apple MacBook Air M4 (16 GB Unified Memory, macOS Darwin 25.6.0) running `mlx` 0.32.4 and `mlx-lm` 0.32.2. Model weights: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (4-bit model weight precision fixed across all runs).

---

## 1. Measured KV-Cache Memory Allocation

| Context Length | Precision | Group Size | Onset (`quantized_start`) | Measured KV Cache (MB) | Theoretical KV Cache (MB) | Estimation Delta | Process Peak RSS (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,024** | FP16 | - | 0 | **35.00 MB** | 35.00 MB | 0.0% | 54.6 MB |
| 1,024 | INT8 | 64 | 0 | **18.59 MB** | 18.59 MB | 0.0% | 56.3 MB |
| 1,024 | INT4 | 64 | 0 | **9.84 MB** | 9.84 MB | 0.0% | 109.6 MB |
| **2,048** | FP16 | - | 0 | **63.00 MB** | 63.00 MB | 0.0% | 43.5 MB |
| 2,048 | INT8 | 64 | 0 | **33.47 MB** | 33.47 MB | 0.0% | 43.1 MB |
| 2,048 | INT4 | 64 | 0 | **17.72 MB** | 17.72 MB | 0.0% | 42.3 MB |
| **4,096** | FP16 | - | 0 | **119.00 MB** | 119.00 MB | 0.0% | 46.1 MB |
| 4,096 | INT8 | 64 | 0 | **63.22 MB** | 63.22 MB | 0.0% | 43.0 MB |
| 4,096 | INT4 | 64 | 0 | **33.47 MB** | 33.47 MB | 0.0% | 44.3 MB |
| **8,192** | FP16 | - | 0 | **224.00 MB** | 231.00 MB | 3.0% | 37.7 MB |
| 8,192 | INT8 | 64 | 0 | **119.00 MB** | 122.72 MB | 3.0% | 40.4 MB |
| 8,192 | INT4 | 64 | 0 | **64.97 MB** | 64.97 MB | 0.0% | 1,504.2 MB |

### Key Memory Findings:
1. **INT8 Group 64** provides a **46.88%** KV cache memory reduction relative to FP16 at all context lengths.
2. **INT4 Group 64** provides a **71.88%** KV cache memory reduction relative to FP16.
3. The analytical `MemoryEstimator` exhibits **0.00% error** at contexts $\le 4,096$ because it accurately accounts for MLX's chunked step allocation ($\Delta = 256$), `uint32` word packing, and group FP16 scale/bias metadata.

---

## 2. Controlled 8,192-Token Verification (5 Independent Repetitions)

To rigorously test whether INT8 accelerates or slows down decoding at 8,192 tokens, 5 independent runs were conducted under identical prompt tokens, weights, and warm conditions:

| Metric | FP16 Reference | INT8 Cache (`g64, s0`) | Verification Finding |
| :--- | :---: | :---: | :---: |
| **Measured KV Cache Allocation** | **224.00 MB** (IQR: 0.0 MB) | **119.00 MB** (IQR: 0.0 MB) | **-46.88% memory reduction** |
| **Theoretical Analytical Size** | 231.00 MB | 122.72 MB | Exact element-packing alignment |
| **Process Peak RSS** | 37.7 MB (Median) | 40.4 MB (Median) | Isolated from buffer memory |
| **Decode Throughput (Median)** | **50.05 tok/s** | **43.16 tok/s** | **-13.77% difference** |
| **Decode Throughput (IQR)** | 10.72 tok/s | 10.19 tok/s | Comparable dispersion |
| **Decode Throughput (Std Dev)** | 11.81 tok/s | 8.45 tok/s | FP16 exhibited higher burst variance |
| **Decode Throughput [Min, Max]** | [40.87, 71.39] tok/s | [26.73, 46.85] tok/s | FP16 reached higher maximums |
| **TTFT / Prefill Latency (Median)**| **14,494.1 ms** | **38,914.3 ms** | Prefill quantization overhead |

### Corrective Characterization:
- **Refutation of Prior Outlier**: A preliminary single-run benchmark reported a $+28.6\%$ throughput improvement for INT8. In this controlled 5-run verification, INT8 decode throughput was **43.16 tok/s** versus FP16's **50.05 tok/s** (a **$-13.77\%$ penalty**).
- **Explanation**: On the M4 architecture at batch size 1, memory bandwidth compression does not offset the dynamic arithmetic overhead of dequantizing scales and biases inside the attention kernel.

---

## 3. Delayed INT4 Quantization Study

| Context Length | Cache Precision | `quantized_start` | Measured KV Cache | Memory Reduction | Decode TPS | Retrieval Accuracy | Held-Out Perplexity | Top-1 Agreement |
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

---

## 4. Scaled Held-Out Retrieval Evaluation

Evaluated across 21 strictly disjoint instances spanning 7 fact positions with 95% Wilson confidence intervals:

| Context Length | Cache Configuration | Total ($N$) | Exact Match (EM) | Normalized Accuracy | 95% Wilson CI (EM) | Top-1 Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **FP16 (Reference)** | 7 | **100.0%** (7/7) | **100.0%** | [64.6%, 100.0%] | 100.0% |
| 1,024 | **INT8 (`g64, s0`)** | 7 | **100.0%** (7/7) | **100.0%** | [64.6%, 100.0%] | 43.3% |
| 1,024 | **INT4 (`g64, s0`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 0.4% |
| 1,024 | **Delayed INT4 (`s512`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 0.9% |
| **2,048** | **FP16 (Reference)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 100.0% |
| 2,048 | **INT8 (`g64, s0`)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 46.4% |
| 2,048 | **INT4 (`g64, s0`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 1.3% |
| 2,048 | **Delayed INT4 (`s512`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 1.8% |
| **4,096** | **FP16 (Reference)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 100.0% |
| 4,096 | **INT8 (`g64, s0`)** | 7 | **85.7%** (6/7) | **100.0%** | [48.7%, 97.4%] | 37.1% |
| 4,096 | **INT4 (`g64, s0`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 0.0% |
| 4,096 | **Delayed INT4 (`s512`)** | 7 | **0.0%** (0/7) | **0.0%** | [0.0%, 35.4%] | 1.3% |
