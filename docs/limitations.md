# TurboKV Engineering Limitations and Hardware Invariants

## 1. Hardware Boundary: Apple Silicon M4 Unified Memory
- **Physical Capacity**: The test machine is configured with 16.0 GB of unified LPDDR5 memory shared dynamically between CPU cores, GPU execution units, and the Apple Neural Engine (ANE).
- **macOS System Memory Pressure**: macOS dynamically reserves 4 to 6 GB for the Darwin kernel, WindowServer, system daemons, and application working sets. When active memory allocation exceeds ~12 GB, macOS invokes the compressed memory compressor (`vm_compressor`) and secondary SSD swap paging (`/var/vm/swapfile*`).
- **Unified Memory Bandwidth vs Cache Size**:
  While smaller quantized KV caches drastically reduce DRAM memory traffic during decoding, at batch size 1 on Apple Silicon M4, the arithmetic cost of dequantizing scales and biases inside the attention kernel causes a median decode throughput drop of approximately 13.8% (43.16 tok/s for INT8 vs 50.05 tok/s for FP16 at 8,192 context). Prefill latency also experiences overhead (TTFT increases from 14.5s to 38.9s) due to quantization kernel scheduling.

## 2. MLX and Kernel-Level Quantization Constraints
- **Granular Buffer Chunking**:
  MLX's `KVCache` and `QuantizedKVCache` allocate storage in discrete chunks of $\Delta = 256$ tokens. Short prompts (e.g. 10 tokens) allocate an initial buffer sized for 256 tokens. The memory savings from INT8 and INT4 scale linearly once sequence lengths surpass this chunk threshold.
- **De-quantization Overhead at Short Contexts**:
  For contexts under 1024 tokens, dequantizing INT4/INT8 representations on-the-fly inside the attention kernel introduces computational overhead that can slightly diminish decode throughput relative to native FP16 memory loads.
- **Uniform vs Non-Uniform INT4 Instability**:
  Naive uniform INT4 quantization (`group_size=64, quantized_start=0`) degrades Qwen2.5 attention heads drastically (yielding 0.0% retrieval accuracy and severe perplexity explosion to 6770+). However, reserving the initial 512 tokens in FP16 (`quantized_start=512`) stabilizes attention sink dynamics, recovering held-out perplexity back to baseline 28.83.

## 3. Scope Boundaries
- **No Weight Quantization Variations**: In accordance with the rigorous experimental control guidelines, model weights are strictly frozen to 4-bit (`mlx-community/Qwen2.5-1.5B-Instruct-4bit`) across all baseline and policy runs to isolate KV-cache precision as the sole independent variable.
- **Evaluation Isolation**: Calibration datasets and final held-out test datasets are generated with completely disjoint passkeys, random seeds, and context corpora to prevent benchmark data contamination.
