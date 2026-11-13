# TurboKV Methodology: Analytical Modeling and Workload-Adaptive Optimization

## 1. Mathematical Modeling of KV-Cache Memory

In autoregressive transformer inference, the Key-Value (KV) cache stores past token representations for all self-attention layers to avoid recomputing Key and Value states during token generation.

### 1.1 Uncompressed (FP16) KV Cache
For an uncompressed cache in 16-bit floating point precision (2 bytes per scalar):
$$M_{\text{FP16}}(\ell) = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times \ell_{\text{step}} \times 2 \text{ bytes}$$

Where:
- $N_{\text{layers}}$: Number of transformer decoder layers (28 for Qwen2.5-1.5B).
- $N_{\text{kv\_heads}}$: Number of key-value projection heads under Grouped-Query Attention (2 for Qwen2.5-1.5B; notice GQA reduces KV heads relative to the 12 query heads).
- $d_{\text{head}}$: Head dimension (128).
- $\ell_{\text{step}}$: Sequence length rounded up to the cache buffer allocation granularity.

In Apple MLX, KV buffers are allocated in dynamic step chunks of $\Delta = 256$ tokens:
$$\ell_{\text{step}} = \left\lceil \frac{\ell_{\text{seq}}}{\Delta} \right\rceil \times \Delta$$

At sequence length 8192 with 128 generated tokens ($\ell_{\text{seq}} = 8320$, $\ell_{\text{step}} = 8448$):
$$M_{\text{FP16}} = 2 \times 28 \times 2 \times 128 \times 8448 \times 2 = 242,221,056 \text{ bytes} \approx 231.00 \text{ MB}$$
This matches the exact measured tensor byte count returned by `mlx.core.array.nbytes`.

### 1.2 Quantized KV Cache Layout and Metadata
When quantized to $b$ bits ($b \in \{4, 8\}$) with quantization group size $G$ ($G \in \{32, 64\}$):
1. **Packed Elements**: Quantized integers are bit-packed into 32-bit unsigned integers (`uint32`, 4 bytes):
   $$B_{\text{packed}} = \frac{b}{8} \text{ bytes/element}$$
2. **Per-Group Scales**: Each group of $G$ values shares an FP16 scaling factor (2 bytes):
   $$\text{Overhead}_{\text{scale}} = \frac{2}{G} \text{ bytes/element}$$
3. **Per-Group Biases**: Each group of $G$ values shares an FP16 bias factor (2 bytes):
   $$\text{Overhead}_{\text{bias}} = \frac{2}{G} \text{ bytes/element}$$

Total effective memory per element:
$$S_{\text{eff}}(b, G) = \frac{b}{8} + \frac{4}{G} \text{ bytes}$$

#### INT8 Group 64:
$$S_{\text{eff}}(8, 64) = 1.0 + \frac{4}{64} = 1.0625 \text{ bytes/element}$$
Theoretical reduction compared to FP16 (2.0 bytes):
$$1 - \frac{1.0625}{2.0} = 46.875\% \text{ reduction}$$
At $\ell_{\text{step}} = 8448$: $122.72 \text{ MB}$ (matching measured MLX memory exactly).

#### INT4 Group 64:
$$S_{\text{eff}}(4, 64) = 0.5 + \frac{4}{64} = 0.5625 \text{ bytes/element}$$
Theoretical reduction compared to FP16:
$$1 - \frac{0.5625}{2.0} = 71.875\% \text{ reduction}$$
At $\ell_{\text{step}} = 8448$: $64.97 \text{ MB}$ (matching measured MLX memory exactly).

---

## 2. Multi-Dimensional Quality Evaluation

Quantization introduces precision degradation that disproportionately impacts critical attention patterns (such as attention sinks and isolated factual tokens). TurboKV implements three orthogonal quality dimensions:

### 2.1 Multi-Needle Long-Context Retrieval
Synthetic documents embedding unique factual assertions (e.g. `The special passkey for system delta is 89471`) at varying normalized depth intervals:
- **Beginning**: $p = 0.10$
- **Middle**: $p = 0.50$
- **End**: $p = 0.90$

We compute:
- **Exact Match (EM)**: Case-sensitive substring equality.
- **Normalized Exact Match (NEM)**: Lowercase, punctuation-stripped token overlap.

### 2.2 Next-Token Distribution Divergence
For each generated step $t$, the full vocabulary logits $z_t \in \mathbb{R}^{V}$ are converted to probabilities $P(x) = \text{softmax}(z_t)$. Taking FP16 as the reference ground-truth distribution $P$ and the candidate quantized cache as $Q$, we compute Kullback-Leibler (KL) divergence:
$$D_{\text{KL}}(P \parallel Q) = \sum_{v \in V} P(v) \log \left( \frac{P(v)}{Q(v) + \epsilon} \right)$$
Additionally, we record the Top-1 Argmax Token Agreement:
$$\mathbb{I}[\arg\max P = \arg\max Q]$$

### 2.3 Held-Out Text Perplexity
Evaluated on a strictly disjoint linguistic corpus:
$$\text{Loss} = -\frac{1}{T} \sum_{t=1}^T \log P_{\text{cand}}(w_t \mid w_{<t})$$
$$\text{Perplexity} = \exp(\text{Loss})$$

---

## 3. Workload-Adaptive Optimization Policy

The TurboKV selection engine formulates KV cache configuration as a **constrained multi-objective optimization problem**:

$$\min_{c \in \mathcal{C}} M_{\text{cache}}(c, \ell)$$
$$\text{subject to:}$$
$$M_{\text{cache}}(c, \ell) \le M_{\text{budget}}$$
$$\Delta \text{PPL}(c) = \text{PPL}(c) - \text{PPL}_{\text{FP16}} \le \tau_{\text{PPL}}$$
$$\text{Acc}_{\text{retrieval}}(c) \ge \tau_{\text{Acc}}$$
$$\Delta \text{TPS}(c) = \frac{\text{TPS}_{\text{FP16}} - \text{TPS}(c)}{\text{TPS}_{\text{FP16}}} \le \tau_{\text{latency}}$$

### Fallback Mechanism:
1. If quantized candidates fail constraints or exceed the budget, but FP16 satisfies the memory budget, the engine triggers an automatic **Fallback to FP16**, ensuring zero quality degradation.
2. If even FP16 exceeds the memory budget, the engine raises an explicit **Infeasible Workload** exception rather than silently violating safety invariants.
