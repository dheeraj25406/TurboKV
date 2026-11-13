"""Perplexity evaluation on held-out reference text using model logits and cross-entropy loss."""

import math
from typing import List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from turbokv.config.policy_config import CacheConfig
from mlx_lm.models.cache import make_prompt_cache


HELDOUT_CORPUS = """
The field of artificial intelligence systems engineering has undergone rapid evolution with the emergence of autoregressive transformer architectures.
Modern large language models employ multi-head self-attention and grouped-query attention mechanisms to capture long-range semantic dependencies.
During autoregressive generation, keys and values computed for historical tokens must be retained in memory to avoid quadratic recomputation.
This accumulation of key-value states forms the KV cache, which grows linearly with context sequence length and batch size.
On resource-constrained hardware architectures, such as unified memory Apple Silicon systems, KV cache footprint can rapidly exceed physical RAM.
Quantization strategies reduce the precision of stored key and value vectors from standard 16-bit floating point to 8-bit or 4-bit integer representations.
However, uniform low-bit quantization often incurs non-trivial degradation in retrieval accuracy and shifts the predicted token probability distributions.
Adaptive memory-budget-aware policies systematically explore the trade-off space between memory reduction, inference throughput, and quality preservation.
By dynamically calibrating quantization precision and onset thresholds, inference engines can maintain robust retrieval while operating strictly within device memory envelopes.
"""


def compute_cross_entropy_perplexity(
    model: nn.Module,
    tokenizer,
    text: str = HELDOUT_CORPUS,
    cache_config: Optional[CacheConfig] = None,
    max_length: int = 512,
) -> Tuple[float, float]:
    """
    Computes perplexity on a held-out text segment under the specified cache configuration.
    Returns (perplexity, mean_cross_entropy_loss).
    """
    token_ids = tokenizer.encode(text)
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    tokens = mx.array(token_ids)[None]  # Shape [1, seq_len]

    # Evaluate loss
    # For full context forward pass without manual cache:
    # We can either run model forward directly or evaluate step-by-step with cache.
    # To test the impact of quantized cache, we evaluate sequentially or in chunks.
    if cache_config is None or cache_config.precision.lower() == "fp16":
        logits = model(tokens)
        target = tokens[:, 1:]
        pred_logits = logits[:, :-1, :]
        loss = nn.losses.cross_entropy(pred_logits, target)
        mx.eval(loss)
        mean_loss = float(mx.mean(loss).item())
        ppl = math.exp(min(20.0, mean_loss))
        return ppl, mean_loss

    # For quantized cache evaluation, we feed tokens with prompt cache
    prompt_cache = make_prompt_cache(model)
    # Configure quantization if specified
    if cache_config.bits is not None:
        from mlx_lm.generate import maybe_quantize_kv_cache
        # Process in chunks of 64 tokens to allow maybe_quantize_kv_cache to take effect
        chunk_size = 64
        total_tokens = tokens.shape[1]
        all_logits = []

        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk = tokens[:, start_idx:end_idx]
            out = model(chunk, cache=prompt_cache)
            maybe_quantize_kv_cache(
                prompt_cache,
                quantized_kv_start=cache_config.quantized_start,
                kv_group_size=cache_config.group_size,
                kv_bits=cache_config.bits,
            )
            mx.eval([c.state for c in prompt_cache])
            all_logits.append(out)

        full_logits = mx.concatenate(all_logits, axis=1)
        target = tokens[:, 1:]
        pred_logits = full_logits[:, :-1, :]
        loss = nn.losses.cross_entropy(pred_logits, target)
        mx.eval(loss)
        mean_loss = float(mx.mean(loss).item())
        ppl = math.exp(min(20.0, mean_loss))
        return ppl, mean_loss

    return float("nan"), float("nan")
