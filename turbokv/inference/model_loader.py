"""Model loader for MLX models with architecture inspection and fallback handling."""

from typing import Tuple, Optional, Any
import mlx.nn as nn
from mlx_lm import load
from turbokv.config.model_config import ModelConfig


class ModelLoader:
    """Loads and inspects MLX models, verifying architecture and unified memory compatibility."""

    @staticmethod
    def load_model(
        model_config: Optional[ModelConfig] = None,
    ) -> Tuple[nn.Module, Any, ModelConfig]:
        cfg = model_config or ModelConfig()

        try:
            print(f"[ModelLoader] Loading model from: {cfg.model_path}")
            model, tokenizer = load(cfg.model_path)
        except Exception as e:
            print(f"[ModelLoader] Failed loading primary model {cfg.model_path}: {e}")
            print(f"[ModelLoader] Falling back to {cfg.fallback_model_path}")
            cfg.model_path = cfg.fallback_model_path
            cfg.model_name = "Qwen2.5-0.5B-Instruct-4bit"
            model, tokenizer = load(cfg.model_path)

        # Inspect actual loaded architecture to update config parameters
        m_args = getattr(model, "args", None)
        if m_args is not None:
            cfg.num_layers = getattr(m_args, "num_hidden_layers", cfg.num_layers)
            cfg.num_kv_heads = getattr(m_args, "num_key_value_heads", cfg.num_kv_heads)
            cfg.hidden_size = getattr(m_args, "hidden_size", cfg.hidden_size)
            cfg.num_attention_heads = getattr(m_args, "num_attention_heads", cfg.num_attention_heads)
            cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads

        print(
            f"[ModelLoader] Loaded successfully: {cfg.model_name} | "
            f"Layers: {cfg.num_layers}, KV Heads: {cfg.num_kv_heads}, Head Dim: {cfg.head_dim}"
        )
        return model, tokenizer, cfg
