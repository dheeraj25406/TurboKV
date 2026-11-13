"""
Plotting and visualization suite for TurboKV benchmarks and evaluation.
Generates publication-quality charts for memory savings, latency profiles, and quality trade-offs.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_kv_memory(baselines: List[Dict[str, Any]], output_path: Path):
    """Plot KV cache memory vs context length for FP16, INT8, INT4."""
    if not baselines:
        print("No baselines data found for memory plot.")
        return

    contexts = sorted(list({b["context_length"] for b in baselines}))
    
    fp16_mem = []
    int8_mem = []
    int4_mem = []
    
    for ctx in contexts:
        for b in baselines:
            if b["context_length"] == ctx:
                p = b["cache_config"]["precision"]
                mem = b["measured_cache_mb"]["median"]
                if p == "fp16":
                    fp16_mem.append(mem)
                elif p == "int8":
                    int8_mem.append(mem)
                elif p == "int4":
                    int4_mem.append(mem)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    ax.plot(contexts, fp16_mem, marker="o", linewidth=2.5, color="#1f77b4", label="FP16 (Baseline)")
    ax.plot(contexts, int8_mem, marker="s", linewidth=2.5, color="#ff7f0e", label="INT8 (Group 64, -46.9% memory)")
    ax.plot(contexts, int4_mem, marker="^", linewidth=2.5, color="#2ca02c", label="INT4 (Group 64, -71.9% memory)")
    
    ax.set_title("Measured KV-Cache Memory Allocation vs Context Length\n(Qwen2.5-1.5B-Instruct-4bit on Apple M4)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Context Length (Tokens)", fontsize=11, fontweight="medium")
    ax.set_ylabel("KV Cache Memory (MB)", fontsize=11, fontweight="medium")
    ax.set_xticks(contexts)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    
    if len(fp16_mem) == len(contexts) and len(int4_mem) == len(contexts):
        idx = -1
        ax.annotate(f"-71.9% ({int4_mem[idx]:.1f} MB)",
                    xy=(contexts[idx], int4_mem[idx]),
                    xytext=(contexts[idx] - 2200, int4_mem[idx] + 25),
                    arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
                    fontweight="bold", color="#2ca02c", fontsize=9)
        ax.annotate(f"-46.9% ({int8_mem[idx]:.1f} MB)",
                    xy=(contexts[idx], int8_mem[idx]),
                    xytext=(contexts[idx] - 2200, int8_mem[idx] + 45),
                    arrowprops=dict(facecolor="#ff7f0e", shrink=0.08, width=1.5, headwidth=6),
                    fontweight="bold", color="#ff7f0e", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved memory plot to {output_path}")


def plot_decode_throughput(baselines: List[Dict[str, Any]], output_path: Path):
    """Plot decode throughput across context lengths."""
    if not baselines:
        return

    contexts = sorted(list({b["context_length"] for b in baselines}))
    precisions = ["fp16", "int8", "int4"]
    colors = {"fp16": "#1f77b4", "int8": "#ff7f0e", "int4": "#2ca02c"}
    labels = {"fp16": "FP16 Baseline", "int8": "INT8 Quantized", "int4": "INT4 Quantized"}
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    x = np.arange(len(contexts))
    width = 0.25
    
    for i, prec in enumerate(precisions):
        tps = []
        for ctx in contexts:
            val = next((b["decode_tps"]["median"] 
                        for b in baselines if b["context_length"] == ctx and b["cache_config"]["precision"] == prec), 0)
            tps.append(val)
        
        rects = ax.bar(x + (i - 1) * width, tps, width, label=labels[prec], color=colors[prec], alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_title("Autoregressive Decode Throughput by Precision & Context\n(Tokens/sec, higher is better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Context Length (Tokens)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Decode Speed (Tokens/sec)", fontsize=11, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    max_val = max([b["decode_tps"]["median"] for b in baselines])
    ax.set_ylim(0, max_val * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved throughput plot to {output_path}")


def plot_quality_comparison(calibration_data: Dict[str, Any], output_path: Path):
    """Plot quality metrics (Perplexity and Retrieval Accuracy) across configurations."""
    if not calibration_data or "by_context_length" not in calibration_data:
        return
        
    profiles = calibration_data["by_context_length"]
    target_ctx = "2048" if "2048" in profiles else list(profiles.keys())[0]
    ctx_profiles = profiles[target_ctx]
    
    configs = list(ctx_profiles.keys())
    names = []
    ppls = []
    accs = []
    
    for cfg in configs:
        data = ctx_profiles[cfg]
        names.append(cfg.replace("_", "\n"))
        raw_ppl = data.get("perplexity", 100.0)
        ppls.append(min(raw_ppl, 100.0))
        accs.append(data.get("retrieval_accuracy", 0.0) * 100.0)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    x = np.arange(len(configs))
    ax1.bar(x, accs, color="#2b5c8f", width=0.5)
    ax1.set_title(f"Needle Retrieval Accuracy (ctx={target_ctx})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Exact Match Accuracy (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(accs):
        ax1.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
        
    ax2.bar(x, ppls, color="#d95f02", width=0.5)
    ax2.set_title(f"Held-Out Perplexity (ctx={target_ctx}, capped at 100)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Perplexity", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(ppls):
        ax2.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
        
    fig.suptitle(f"TurboKV Calibration: Quality Impact Across Quantization Configurations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved quality plot to {output_path}")


def plot_pareto_frontier(final_comparison_data: List[Dict[str, Any]], output_path: Path):
    """Plot Memory Reduction (%) vs Retrieval Accuracy and Perplexity."""
    if not final_comparison_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    colors = {
        "FP16": "#1f77b4",
        "INT8": "#ff7f0e",
        "INT4": "#d62728",
        "TurboKV Adaptive": "#2ca02c"
    }
    markers = {
        "FP16": "o",
        "INT8": "s",
        "INT4": "x",
        "TurboKV Adaptive": "*"
    }
    
    # Filter by context 4096 or latest
    target_ctx = 4096
    records = [r for r in final_comparison_data if r["context_length"] == target_ctx]
    if not records:
        records = final_comparison_data

    for r in records:
        method = r["method_name"]
        mem_red = r["memory_reduction_vs_fp16_pct"]
        acc = r["retrieval_accuracy"] * 100.0
        c = colors.get(method, "gray")
        m = markers.get(method, "o")
        s = 180 if method == "TurboKV Adaptive" else 100
        
        ax.scatter([mem_red], [acc], color=c, marker=m, s=s, label=method, zorder=5)
        ax.annotate(f"{method}\n({mem_red:.1f}% saved, {acc:.0f}% acc)",
                    xy=(mem_red, acc),
                    xytext=(0, 10 if acc < 50 else -20),
                    textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold", color=c)

    ax.set_title(f"Pareto Trade-off: Memory Reduction vs Retrieval Accuracy (ctx={target_ctx})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("KV Cache Memory Reduction vs FP16 (%)", fontsize=10, fontweight="medium")
    ax.set_ylabel("Held-Out Retrieval Accuracy (%)", fontsize=10, fontweight="medium")
    ax.set_xlim(-5, 85)
    ax.set_ylim(-10, 115)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved Pareto frontier plot to {output_path}")


def generate_all_plots(results_dir: Path):
    baselines_path = results_dir / "baselines_benchmark.json"
    calibration_path = results_dir / "calibration_profile.json"
    final_comp_path = results_dir / "final_comparison_results.json"
    
    baselines_data = load_json(baselines_path)
    calibration_data = load_json(calibration_path)
    final_comp_data = load_json(final_comp_path)
    
    if baselines_data:
        plot_kv_memory(baselines_data, results_dir / "kv_memory_vs_context.png")
        plot_decode_throughput(baselines_data, results_dir / "decode_throughput_vs_config.png")
        
    if calibration_data:
        plot_quality_comparison(calibration_data, results_dir / "quality_vs_quantization.png")
        
    if final_comp_data:
        plot_pareto_frontier(final_comp_data, results_dir / "pareto_frontier.png")


if __name__ == "__main__":
    generate_all_plots(Path("results"))
