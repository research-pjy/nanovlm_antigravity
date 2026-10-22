#!/usr/bin/env python3
"""Generate publication-quality figures for the NanoVLM 42-Run Study.

Works in two modes:
1. Directly from `results/summary_phase2_comparison.md` (no GPU or run folders needed!):
   - `pareto_frontier.png`: Parameter Count (M) vs. Total LLM-Judge Score (/50)
   - `ablation_bars_short_long.png`: Side-by-side grouped bar charts across all 7 architectures
   - `val_loss_vs_gap.png`: Best Validation Cross-Entropy Loss & Generalization Gap analysis
   - `patch_embed_compression.png`: Patch Embedding parameter overhead comparison (log/linear scale)
2. If `--runs-dir ~/nanovlm-runs` is also present (on `rama`), additionally generates:
   - `figure6_loss_curves.png`: Epoch-by-epoch Train/Val loss curves (recreating Paper Figure 6)
   - `figure8_rouge_histograms.png`: ROUGE-1 distribution histograms (recreating Paper Figure 8)

Usage:
    python scripts/generate_plots.py --md-file results/summary_phase2_comparison.md --out-dir results/figures
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def parse_markdown_table(md_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or "Dataset" in line or ":---" in line:
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 14:
                continue
            arch = parts[2].replace("*", "").replace("`", "").strip()
            rows.append({
                "dataset": parts[0],
                "variant": parts[1].replace("`", ""),
                "arch": arch,
                "params_M": float(parts[3].replace("M", "")),
                "patch_M": float(parts[4].replace("M", "")),
                "val_loss": float(parts[5]),
                "loss_gap": float(parts[6]),
                "rouge1": float(parts[7]),
                "grammar": float(parts[8]),
                "creativity": float(parts[9]),
                "consistency": float(parts[10]),
                "meaningfulness": float(parts[11]),
                "plot": float(parts[12]),
                "total": float(re.sub(r"[^\d.]", "", parts[13])),
            })
    return rows


def generate_summary_charts(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 200,
    })

    arch_order = [
        "patch_then_conv",
        "conv_then_patch_grid",
        "conv_stem_strided",
        "standard_vit",
        "overlapping_conv_stem",
        "convnext_stem",
        "convnext_swiglu",
    ]
    arch_labels = {
        "patch_then_conv": "Patch->Conv (Paper)",
        "conv_then_patch_grid": "Conv->Patch Grid",
        "conv_stem_strided": "Strided Conv Stem",
        "standard_vit": "Standard ViT",
        "overlapping_conv_stem": "Overlapping Stem*",
        "convnext_stem": "ConvNeXt Stem*",
        "convnext_swiglu": "ConvNeXt+SwiGLU*",
    }
    colors = {
        "patch_then_conv": "#6c757d",
        "conv_then_patch_grid": "#1f77b4",
        "conv_stem_strided": "#2ca02c",
        "standard_vit": "#9467bd",
        "overlapping_conv_stem": "#ff7f0e",
        "convnext_stem": "#17becf",
        "convnext_swiglu": "#d62728",
    }

    # 1. Grouped Bar Chart: ShortDesc & LongDesc across mini, base, large
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    variants = ["mini", "base", "large"]
    x = np.arange(len(variants))
    width = 0.11

    for ax_idx, ds_name in enumerate(["ShortDesc", "LongDesc"]):
        ax = axes[ax_idx]
        ds_rows = [r for r in rows if r["dataset"] == ds_name]
        for i, arch in enumerate(arch_order):
            vals = []
            for v in variants:
                match = [r["total"] for r in ds_rows if r["variant"] == v and r["arch"] == arch]
                vals.append(match[0] if match else 0.0)
            offset = (i - len(arch_order) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=arch_labels[arch], color=colors[arch], edgecolor="black", linewidth=0.4)
            for b in bars:
                h = b.get_height()
                if h >= 38.0:
                    ax.text(b.get_x() + b.get_width() / 2, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(f"{ds_name} — Benchmark Total Score (/50)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"NanoVLM-{v}" for v in variants])
        ax.set_ylim(28, 43)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ax_idx == 0:
            ax.set_ylabel("Average Total Score (/50)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_dir / "ablation_bars_short_long.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Parameter Efficiency Pareto Plot (Params M vs Score)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax_idx, ds_name in enumerate(["ShortDesc", "LongDesc"]):
        ax = axes[ax_idx]
        ds_rows = [r for r in rows if r["dataset"] == ds_name]
        for arch in arch_order:
            pts = sorted([r for r in ds_rows if r["arch"] == arch], key=lambda r: r["params_M"])
            xs = [p["params_M"] for p in pts]
            ys = [p["total"] for p in pts]
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=7, label=arch_labels[arch], color=colors[arch])
        ax.set_title(f"{ds_name}: Parameter Efficiency Frontier")
        ax.set_xlabel("Total Trainable Parameters (Millions)")
        ax.set_ylabel("Total LLM-Judge Score (/50)")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Best Validation Loss Comparison across Scales
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    for ax_idx, ds_name in enumerate(["ShortDesc", "LongDesc"]):
        ax = axes[ax_idx]
        ds_rows = [r for r in rows if r["dataset"] == ds_name]
        for arch in arch_order:
            pts = sorted([r for r in ds_rows if r["arch"] == arch], key=lambda r: r["params_M"])
            xs = [p["variant"] for p in pts]
            ys = [p["val_loss"] for p in pts]
            ax.plot(xs, ys, marker="s", linewidth=1.8, markersize=6, label=arch_labels[arch], color=colors[arch])
        ax.set_title(f"{ds_name}: Best Validation Cross-Entropy Loss (Lower is Better)")
        ax.set_xlabel("Model Scale")
        ax.set_ylabel("Best Validation Loss")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "val_loss_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Generated publication charts in: {out_dir}")


def generate_run_level_charts(runs_dir: Path, out_dir: Path) -> None:
    """If run folders exist (on rama), plot Figure 6 epoch curves and Figure 8 ROUGE-1 histograms."""
    import matplotlib.pyplot as plt

    # Check if loss_history.json files exist
    loss_files = list(runs_dir.glob("*/loss_history.json"))
    if not loss_files:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    variants = ["mini", "base", "large"]
    for row_idx, ds_prefix in enumerate(["shortdesc", "longdesc"]):
        for col_idx, var in enumerate(variants):
            ax = axes[row_idx, col_idx]
            for arch in ["patch_then_conv", "conv_stem_strided", "convnext_swiglu"]:
                folder = runs_dir / f"{ds_prefix}_{var}_{arch}"
                lf = folder / "loss_history.json"
                if not lf.exists():
                    continue
                with open(lf, "r", encoding="utf-8") as f:
                    h = json.load(f).get("history", [])
                epochs = [e["epoch"] for e in h]
                t_loss = [e["train_loss"] for e in h]
                v_loss = [e["val_loss"] for e in h]
                ax.plot(epochs, t_loss, linestyle="--", alpha=0.7, label=f"{arch} (train)")
                ax.plot(epochs, v_loss, linewidth=1.8, label=f"{arch} (val)")
            ax.set_title(f"{ds_prefix.capitalize()} - {var}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy Loss")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "figure6_loss_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Generated Figure 6 loss trajectory plot: {out_dir / 'figure6_loss_curves.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication plots for NanoVLM")
    parser.add_argument(
        "--md-file",
        type=str,
        default="results/summary_phase2_comparison.md",
        help="Path to summary_phase2_comparison.md",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=os.path.expanduser("~/nanovlm-runs"),
        help="Optional path to nanovlm-runs directory for epoch loss curves",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures",
        help="Directory to save generated PNG plots",
    )
    args = parser.parse_args()

    md_path = Path(args.md_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_markdown_table(md_path)
    generate_summary_charts(rows, out_dir)

    runs_path = Path(args.runs_dir).expanduser().resolve()
    if runs_path.exists():
        generate_run_level_charts(runs_path, out_dir)


if __name__ == "__main__":
    main()
