#!/usr/bin/env python3
"""Phase 2 Novel Ideas Runner for NanoVLM.

Tests our 3 custom architectural innovations against the Phase 1 baselines:
1. `overlapping_conv_stem` (with standard `linear_gelu` connector)
2. `convnext_stem`         (with standard `linear_gelu` connector)
3. `convnext_stem` + `swiglu_token_mixer` (Combined Vision Stem + Gated Spatial Token Mixer Connector)

Automatically merges the new results with your existing 24-run Phase 1 results
and generates `summary_phase2_comparison.md` and `summary_phase2_comparison.csv`.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scripts.run_full_matrix import (
    generate_completions_for_run,
    grade_all_pending_evals,
    resolve_run_dir,
    ALL_PATCH_MODES as PHASE1_MODES,
)

# The 3 novel Phase 2 configurations: (label, patch_mode, connector_mode)
PHASE2_CONFIGS: List[Tuple[str, str, str]] = [
    ("overlapping_conv_stem", "overlapping_conv_stem", "linear_gelu"),
    ("convnext_stem", "convnext_stem", "linear_gelu"),
    ("convnext_swiglu", "convnext_stem", "swiglu_token_mixer"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 novel NanoVLM experiments")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.expanduser("~/datasets/nanovlm_clean"),
        help="Directory containing shortdesc.json and longdesc.json",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=os.path.expanduser("~/datasets"),
        help="Root directory containing images/train2017 and images/val2017",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=os.path.expanduser("~/nanovlm-runs"),
        help="Directory where all model checkpoints and logs are stored",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["short", "long"],
        choices=["short", "long"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["mini", "base", "large"],
        choices=["mini", "base", "large"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=25)
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    runs_root = Path(args.runs_dir).expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_script = Path(__file__).resolve().parent / "train.py"

    total_runs = len(args.datasets) * len(args.variants) * len(PHASE2_CONFIGS)
    eval_queue: List[Tuple[str, str, str, Path]] = []
    run_idx = 0

    for ds_mode in args.datasets:
        ds_json = dataset_dir / f"{ds_mode}desc.json"
        for variant in args.variants:
            for exp_label, patch_mode, connector_mode in PHASE2_CONFIGS:
                run_idx += 1
                run_dir = runs_root / f"{ds_mode}desc_{variant}_{exp_label}"
                best_ckpt = run_dir / f"nanovlm_{variant}_{patch_mode}_best.pt"
                tok_path = run_dir / "tokenizer.json"
                loss_path = run_dir / "loss_history.json"

                print(
                    f"\n[Phase2 {run_idx:02d}/{total_runs:02d}] "
                    f"{ds_mode.upper()}DESC | {variant.upper()} | {exp_label} "
                    f"(patch={patch_mode}, conn={connector_mode})"
                )
                if best_ckpt.exists() and tok_path.exists() and loss_path.exists():
                    print(f"  -> Checkpoint already exists at {run_dir.name}, skipping training.")
                else:
                    cmd = [
                        sys.executable,
                        str(train_script),
                        "--variant", variant,
                        "--patch-mode", patch_mode,
                        "--connector-mode", connector_mode,
                        "--train-json", str(ds_json),
                        "--image-root", str(image_root),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--save-dir", str(run_dir),
                    ]
                    subprocess.run(cmd, check=True)

                eval_path = generate_completions_for_run(
                    run_dir=run_dir,
                    ckpt_path=best_ckpt,
                    tok_path=tok_path,
                    dataset_json=ds_json,
                    image_root=image_root,
                    ds_mode=ds_mode,
                    num_samples=args.eval_samples,
                    device=device,
                )
                eval_queue.append((ds_mode, variant, exp_label, eval_path))

    # Grade all newly generated Phase 2 evaluations in one single LLM load
    grade_all_pending_evals(eval_queue, judge_model_id=args.judge_model)

    # Compile unified comparison table (Phase 1 baselines + Phase 2 novel ideas)
    all_rows: List[Dict[str, Any]] = []
    for ds_mode in args.datasets:
        for variant in args.variants:
            # Collect Phase 1 modes + Phase 2 modes
            mode_entries = [(m, resolve_run_dir(runs_root, ds_mode, variant, m)) for m in PHASE1_MODES]
            for exp_label, _, _ in PHASE2_CONFIGS:
                mode_entries.append((f"*{exp_label}*", runs_root / f"{ds_mode}desc_{variant}_{exp_label}"))

            for display_name, r_dir in mode_entries:
                loss_file = r_dir / "loss_history.json"
                eval_file = r_dir / f"eval_{ds_mode}.json"
                if not loss_file.exists() or not eval_file.exists():
                    continue
                with open(loss_file, "r", encoding="utf-8") as f:
                    loss_data = json.load(f)
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)

                stats = loss_data.get("stats", {})
                history = loss_data.get("history", [])
                best_rec = min(history, key=lambda r: r["val_loss"]) if history else {}
                final_rec = history[-1] if history else {}
                avg_m = eval_data.get("avg_metrics", {})

                all_rows.append({
                    "dataset": "ShortDesc" if ds_mode == "short" else "LongDesc",
                    "variant": variant,
                    "architecture": display_name,
                    "total_params_M": round(stats.get("Total", {}).get("millions", 0.0), 3),
                    "patch_params_M": round(
                        stats.get("Patch Embedding (inside Visual Encoder)", {}).get("millions", 0.0), 3
                    ),
                    "best_val_loss": round(best_rec.get("val_loss", 0.0), 4),
                    "loss_gap": round(
                        abs(final_rec.get("val_loss", 0.0) - final_rec.get("train_loss", 0.0)), 4
                    ),
                    "rouge1": eval_data.get("avg_rouge1", 0.0),
                    "grammar": avg_m.get("grammar", 0.0),
                    "creativity": avg_m.get("creativity", 0.0),
                    "consistency": avg_m.get("consistency", 0.0),
                    "meaningfulness": avg_m.get("meaningfulness", 0.0),
                    "plot": avg_m.get("plot", 0.0),
                    "avg_total": eval_data.get("avg_total", 0.0),
                })

    if all_rows:
        md_out = runs_root / "summary_phase2_comparison.md"
        csv_out = runs_root / "summary_phase2_comparison.csv"
        json_out = runs_root / "summary_phase2_comparison.json"

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        with open(csv_out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        lines = [
            "# NanoVLM Phase 1 Baselines vs. Phase 2 Novel Architectures",
            "",
            "| Dataset | Variant | Architecture | Params (M) | PatchEmbed (M) | Best Val Loss | Loss Gap | ROUGE-1 | Grammar | Creativity | Consistency | Meaningfulness | Plot | **Total (/50)** |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in all_rows:
            lines.append(
                f"| {r['dataset']} | `{r['variant']}` | {r['architecture']} | "
                f"{r['total_params_M']:.2f}M | {r['patch_params_M']:.3f}M | "
                f"{r['best_val_loss']:.4f} | {r['loss_gap']:.4f} | {r['rouge1']:.4f} | "
                f"{r['grammar']:.2f} | {r['creativity']:.2f} | {r['consistency']:.2f} | "
                f"{r['meaningfulness']:.2f} | {r['plot']:.2f} | **{r['avg_total']:.2f}** |"
            )
        md_text = "\n".join(lines) + "\n"
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(md_text)

        print("\n" + "=" * 115)
        print(md_text)
        print("=" * 115)
        print(f"Saved unified Phase 1 + Phase 2 reports to: {md_out}")


if __name__ == "__main__":
    main()
