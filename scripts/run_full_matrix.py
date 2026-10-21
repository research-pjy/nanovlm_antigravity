#!/usr/bin/env python3
"""Automated Full-Matrix Experiment Runner & Report Generator for NanoVLM.

Runs all 24 experiments across:
- Datasets    : ShortDesc (`short`), LongDesc (`long`)
- Variants    : `mini`, `base`, `large`
- Patch Modes : `patch_then_conv`, `conv_then_patch_grid`, `conv_stem_strided`, `standard_vit`

Key efficiency & safety features for `rama` (L40S 44 GiB):
1. Resume-safe: Automatically skips training if `*_best.pt` and `loss_history.json` already
   exist in a run folder (so your 3 already-trained `shortdesc_mini_*` runs are preserved!).
2. Batched Single-Load LLM Judge: Loads `Qwen/Qwen3-VL-8B-Instruct` ONLY ONCE at the end
   of the pipeline to grade any ungraded evaluation files, saving ~30 minutes of model reload overhead.
3. Automatic Report & Table Generation: Compiles `summary_results.md`, `summary_results.csv`,
   and `summary_results.json` with parameter counts, loss gaps (Figure 6), ROUGE-1 (Figure 8),
   and Table 3 benchmark scores.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nanovlm.models.nanovlm import NanoVLM
from nanovlm.data.dataset import NanoVLMDataset, PROMPT_EVAL_JUDGE
from nanovlm.data.tokenizer import SimpleChildTokenizer
from scripts.evaluate import compute_rouge1_f1, parse_judge_scores


ALL_DATASETS = ["short", "long"]
ALL_VARIANTS = ["mini", "base", "large"]
ALL_PATCH_MODES = [
    "patch_then_conv",
    "conv_then_patch_grid",
    "conv_stem_strided",
    "standard_vit",
]

# Legacy directory name mapping so existing runs are automatically recognized and reused
LEGACY_DIR_ALIASES = {
    ("short", "mini", "conv_then_patch_grid"): "shortdesc_mini_conv_then_patch",
}


def resolve_run_dir(runs_root: Path, ds_mode: str, variant: str, patch_mode: str) -> Path:
    alias = LEGACY_DIR_ALIASES.get((ds_mode, variant, patch_mode))
    if alias and (runs_root / alias).exists():
        return runs_root / alias
    return runs_root / f"{ds_mode}desc_{variant}_{patch_mode}"


@torch.no_grad()
def generate_completions_for_run(
    run_dir: Path,
    ckpt_path: Path,
    tok_path: Path,
    dataset_json: Path,
    image_root: Path,
    ds_mode: str,
    num_samples: int,
    device: torch.device,
) -> Path:
    """Generate held-out text completions and ROUGE-1 scores using the trained NanoVLM checkpoint."""
    eval_path = run_dir / f"eval_{ds_mode}.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data.get("results", [])) >= num_samples:
            return eval_path

    tokenizer = SimpleChildTokenizer.load(tok_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NanoVLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = NanoVLMDataset(
        dataset_json,
        image_root=image_root,
        split="held_out",
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )

    num_eval = min(num_samples, len(ds))
    prompt_words = 6 if ds_mode == "short" else 18
    max_new_tokens = 30 if ds_mode == "short" else 65

    results: List[Dict[str, Any]] = []
    rouge_scores: List[float] = []

    for i in range(num_eval):
        sample = ds[i]
        raw_entry = ds.samples[i]
        ref_text = raw_entry["description"]
        words = ref_text.split()
        prefix_text = " ".join(words[:prompt_words])

        prompt_ids = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        img = sample["images"].unsqueeze(0).to(device)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out_ids = model.generate(
            img,
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=20,
            eos_token_id=tokenizer.eos_token_id,
        )
        completed_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        r1 = compute_rouge1_f1(completed_text, ref_text)
        rouge_scores.append(r1)

        results.append({
            "image_id": raw_entry["image_id"],
            "prompt": prefix_text,
            "generated": completed_text,
            "reference": ref_text,
            "rouge1_f1": round(r1, 4),
        })

    avg_rouge = sum(rouge_scores) / max(len(rouge_scores), 1)
    payload = {"avg_rouge1": round(avg_rouge, 4), "results": results}
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_path


@torch.no_grad()
def grade_all_pending_evals(
    eval_Queue: List[Tuple[str, str, str, Path]],
    judge_model_id: str,
    batch_size: int = 16,
) -> None:
    """Load the LLM Judge ONCE and grade all runs that don't yet have 'scores'."""
    pending: List[Tuple[str, str, str, Path, Dict[str, Any]]] = []
    for ds_mode, variant, patch_mode, eval_path in eval_Queue:
        if not eval_path.exists():
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        if results and all("scores" in item for item in results) and "avg_metrics" in data:
            continue
        # If scores exist on items but avg_metrics wasn't cached at top-level, compute it without reloading LLM
        if results and all("scores" in item for item in results):
            metrics = ["grammar", "creativity", "consistency", "meaningfulness", "plot"]
            avg_m = {
                m: round(sum(item["scores"][m] for item in results) / len(results), 2)
                for m in metrics
            }
            data["avg_metrics"] = avg_m
            data["avg_total"] = round(sum(avg_m.values()), 2)
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            continue
        pending.append((ds_mode, variant, patch_mode, eval_path, data))

    if not pending:
        print("\n[Judge] All evaluation files already have LLM-as-a-Judge scores!")
        return

    print(f"\n{'=' * 85}")
    print(f"[Judge] Loading '{judge_model_id}' ONCE to grade {len(pending)} experiment runs...")
    print(f"{'=' * 85}")
    from scripts.build_descriptions import load_teacher_model_and_tokenizer
    judge_model, judge_tok = load_teacher_model_and_tokenizer(judge_model_id)

    for ds_mode, variant, patch_mode, eval_path, data in pending:
        results = data["results"]
        print(f"[Judge] Grading {ds_mode}desc | {variant} | {patch_mode} ({len(results)} samples)...")

        prompts = []
        for item in results:
            messages = [
                {"role": "system", "content": "You are a fair teacher grading a 3-4 year old's story completion."},
                {
                    "role": "user",
                    "content": (
                        f"{PROMPT_EVAL_JUDGE}\n\n"
                        f"Text Beginning: {item['prompt']}\n"
                        f"Completed Text: {item['generated']}\n\n"
                        "Provide scores strictly in this format:\n"
                        "Grammar: X/10\nCreativity: X/10\nConsistency: X/10\nMeaningfulness: X/10\nPlot: X/10"
                    ),
                },
            ]
            try:
                rendered = judge_tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                rendered = judge_tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(rendered)

        # Process in batches for high speed on L40S
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = judge_tok(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(judge_model.device)
            out = judge_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=judge_tok.pad_token_id,
                eos_token_id=judge_tok.eos_token_id,
            )
            in_len = inputs["input_ids"].shape[1]
            replies = judge_tok.batch_decode(out[:, in_len:], skip_special_tokens=True)

            for offset, reply in enumerate(replies):
                item = results[start + offset]
                scores = parse_judge_scores(reply)
                item["scores"] = scores
                item["total_score"] = round(sum(scores.values()), 2)

        metrics = ["grammar", "creativity", "consistency", "meaningfulness", "plot"]
        avg_metrics = {
            m: round(sum(item["scores"][m] for item in results) / len(results), 2)
            for m in metrics
        }
        avg_total = round(sum(avg_metrics.values()), 2)
        data["avg_metrics"] = avg_metrics
        data["avg_total"] = avg_total

        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(
            f"  -> {ds_mode}desc | {variant} | {patch_mode}: "
            f"Grammar={avg_metrics['grammar']} Creativity={avg_metrics['creativity']} "
            f"Consistency={avg_metrics['consistency']} Meaningfulness={avg_metrics['meaningfulness']} "
            f"Plot={avg_metrics['plot']} || Total={avg_total}/50"
        )


def build_summary_reports(
    runs_root: Path,
    datasets: List[str],
    variants: List[str],
    patch_modes: List[str],
) -> None:
    """Compile Markdown, CSV, and JSON summary tables across all completed runs."""
    rows: List[Dict[str, Any]] = []

    for ds_mode in datasets:
        for variant in variants:
            for patch_mode in patch_modes:
                run_dir = resolve_run_dir(runs_root, ds_mode, variant, patch_mode)
                loss_file = run_dir / "loss_history.json"
                eval_file = run_dir / f"eval_{ds_mode}.json"
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
                row = {
                    "dataset": "ShortDesc" if ds_mode == "short" else "LongDesc",
                    "variant": variant,
                    "patch_mode": patch_mode,
                    "total_params_M": round(stats.get("Total", {}).get("millions", 0.0), 3),
                    "patch_params_M": round(
                        stats.get("Patch Embedding (inside Visual Encoder)", {}).get("millions", 0.0), 3
                    ),
                    "vis_encoder_M": round(stats.get("Visual Encoder (Total)", {}).get("millions", 0.0), 3),
                    "best_val_loss": round(best_rec.get("val_loss", 0.0), 4),
                    "final_train_loss": round(final_rec.get("train_loss", 0.0), 4),
                    "final_val_loss": round(final_rec.get("val_loss", 0.0), 4),
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
                }
                rows.append(row)

    if not rows:
        print("[Report] No completed runs found to summarize.")
        return

    # 1. Save JSON
    json_out = runs_root / "summary_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # 2. Save CSV
    csv_out = runs_root / "summary_results.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 3. Save Markdown Table
    md_out = runs_root / "summary_results.md"
    lines = [
        "# NanoVLM Full-Matrix Experimental Results",
        "",
        "| Dataset | Variant | Patch Mode | Params (M) | PatchEmbed (M) | Best Val Loss | Loss Gap | ROUGE-1 | Grammar | Creativity | Consistency | Meaningfulness | Plot | **Total (/50)** |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | `{r['variant']}` | `{r['patch_mode']}` | "
            f"{r['total_params_M']:.2f}M | {r['patch_params_M']:.3f}M | "
            f"{r['best_val_loss']:.4f} | {r['loss_gap']:.4f} | {r['rouge1']:.4f} | "
            f"{r['grammar']:.2f} | {r['creativity']:.2f} | {r['consistency']:.2f} | "
            f"{r['meaningfulness']:.2f} | {r['plot']:.2f} | **{r['avg_total']:.2f}** |"
        )

    md_text = "\n".join(lines) + "\n"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_text)

    print("\n" + "=" * 110)
    print(md_text)
    print("=" * 110)
    print(f"[Report] Saved Markdown, CSV, and JSON reports to: {runs_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full NanoVLM experiment matrix")
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
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Which datasets to run: short, long",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
        help="Which model sizes to run: mini, base, large",
    )
    parser.add_argument(
        "--patch-modes",
        nargs="+",
        default=ALL_PATCH_MODES,
        choices=ALL_PATCH_MODES,
        help="Which patch modes to run",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=25)
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-a-Judge grading")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    runs_root = Path(args.runs_dir).expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_script = Path(__file__).resolve().parent / "train.py"

    total_runs = len(args.datasets) * len(args.variants) * len(args.patch_modes)
    eval_queue = []
    run_idx = 0

    for ds_mode in args.datasets:
        ds_json = dataset_dir / f"{ds_mode}desc.json"
        if not ds_json.exists():
            print(f"[ERROR] Dataset file not found: {ds_json}")
            sys.exit(1)

        for variant in args.variants:
            for patch_mode in args.patch_modes:
                run_idx += 1
                run_dir = resolve_run_dir(runs_root, ds_mode, variant, patch_mode)
                best_ckpt = run_dir / f"nanovlm_{variant}_{patch_mode}_best.pt"
                tok_path = run_dir / "tokenizer.json"
                loss_path = run_dir / "loss_history.json"

                print(f"\n[{run_idx:02d}/{total_runs:02d}] {ds_mode.upper()}DESC | {variant.upper()} | {patch_mode}")
                if best_ckpt.exists() and tok_path.exists() and loss_path.exists():
                    print(f"  -> Checkpoint already exists at {run_dir.name}, skipping training.")
                else:
                    cmd = [
                        sys.executable,
                        str(train_script),
                        "--variant", variant,
                        "--patch-mode", patch_mode,
                        "--train-json", str(ds_json),
                        "--image-root", str(image_root),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--save-dir", str(run_dir),
                    ]
                    subprocess.run(cmd, check=True)

                # Generate held-out completions + ROUGE-1
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
                eval_queue.append((ds_mode, variant, patch_mode, eval_path))

    # Grade all pending evaluations by loading Qwen3-VL-8B-Instruct only ONCE
    if not args.skip_judge:
        grade_all_pending_evals(eval_queue, judge_model_id=args.judge_model)

    # Build final Markdown, CSV, and JSON summary tables
    build_summary_reports(runs_root, args.datasets, args.variants, args.patch_modes)


if __name__ == "__main__":
    main()
