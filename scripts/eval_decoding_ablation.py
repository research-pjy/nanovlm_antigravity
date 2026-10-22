#!/usr/bin/env python3
"""Zero-Retraining Decoding Ablation for LongDesc Models (`repetition_penalty=1.15`, `temp=0.5`).

Tests the hypothesis from Section 4 of the NanoVLM paper:
Because LongDesc requires generating 60-70 word (5-6 sentence) paragraphs with a tiny
4M-24M decoder, standard multinomial sampling (`temp=0.7`, `repetition_penalty=1.0`)
can suffer from phrase repetition loops.

This script loads existing trained LongDesc checkpoints (zero retraining required!),
generates held-out completions with `repetition_penalty=1.15` and `temperature=0.5`,
grades them in a single pass with `Qwen/Qwen3-VL-8B-Instruct`, and compares them
side-by-side against the baseline decoding results.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nanovlm.models.nanovlm import NanoVLM
from nanovlm.data.dataset import NanoVLMDataset, PROMPT_EVAL_JUDGE
from nanovlm.data.tokenizer import SimpleChildTokenizer
from scripts.evaluate import compute_rouge1_f1, parse_judge_scores
from scripts.run_full_matrix import resolve_run_dir


ARCH_LIST = [
    ("patch_then_conv", "patch_then_conv"),
    ("conv_then_patch_grid", "conv_then_patch_grid"),
    ("conv_stem_strided", "conv_stem_strided"),
    ("standard_vit", "standard_vit"),
    ("overlapping_conv_stem", "overlapping_conv_stem"),
    ("convnext_stem", "convnext_stem"),
    ("convnext_swiglu", "convnext_stem"),
]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LongDesc checkpoints with repetition penalty")
    parser.add_argument("--dataset-dir", type=str, default=os.path.expanduser("~/datasets/nanovlm_clean"))
    parser.add_argument("--image-root", type=str, default=os.path.expanduser("~/datasets"))
    parser.add_argument("--runs-dir", type=str, default=os.path.expanduser("~/nanovlm-runs"))
    parser.add_argument("--variants", nargs="+", default=["base", "large"], choices=["mini", "base", "large"])
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--eval-samples", type=int, default=25)
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    runs_root = Path(args.runs_dir).expanduser().resolve()
    long_json = dataset_dir / "longdesc.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pending_grade: List[Tuple[str, str, Path, Dict[str, Any]]] = []

    for variant in args.variants:
        for exp_label, patch_mode in ARCH_LIST:
            run_dir = resolve_run_dir(runs_root, "long", variant, exp_label)
            ckpt_path = run_dir / f"nanovlm_{variant}_{patch_mode}_best.pt"
            tok_path = run_dir / "tokenizer.json"
            if not ckpt_path.exists() or not tok_path.exists():
                continue

            eval_rp_path = run_dir / "eval_long_rp115.json"
            if eval_rp_path.exists():
                with open(eval_rp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "avg_total" in data:
                    pending_grade.append((variant, exp_label, eval_rp_path, data))
                    continue

            print(f"[Decoding] Generating LongDesc (temp={args.temperature}, rep_pen={args.repetition_penalty}) for {variant} | {exp_label}...")
            tokenizer = SimpleChildTokenizer.load(tok_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg = ckpt["config"]
            model = NanoVLM(cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            ds = NanoVLMDataset(
                long_json,
                image_root=image_root,
                split="held_out",
                tokenizer=tokenizer,
                max_seq_len=cfg.max_seq_len,
                pad_token_id=tokenizer.pad_token_id,
            )

            num_eval = min(args.eval_samples, len(ds))
            results: List[Dict[str, Any]] = []
            rouge_scores: List[float] = []

            for i in range(num_eval):
                sample = ds[i]
                raw_entry = ds.samples[i]
                ref_text = raw_entry["description"]
                prefix_text = " ".join(ref_text.split()[:18])

                prompt_ids = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
                img = sample["images"].unsqueeze(0).to(device)
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

                out_ids = model.generate(
                    img,
                    prompt_tensor,
                    max_new_tokens=65,
                    temperature=args.temperature,
                    top_k=20,
                    repetition_penalty=args.repetition_penalty,
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

            payload = {"avg_rouge1": round(sum(rouge_scores) / max(len(rouge_scores), 1), 4), "results": results}
            with open(eval_rp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            pending_grade.append((variant, exp_label, eval_rp_path, payload))
            del model
            torch.cuda.empty_cache()

    to_judge = [item for item in pending_grade if "avg_total" not in item[3]]
    if to_judge:
        from scripts.build_descriptions import load_teacher_model_and_tokenizer
        judge_model, judge_tok = load_teacher_model_and_tokenizer(args.judge_model)
        for variant, exp_label, eval_rp_path, data in to_judge:
            results = data["results"]
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
                    rendered = judge_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(rendered)

            for start in range(0, len(prompts), 16):
                batch_prompts = prompts[start : start + 16]
                inputs = judge_tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(judge_model.device)
                out = judge_model.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=judge_tok.pad_token_id, eos_token_id=judge_tok.eos_token_id,
                )
                in_len = inputs["input_ids"].shape[1]
                replies = judge_tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
                for offset, reply in enumerate(replies):
                    scores = parse_judge_scores(reply)
                    results[start + offset]["scores"] = scores
                    results[start + offset]["total_score"] = round(sum(scores.values()), 2)

            metrics = ["grammar", "creativity", "consistency", "meaningfulness", "plot"]
            avg_m = {m: round(sum(r["scores"][m] for r in results) / len(results), 2) for m in metrics}
            data["avg_metrics"] = avg_m
            data["avg_total"] = round(sum(avg_m.values()), 2)
            with open(eval_rp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    print("\n# LongDesc Repetition-Penalty Decoding Comparison (temp=0.5, rep_penalty=1.15)")
    print("| Variant | Architecture | Baseline Score (temp=0.7) | Penalized Score (rep=1.15) | Delta |")
    print("| :--- | :--- | :---: | :---: | :---: |")
    for variant, exp_label, eval_rp_path, rp_data in pending_grade:
        base_eval_path = eval_rp_path.parent / "eval_long.json"
        base_score = 0.0
        if base_eval_path.exists():
            with open(base_eval_path, "r", encoding="utf-8") as f:
                base_score = json.load(f).get("avg_total", 0.0)
        rp_score = rp_data.get("avg_total", 0.0)
        delta = rp_score - base_score
        sign = "+" if delta >= 0 else ""
        print(f"| `{variant}` | `{exp_label}` | {base_score:.2f} | **{rp_score:.2f}** | {sign}{delta:.2f} |")


if __name__ == "__main__":
    main()
