#!/usr/bin/env python3
"""Evaluate trained NanoVLM models on the 25 held-out samples (ROUGE-1 + LLM-as-a-Judge).

Implements Section 3 & Section 4 (Table 3 & Figure 8) of the NanoVLM paper:
- Takes the first 6-7 words (ShortDesc) or 18-20 words (LongDesc) of each held-out sample
  along with its image, and generates the completion using NanoVLM.
- Computes ROUGE-1 overlap against the training/reference descriptions (Figure 8).
- Optionally grades the completions on Grammar, Creativity, Consistency, Meaningfulness,
  and Plot (each /10) using local `Qwen/Qwen3-VL-8B-Instruct` via Prompt 3 (Figure 2).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nanovlm.models.nanovlm import NanoVLM
from nanovlm.data.dataset import NanoVLMDataset, PROMPT_EVAL_JUDGE
from nanovlm.data.tokenizer import SimpleChildTokenizer


def compute_rouge1_f1(pred_text: str, ref_text: str) -> float:
    """Zero-dependency unigram ROUGE-1 F1 score."""
    pred_tokens = SimpleChildTokenizer.split_tokens(pred_text)
    ref_tokens = SimpleChildTokenizer.split_tokens(ref_text)
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    for t in pred_tokens:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1

    prec = overlap / len(pred_tokens)
    rec = overlap / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def parse_judge_scores(text: str) -> Dict[str, float]:
    """Extract numeric scores out of 10 for Grammar, Creativity, Consistency, Meaningfulness, Plot."""
    metrics = ["grammar", "creativity", "consistency", "meaningfulness", "plot"]
    scores: Dict[str, float] = {}
    lower = text.lower()
    for m in metrics:
        match = re.search(rf"{m}\s*[:\-=]?\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?", lower)
        if match:
            val = float(match.group(1))
            scores[m] = min(max(val, 0.0), 10.0)
        else:
            scores[m] = 5.0
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NanoVLM on held-out test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json in run dir")
    parser.add_argument("--dataset-json", type=str, required=True, help="Path to shortdesc.json or longdesc.json")
    parser.add_argument("--image-root", type=str, required=True, help="Path to ~/datasets")
    parser.add_argument("--num-samples", type=int, default=25, help="Number of held-out samples (Paper: 25)")
    parser.add_argument("--mode", type=str, default="short", choices=["short", "long"])
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (<=0 for greedy)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (e.g. 1.15)")
    parser.add_argument("--run-judge", action="store_true", help="Grade completions with Qwen3-VL-8B-Instruct")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleChildTokenizer.load(args.tokenizer)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NanoVLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = NanoVLMDataset(
        args.dataset_json,
        image_root=args.image_root,
        split="held_out",
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )

    num_eval = min(args.num_samples, len(ds))
    prompt_words = 6 if args.mode == "short" else 18
    max_new_tokens = 30 if args.mode == "short" else 65

    results: List[Dict] = []
    rouge_scores: List[float] = []

    print(f"Evaluating {num_eval} held-out samples ({args.mode.upper()} mode, prompt_words={prompt_words})...")
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

    avg_rouge = sum(rouge_scores) / max(len(rouge_scores), 1)
    print(f"Average ROUGE-1 Score: {avg_rouge:.4f} (Paper target < 0.50 to confirm non-memorization)")

    avg_metrics = None
    avg_total = None
    if args.run_judge:
        from scripts.build_descriptions import load_teacher_model_and_tokenizer
        judge_model, judge_tok = load_teacher_model_and_tokenizer(args.judge_model)

        metric_totals = {k: 0.0 for k in ["grammar", "creativity", "consistency", "meaningfulness", "plot"]}
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

            inputs = judge_tok([rendered], return_tensors="pt").to(judge_model.device)
            with torch.no_grad():
                out = judge_model.generate(**inputs, max_new_tokens=80, do_sample=False)
            reply = judge_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            scores = parse_judge_scores(reply)
            item["scores"] = scores
            item["total_score"] = round(sum(scores.values()), 2)
            for k, v in scores.items():
                metric_totals[k] += v

        avg_metrics = {k: round(v / num_eval, 2) for k, v in metric_totals.items()}
        avg_total = round(sum(avg_metrics.values()), 2)
        print("\n=== Table 3 Benchmark Scores ===")
        print(f"Grammar: {avg_metrics['grammar']} | Creativity: {avg_metrics['creativity']} | "
              f"Consistency: {avg_metrics['consistency']} | Meaningfulness: {avg_metrics['meaningfulness']} | "
              f"Plot: {avg_metrics['plot']} || Average Total: {avg_total} / 50")

    out_path = Path(args.checkpoint).parent / f"eval_{args.mode}.json"
    payload = {"avg_rouge1": round(avg_rouge, 4), "results": results}
    if avg_metrics is not None:
        payload["avg_metrics"] = avg_metrics
        payload["avg_total"] = avg_total
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
