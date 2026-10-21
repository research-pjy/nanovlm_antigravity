#!/usr/bin/env python3
"""Build clean, verified ShortDesc and LongDesc datasets for NanoVLM on L40S (`rama`).

Implements Section 2.1.2 & Figure 2/3 of:
'NanoVLMs: How small can we go and still make coherent Vision Language Models?'

Key features to make dataset generation rock-solid and frustration-free:
1. Works directly with your cached `Qwen/Qwen3-VL-8B-Instruct` (in fast text-only mode
   using the 5 COCO captions from `image_selection.json`, or with any causal LLM).
2. Incremental, crash-proof `.jsonl` checkpointing: every batch is flushed to disk
   immediately. If interrupted, re-running the exact same command resumes seamlessly.
3. Strict output cleaning & validation:
   - Strips quotes, markdown, "<think>...</think>" blocks, and chatty prefaces.
   - Removes banned generic openings ("Oh,", "Wow,", "Look,") per Figure 2.
   - Enforces complete sentences and word-count bounds (with automatic retry for outliers).
4. Compiles final ready-to-train `shortdesc.json` and `longdesc.json` files preserving
   the exact `held_out` (100), `train` (25,200), and `val` (2,800) structure.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nanovlm.data.dataset import PROMPT_SHORT_DESC, PROMPT_LONG_DESC


BANNED_STARTS = (
    "oh,", "oh!", "oh ",
    "wow,", "wow!", "wow ",
    "look,", "look!", "look at ",
    "here is", "here's", "sure,", "sure!", "in this image", "the image shows",
)


def build_messages(captions: List[str], mode: str, retry_strict: bool = False) -> List[Dict[str, str]]:
    """Construct chat messages matching Figure 2 & Figure 3 of the NanoVLM paper."""
    base_prompt = PROMPT_SHORT_DESC if mode == "short" else PROMPT_LONG_DESC
    if mode == "short":
        length_rule = (
            "Write ONLY the description itself (2 to 3 complete sentences, strictly between 18 and 28 words total). "
            "Do NOT include any intro, quotes, or word count."
        )
    else:
        length_rule = (
            "Write ONLY the description itself (5 to 6 complete sentences, strictly between 55 and 75 words total). "
            "Do NOT include any intro, quotes, or word count."
        )

    if retry_strict:
        if mode == "short":
            length_rule += " Keep it very concise: 2 short sentences totaling around 22 words!"
        else:
            length_rule += " Make sure to write 5 or 6 full sentences totaling around 65 words!"

    captions_block = "\n".join(f"{i+1}. {c.strip()}" for i, c in enumerate(captions[:5]))
    user_content = (
        f"{base_prompt}\n\n"
        f"{length_rule}\n\n"
        f"5 Captions:\n{captions_block}\n\n"
        f"Description:"
    )

    return [
        {
            "role": "system",
            "content": (
                "You write simple, imaginative, child-friendly scene descriptions using basic English words "
                "for 3-4 year old children. Output ONLY the story description paragraph with no extra commentary."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def sanitize_description(raw_text: str) -> str:
    """Clean LLM output, removing <think> tags, quotes, prefaces, and banned starts."""
    text = raw_text.strip()
    # Strip any <think>...</think> reasoning blocks if a Thinking model was used
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remove conversational prefaces like "Here is a short description:"
    text = re.sub(
        r"^(?:sure[!,.]?\s*)?(?:here(?:'s| is)[^:\n]*:\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Strip surrounding quotes and whitespace
    text = text.strip("\"'` \n\t")
    # Collapse internal newlines/multiple spaces into single space
    text = re.sub(r"\s+", " ", text).strip()

    # Strip banned interjection starts ("Oh, ", "Wow! ", "Look, ") if the model still used one
    lower = text.lower()
    for prefix in ("oh, ", "oh! ", "wow, ", "wow! ", "look, ", "look! "):
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            if text:
                text = text[0].upper() + text[1:]
            break

    # Ensure it ends with proper sentence punctuation if truncated mid-clause
    if text and text[-1] not in ".!?":
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct > len(text) // 2:
            text = text[: last_punct + 1].strip()
        else:
            text = text.rstrip(",;:- ") + "."

    return text


def is_valid_description(text: str, mode: str) -> Tuple[bool, int]:
    """Validate word count and banned prefixes."""
    words = text.split()
    wc = len(words)
    if not text or text[-1] not in ".!?":
        return False, wc

    lower = text.lower()
    if any(lower.startswith(b) for b in BANNED_STARTS):
        return False, wc

    if mode == "short":
        # Target: 20-25 words (accept 14 to 34 as clean natural range before triggering retry)
        return (14 <= wc <= 34), wc
    else:
        # Target: 60-70 words (accept 45 to 85 as clean natural range before triggering retry)
        return (45 <= wc <= 85), wc


def load_teacher_model_and_tokenizer(model_id: str, device: str = "cuda"):
    """Load either Qwen3-VL-8B-Instruct (in text-only mode) or any standard CausalLM."""
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

    print(f"[Model] Loading tokenizer/processor for '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    except Exception:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = getattr(proc, "tokenizer", proc)
        tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[Model] Loading model weights for '{model_id}' in BF16 on {device}...")
    is_vl = "vl" in model_id.lower()
    model = None
    if is_vl:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception:
            pass

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    batch_items: List[Dict[str, Any]],
    mode: str,
    retry_strict: bool = False,
) -> List[str]:
    """Run batched generation in BF16 for a list of manifest items."""
    prompts = []
    for item in batch_items:
        messages = build_messages(item["coco_captions"], mode=mode, retry_strict=retry_strict)
        try:
            # Disable thinking mode if Qwen3 template supports enable_thinking=False
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(rendered)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    max_new_tokens = 60 if mode == "short" else 135
    temp = 0.6 if retry_strict else 0.7

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[:, input_len:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return [sanitize_description(text) for text in decoded]


def load_checkpoint(ckpt_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load already-generated items from an existing JSONL checkpoint file."""
    completed: Dict[int, Dict[str, Any]] = {}
    if not ckpt_path.exists():
        return completed

    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completed[int(obj["image_id"])] = obj
            except Exception as e:
                print(f"[Warning] Skipping malformed line {line_num} in {ckpt_path.name}: {e}")
    return completed


def process_mode(
    mode: str,
    manifest: Dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    limit: int,
) -> None:
    """Generate, checkpoint, validate, and compile dataset for either 'short' or 'long'."""
    name = "shortdesc" if mode == "short" else "longdesc"
    ckpt_path = output_dir / f"{name}_checkpoint.jsonl"
    final_json_path = output_dir / f"{name}.json"

    completed = load_checkpoint(ckpt_path)
    print(f"\n{'=' * 80}")
    print(f"[{name.upper()}] Loaded {len(completed)} existing records from {ckpt_path.name}")
    print(f"{'=' * 80}")

    # Collect all items in deterministic order: held_out -> train -> val
    all_tasks: List[Tuple[str, Dict[str, Any]]] = []
    for split_name in ("held_out", "train", "val"):
        entries = manifest.get(split_name, [])
        if limit > 0:
            entries = entries[:limit]
        for entry in entries:
            if int(entry["image_id"]) not in completed:
                all_tasks.append((split_name, entry))

    print(f"[{name.upper()}] Remaining items to generate: {len(all_tasks)} (batch_size={batch_size})")

    if all_tasks:
        t0 = time.time()
        with open(ckpt_path, "a", encoding="utf-8") as ckpt_file:
            pbar = tqdm(range(0, len(all_tasks), batch_size), desc=f"Generating {name}")
            for start_idx in pbar:
                batch_slice = all_tasks[start_idx : start_idx + batch_size]
                batch_entries = [item for _, item in batch_slice]

                descriptions = generate_batch(model, tokenizer, batch_entries, mode=mode, retry_strict=False)

                # Check for any outliers that need a quick strict retry
                retry_indices = []
                for idx, desc in enumerate(descriptions):
                    ok, _ = is_valid_description(desc, mode=mode)
                    if not ok:
                        retry_indices.append(idx)

                if retry_indices:
                    retry_entries = [batch_entries[i] for i in retry_indices]
                    retry_descs = generate_batch(model, tokenizer, retry_entries, mode=mode, retry_strict=True)
                    for i, new_desc in zip(retry_indices, retry_descs):
                        ok_new, wc_new = is_valid_description(new_desc, mode=mode)
                        _, wc_old = is_valid_description(descriptions[i], mode=mode)
                        target_mid = 22 if mode == "short" else 65
                        if ok_new or abs(wc_new - target_mid) < abs(wc_old - target_mid):
                            descriptions[i] = new_desc

                # Immediately flush batch to JSONL checkpoint
                for (split_name, entry), desc in zip(batch_slice, descriptions):
                    wc = len(desc.split())
                    record = {
                        "dataset_split": split_name,
                        "image_id": int(entry["image_id"]),
                        "file_name": entry["file_name"],
                        "coco_split": entry["split"],
                        "image_path": f"images/{entry['split']}/{entry['file_name']}",
                        "description": desc,
                        "word_count": wc,
                        "coco_captions": entry["coco_captions"],
                    }
                    ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed[int(entry["image_id"])] = record
                ckpt_file.flush()

        elapsed = time.time() - t0
        print(f"[{name.upper()}] Generation finished in {elapsed / 60:.2f} minutes.")

    # Compile final structured JSON dataset
    compiled_splits: Dict[str, List[Dict[str, Any]]] = {"held_out": [], "train": [], "val": []}
    word_counts: List[int] = []

    for split_name in ("held_out", "train", "val"):
        entries = manifest.get(split_name, [])
        if limit > 0:
            entries = entries[:limit]
        for entry in entries:
            img_id = int(entry["image_id"])
            if img_id in completed:
                rec = completed[img_id]
                compiled_splits[split_name].append(rec)
                word_counts.append(rec["word_count"])

    avg_wc = sum(word_counts) / max(len(word_counts), 1)
    min_wc = min(word_counts) if word_counts else 0
    max_wc = max(word_counts) if word_counts else 0

    final_payload = {
        "dataset_name": "ShortDesc" if mode == "short" else "LongDesc",
        "seed": manifest.get("seed", 42),
        "stats": {
            "total_records": len(word_counts),
            "held_out_count": len(compiled_splits["held_out"]),
            "train_count": len(compiled_splits["train"]),
            "val_count": len(compiled_splits["val"]),
            "avg_word_count": round(avg_wc, 2),
            "min_word_count": min_wc,
            "max_word_count": max_wc,
        },
        "held_out": compiled_splits["held_out"],
        "train": compiled_splits["train"],
        "val": compiled_splits["val"],
    }

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print(f"[{name.upper()}] Compiled final dataset -> {final_json_path}")
    print(f"[{name.upper()}] Stats: {final_payload['stats']}")

    # Print 3 sample descriptions for immediate human inspection
    sample_list = compiled_splits["held_out"][:3] or compiled_splits["train"][:3]
    print(f"\n--- Sample {name.upper()} Outputs ---")
    for s in sample_list:
        print(f"  [image_id={s['image_id']} | {s['word_count']} words]: {s['description']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NanoVLM ShortDesc and LongDesc datasets")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.expanduser("~/datasets"),
        help="Directory containing image_selection.json and images/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/datasets/nanovlm_clean"),
        help="Directory to store checkpoint .jsonl and compiled .json files",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Hugging Face teacher model ID (cached on rama: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["short", "long", "both"],
        help="Generate 'short' (ShortDesc), 'long' (LongDesc), or 'both'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for BF16 inference on L40S (64 uses ~20-24 GiB VRAM)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process the first N items per split (use --limit 5 for smoke test!)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = dataset_dir / "image_selection.json"
    if not manifest_path.exists():
        print(f"[ERROR] Could not find manifest at {manifest_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model, tokenizer = load_teacher_model_and_tokenizer(args.model_id)

    modes = ["short", "long"] if args.mode == "both" else [args.mode]
    for m in modes:
        process_mode(
            mode=m,
            manifest=manifest,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
