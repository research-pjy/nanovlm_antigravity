"""Training script for NanoVLM with mixed-precision (bf16/fp16), word-level tokenizer, and loss history logging.

Usage:
    # Run quick synthetic smoke test:
    python scripts/train.py --variant mini --patch-mode patch_then_conv --synthetic --epochs 2

    # Run on COCO ShortDesc / LongDesc JSON annotations:
    python scripts/train.py --variant mini --patch-mode patch_then_conv \
        --train-json ~/datasets/nanovlm_clean/shortdesc.json \
        --image-root ~/datasets \
        --epochs 20 --batch-size 64 \
        --save-dir ~/nanovlm-runs/shortdesc_mini_patch_then_conv
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from nanovlm.config import get_config
from nanovlm.models.nanovlm import NanoVLM
from nanovlm.data.dataset import NanoVLMDataset
from nanovlm.data.tokenizer import SimpleChildTokenizer


def build_synthetic_loader(batch_size: int, num_samples: int = 128, seq_len: int = 32, vocab_size: int = 8192) -> DataLoader:
    images = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    ds = TensorDataset(images, input_ids, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def get_or_build_tokenizer(json_path: str | Path, save_dir: Path, max_vocab_size: int = 8192) -> SimpleChildTokenizer:
    tok_path = save_dir / "tokenizer.json"
    if tok_path.exists():
        return SimpleChildTokenizer.load(tok_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    if isinstance(data, dict):
        for split_key in ("train", "val", "held_out"):
            for item in data.get(split_key, []):
                texts.append(item["description"])
    elif isinstance(data, list):
        for item in data:
            texts.append(item["description"])

    tokenizer = SimpleChildTokenizer.build_from_texts(texts, max_vocab_size=max_vocab_size)
    tokenizer.save(tok_path)
    return tokenizer


def evaluate(
    model: NanoVLM,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                images = batch["images"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
            else:
                images, input_ids, targets = [t.to(device, non_blocking=True) for t in batch]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(images, input_ids, targets=targets)
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)


@torch.no_grad()
def preview_generation(
    model: NanoVLM,
    val_ds: NanoVLMDataset,
    tokenizer: SimpleChildTokenizer,
    device: torch.device,
    prompt_words: int = 6,
) -> str:
    """Generate a sample completion from the first validation item to monitor fluency."""
    sample = val_ds[0]
    raw_desc = val_ds.samples[0]["description"]
    words = raw_desc.split()
    prefix_text = " ".join(words[:prompt_words])
    prompt_ids = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)

    img = sample["images"].unsqueeze(0).to(device)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out_ids = model.generate(
        img,
        prompt_tensor,
        max_new_tokens=40,
        temperature=0.7,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
    )
    completed = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    return f"Prompt: '{prefix_text}' -> Output: '{completed}'"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoVLM")
    parser.add_argument("--variant", type=str, default="mini", choices=["mini", "base", "large"])
    parser.add_argument(
        "--patch-mode",
        type=str,
        default="patch_then_conv",
        choices=[
            "patch_then_conv",
            "conv_then_patch_grid",
            "conv_stem_strided",
            "standard_vit",
            "overlapping_conv_stem",
            "convnext_stem",
        ],
    )
    parser.add_argument(
        "--connector-mode",
        type=str,
        default="linear_gelu",
        choices=["linear_gelu", "swiglu_token_mixer"],
    )
    parser.add_argument("--train-json", type=str, default=None, help="Path to dataset JSON (e.g. shortdesc.json)")
    parser.add_argument("--val-json", type=str, default=None, help="Optional separate validation JSON")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory containing images/")
    parser.add_argument("--synthetic", action="store_true", help="Run a synthetic training smoke test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Section 2.3: 1e-3)")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    save_path = Path(args.save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer: Optional[SimpleChildTokenizer] = None
    vocab_size = 8192
    if not args.synthetic and args.train_json is not None:
        tokenizer = get_or_build_tokenizer(args.train_json, save_path, max_vocab_size=8192)
        vocab_size = tokenizer.vocab_size
        print(f"[Tokenizer] Built child-vocabulary tokenizer with {vocab_size} tokens -> {save_path / 'tokenizer.json'}")

    cfg = get_config(
        variant=args.variant,  # type: ignore[arg-type]
        patch_mode=args.patch_mode,  # type: ignore[arg-type]
        connector_mode=args.connector_mode,  # type: ignore[arg-type]
        learning_rate=args.lr,
        vocab_size=vocab_size,
    )
    model = NanoVLM(cfg).to(device)

    stats = model.parameter_breakdown()
    print(f"Initialized NanoVLM-{args.variant} (patch_mode={args.patch_mode}) on {device} (AMP={amp_dtype})")
    print(
        f"Total Parameters: {stats['Total']['millions']:.3f}M | "
        f"PatchEmbed: {stats['Patch Embedding (inside Visual Encoder)']['millions']:.3f}M | "
        f"VisEncoder: {stats['Visual Encoder (Total)']['millions']:.3f}M ({stats['Visual Encoder (Total)']['pct']:.1f}%)"
    )

    val_ds = None
    if args.synthetic or args.train_json is None:
        print("Using synthetic data loader for verification...")
        train_loader = build_synthetic_loader(args.batch_size, num_samples=128, vocab_size=cfg.vocab_size)
        val_loader = build_synthetic_loader(args.batch_size, num_samples=32, vocab_size=cfg.vocab_size)
    else:
        pad_id = tokenizer.pad_token_id if tokenizer else 0
        train_ds = NanoVLMDataset(
            args.train_json,
            image_root=args.image_root,
            split="train",
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            pad_token_id=pad_id,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_json_path = args.val_json if args.val_json else args.train_json
        val_ds = NanoVLMDataset(
            val_json_path,
            image_root=args.image_root,
            split="val",
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            pad_token_id=pad_id,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, dict):
                images = batch["images"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
            else:
                images, input_ids, targets = [t.to(device, non_blocking=True) for t in batch]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(images, input_ids, targets=targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype) if val_loader else float("nan")
        dt = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "time_sec": round(dt, 2),
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"gap: {abs(val_loss - train_loss):.4f} | time: {dt:.2f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = save_path / f"nanovlm_{args.variant}_{args.patch_mode}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": cfg,
                    "model_state_dict": model.state_dict(),
                },
                best_ckpt,
            )

        # Print sample completion every 5 epochs (or final epoch)
        if val_ds is not None and tokenizer is not None and (epoch % 5 == 0 or epoch == args.epochs):
            sample_str = preview_generation(model, val_ds, tokenizer, device)
            print(f"  [Sample Gen] {sample_str}")

        # Flush loss history JSON after every epoch
        with open(save_path / "loss_history.json", "w", encoding="utf-8") as f:
            json.dump({"config": vars(cfg), "stats": stats, "history": history}, f, indent=2)

    final_ckpt = save_path / f"nanovlm_{args.variant}_{args.patch_mode}_last.pt"
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, final_ckpt)
    print(f"Saved checkpoints and loss_history.json to {save_path}")


if __name__ == "__main__":
    main()
