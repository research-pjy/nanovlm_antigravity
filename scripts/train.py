"""Training script for NanoVLM with mixed-precision (bf16/fp16) support for L40S / A100 GPUs.

Usage:
    # Run quick synthetic smoke test (no dataset required):
    python scripts/train.py --variant mini --patch-mode patch_then_conv --synthetic --epochs 2

    # Run on actual COCO ShortDesc / LongDesc JSON annotations:
    python scripts/train.py --variant base --patch-mode conv_then_patch_grid \
        --train-json data/short_desc_train.json --val-json data/short_desc_val.json \
        --epochs 20 --batch-size 64
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from nanovlm.config import get_config
from nanovlm.models.nanovlm import NanoVLM
from nanovlm.data.dataset import NanoVLMDataset


def build_synthetic_loader(batch_size: int, num_samples: int = 128, seq_len: int = 32, vocab_size: int = 8192) -> DataLoader:
    images = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    ds = TensorDataset(images, input_ids, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def evaluate(model: NanoVLM, val_loader: DataLoader, device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoVLM")
    parser.add_argument("--variant", type=str, default="mini", choices=["mini", "base", "large"])
    parser.add_argument(
        "--patch-mode",
        type=str,
        default="patch_then_conv",
        choices=["patch_then_conv", "conv_then_patch_grid", "conv_stem_strided", "standard_vit"],
    )
    parser.add_argument("--train-json", type=str, default=None, help="Path to training annotations JSON/JSONL")
    parser.add_argument("--val-json", type=str, default=None, help="Path to validation annotations JSON/JSONL")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory for COCO images")
    parser.add_argument("--synthetic", action="store_true", help="Run a synthetic training smoke test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Section 2.3: 1e-3)")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    cfg = get_config(variant=args.variant, patch_mode=args.patch_mode, learning_rate=args.lr)  # type: ignore[arg-type]
    model = NanoVLM(cfg).to(device)

    stats = model.parameter_breakdown()
    print(f"Initialized NanoVLM-{args.variant} (patch_mode={args.patch_mode}) on {device}")
    print(f"Total Trainable Parameters: {stats['Total']['millions']:.3f}M")

    if args.synthetic or args.train_json is None:
        print("Using synthetic data loader for verification...")
        train_loader = build_synthetic_loader(args.batch_size, num_samples=128, vocab_size=cfg.vocab_size)
        val_loader = build_synthetic_loader(args.batch_size, num_samples=32, vocab_size=cfg.vocab_size)
    else:
        train_ds = NanoVLMDataset(args.train_json, image_root=args.image_root, split="train", max_seq_len=cfg.max_seq_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_json_path = args.val_json if args.val_json else args.train_json
        val_ds = NanoVLMDataset(val_json_path, image_root=args.image_root, split="val", max_seq_len=cfg.max_seq_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

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
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype) if val_loader else float("nan")
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | time: {dt:.2f}s")

    ckpt_file = save_path / f"nanovlm_{args.variant}_{args.patch_mode}.pt"
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, ckpt_file)
    print(f"Saved checkpoint to {ckpt_file}")


if __name__ == "__main__":
    main()
