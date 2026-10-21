#!/usr/bin/env python3
"""Verify the L40S (`rama`) execution environment, dataset paths, and cached Hugging Face models.

Usage on `rama`:
    python scripts/verify_env.py --dataset-dir ~/datasets
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify L40S environment and dataset integrity")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.expanduser("~/datasets"),
        help="Path containing image_selection.json and images/",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("1. PYTHON & PACKAGE VERSIONS")
    print("=" * 80)
    print(f"Python version     : {sys.version.split()[0]}")

    try:
        import torch
        print(f"PyTorch version    : {torch.__version__}")
        print(f"CUDA available     : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime       : {torch.version.cuda}")
            print(f"Device count       : {torch.cuda.device_count()}")
            print(f"Device name        : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram_gib = props.total_memory / (1024 ** 3)
            print(f"Total VRAM         : {vram_gib:.2f} GiB")
            print(f"BF16 supported     : {torch.cuda.is_bf16_supported()}")
    except ImportError as e:
        print(f"[ERROR] PyTorch import failed: {e}")

    for pkg in ("transformers", "accelerate", "PIL", "tqdm"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"{pkg:<18} : {ver}")
        except ImportError:
            print(f"{pkg:<18} : NOT INSTALLED")

    print("\n" + "=" * 80)
    print("2. DISK SPACE & HUGGING FACE CACHE")
    print("=" * 80)
    ds_path = Path(args.dataset_dir).expanduser().resolve()
    check_dir = ds_path if ds_path.exists() else Path.home()
    usage = shutil.disk_usage(check_dir)
    print(f"Disk free at {check_dir}: {usage.free / (1024 ** 3):.2f} GiB / {usage.total / (1024 ** 3):.2f} GiB")

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        cached_models = sorted([p.name for p in hf_cache.iterdir() if p.is_dir() and p.name.startswith("models--")])
        print(f"Cached HF models ({len(cached_models)}):")
        for m in cached_models:
            print(f"  - {m.replace('models--', '').replace('--', '/')}")
    else:
        print(f"HF cache dir not found at {hf_cache}")

    print("\n" + "=" * 80)
    print("3. DATASET MANIFEST & IMAGE VERIFICATION")
    print("=" * 80)
    manifest_path = ds_path / "image_selection.json"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Manifest path      : {manifest_path}")
    print(f"Seed               : {manifest.get('seed')}")
    for split_key in ("held_out", "train", "val"):
        entries = manifest.get(split_key, [])
        missing = 0
        for item in entries[:200]:  # Spot-check first 200 images per split for instant feedback
            img_file = ds_path / "images" / item["split"] / item["file_name"]
            if not img_file.exists():
                missing += 1
        print(
            f"Split '{split_key:<8}' : {len(entries):>6} items | "
            f"Spot-check (first 200 files): {200 - missing if len(entries) >= 200 else len(entries) - missing} found, {missing} missing"
        )

    print("=" * 80)
    print("Verification complete!")


if __name__ == "__main__":
    main()
