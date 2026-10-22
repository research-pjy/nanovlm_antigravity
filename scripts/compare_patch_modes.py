"""Compare parameter counts, module percentages, and forward-pass shapes across Patch vs. Conv modes.

Usage:
    python scripts/compare_patch_modes.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nanovlm.config import get_config, PatchMode
from nanovlm.models.nanovlm import NanoVLM


PATCH_MODES: list[PatchMode] = [
    "patch_then_conv",
    "conv_then_patch_grid",
    "conv_stem_strided",
    "standard_vit",
    "overlapping_conv_stem",
    "convnext_stem",
]

VARIANTS = ["mini", "base", "large"]


def run_comparison() -> None:
    print("=" * 95)
    print("NanoVLM Architecture & Patch-vs-Convolution Parameter Comparison")
    print("=" * 95)

    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_txt = torch.randint(0, 1000, (2, 24))
    dummy_tgt = torch.randint(0, 1000, (2, 24))

    for variant in VARIANTS:
        print(f"\n>>> Variant: NanoVLM-{variant.upper()} <<<")
        print("-" * 95)
        header = (
            f"{'Patch Mode':<22} | {'PatchEmbed':>12} | {'VisEncoder':>14} | "
            f"{'Projector':>12} | {'Decoder':>12} | {'TOTAL':>10}"
        )
        print(header)
        print("-" * 95)

        for mode in PATCH_MODES:
            cfg = get_config(variant=variant, patch_mode=mode)  # type: ignore[arg-type]
            model = NanoVLM(cfg)
            stats = model.parameter_breakdown()

            # Verify forward pass & shapes
            with torch.no_grad():
                logits, loss = model(dummy_img, dummy_txt, targets=dummy_tgt)
                assert logits.shape == (2, 24, cfg.vocab_size)
                assert loss is not None

            pe = stats["Patch Embedding (inside Visual Encoder)"]
            ve = stats["Visual Encoder (Total)"]
            pr = stats["Multimodal Projector"]
            de = stats["Language Decoder"]
            tot = stats["Total"]

            print(
                f"{mode:<22} | "
                f"{pe['millions']:>6.3f}M ({pe['pct']:>2.0f}%) | "
                f"{ve['millions']:>6.3f}M ({ve['pct']:>4.1f}%) | "
                f"{pr['millions']:>5.3f}M ({pr['pct']:>3.1f}%) | "
                f"{de['millions']:>5.3f}M ({de['pct']:>3.1f}%) | "
                f"{tot['millions']:>7.3f}M"
            )
        print("-" * 95)


if __name__ == "__main__":
    run_comparison()
