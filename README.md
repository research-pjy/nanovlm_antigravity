# NanoVLM: Clean PyTorch Recreation & Patch-vs-Convolution Testbed

A clean, modular PyTorch implementation of **"NanoVLMs: How small can we go and still make coherent Vision Language Models?"** (Agarwalla et al., 2025), designed to test and compare architectural variations in the Visual Encoder's patch-and-convolution pipeline.

## Project Structure

```text
NanoVLM/
├── nanovlm/
│   ├── __init__.py
│   ├── config.py                # Hyperparameter presets (mini: 5M, base: 16M, large: 25M)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py         # Pre-LN Multi-Head Attention & TransformerBlock (Flash/SDPA ready)
│   │   ├── vision_encoder.py    # Configurable PatchEmbedding (4 modes) + VisionEncoder
│   │   ├── connector.py         # MultimodalProjector (Linear + GELU)
│   │   ├── decoder.py           # Autoregressive Causal LanguageDecoder
│   │   └── nanovlm.py           # Full NanoVLM model + parameter breakdown utility
│   └── data/
│       ├── __init__.py
│       └── dataset.py           # ShortDesc / LongDesc dataset loader & GPT-4o prompts
├── scripts/
│   ├── compare_patch_modes.py   # Compares parameter counts & module splits across all 4 patch modes
│   └── train.py                 # Mixed-precision (bf16/fp16) training pipeline for L40S / A100
└── requirements.txt
```

## Comparing the Patch vs. Convolution Modes

Run the parameter and shape comparison script:

```bash
python scripts/compare_patch_modes.py
```

The four selectable `--patch-mode` options in [`PatchEmbedding`](nanovlm/models/vision_encoder.py) are:
1. `patch_then_conv`: Divide $224\times 224$ image into $196$ patches of $(3, 16, 16)$ $\to$ 2x `Conv2d` per patch $\to$ Flatten ($4096$-d) $\to$ `Linear(4096, img_embd_dim)`.
2. `conv_then_patch_grid`: Run 2x `Conv2d` on the full $(3, 224, 224)$ image first $\to$ slice feature map into $196$ patches of $(C_{\text{out}}, 16, 16)$ $\to$ Flatten ($4096$-d) $\to$ `Linear(4096, img_embd_dim)`. *(Identical parameter count to `patch_then_conv`, but convolutions share receptive fields across patch boundaries!)*
3. `conv_stem_strided`: Run 2x strided `Conv2d` (stride 4 each) on the full $(3, 224, 224)$ image to downsample directly to $(C_{\text{out}}, 14, 14)$ ($196$ spatial tokens) $\to$ `Linear(C_out, img_embd_dim)`. *(Eliminates ~1.6M–2.1M parameters from the FC bottleneck!)*
4. `standard_vit`: Standard ViT single `Conv2d(3, img_embd_dim, kernel_size=16, stride=16)` baseline.

## Training

```bash
# Quick synthetic smoke test:
python scripts/train.py --variant mini --patch-mode patch_then_conv --synthetic --epochs 2

# Train on ShortDesc / LongDesc dataset:
python scripts/train.py --variant base --patch-mode conv_then_patch_grid \
    --train-json data/short_desc_train.json --val-json data/short_desc_val.json \
    --image-root data/coco/train2017 --epochs 20 --batch-size 64
```
