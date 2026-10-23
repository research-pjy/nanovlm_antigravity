# NanoVLM: Patch-vs-Convolution Dissection & Ultra-Compact Vision Stems

A clean, modular PyTorch implementation and empirical study extending **"NanoVLMs: How small can we go and still make coherent Vision Language Models?"** (Agarwalla et al., 2025).

This repository resolves the **Patch-then-Conv vs. Conv-then-Patch** architectural ambiguity in the NanoVLM visual encoder and introduces three novel ultra-compact vision-stem & connector architectures (`overlapping_conv_stem`, `convnext_stem`, and `convnext_swiglu`), validated across **42 full training runs** on 28,100 COCO images (`ShortDesc` and `LongDesc`).

---

## Key Headline Results

| Dataset | Scale | Architecture | Total Params | PatchEmbed Params | Best Val Loss | **Total Score (/50)** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **ShortDesc** | `mini` | `patch_then_conv` *(Paper Baseline)* | 5.58M | 1.648M | 2.9718 | 34.88 |
| **ShortDesc** | `mini` | `conv_then_patch_grid` *(Conv First)* | 5.58M | 1.648M | 2.9798 | 36.16 *(+1.28)* |
| **ShortDesc** | `mini` | **`conv_stem_strided`** *(Strided Stem)* | **3.94M** *(-29%)* | **0.009M** | 2.9790 | **38.04** *(+3.16)* |
| **ShortDesc** | `base` | `patch_then_conv` *(Paper Baseline)* | 15.23M | 2.107M | 2.8729 | 38.76 |
| **ShortDesc** | `base` | **`*convnext_swiglu*`** *(Ours)* | **13.31M** *(-13%)* | **0.118M** | **2.8574** | **40.44** *(+1.68, SOTA)* |
| **LongDesc** | `base` | `patch_then_conv` *(Paper Baseline)* | 15.46M | 2.107M | 2.5034 | 34.92 *(38.88 w/ rep=1.15)* |
| **LongDesc** | `base` | **`*convnext_swiglu*`** *(Ours)* | **13.54M** *(-12%)* | **0.118M** | **2.4909** | **39.64** *(Best Base w/ rep=1.15)* |
| **LongDesc** | `large` | `patch_then_conv` *(Paper Baseline)* | 25.73M | 2.107M | 2.4735 | 37.84 *(38.36 w/ rep=1.15)* |
| **LongDesc** | `large` | **`*convnext_stem*`** *(Ours)* | **23.74M** *(-8%)* | **0.118M** | **2.4701** | **40.72** *(+2.36 vs Paper)* |
| **LongDesc** | `large` | **`conv_then_patch_grid`** *(Conv First)* | 25.73M | 2.107M | 2.4810 | **41.04** *(+2.68, Best Overall)* |

- Read the full scientific analysis in **[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md)**.
- Browse all 42 experimental runs in **[`results/summary_phase2_comparison.md`](results/summary_phase2_comparison.md)**.
- Follow the step-by-step reproduction commands in **[`RUNBOOK.md`](RUNBOOK.md)**.

---

## Repository Structure

```text
NanoVLM/
├── nanovlm/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameter presets (mini, base, large) + 6 PatchModes + 2 ConnectorModes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py           # Pre-LN Multi-Head Attention & TransformerBlock (FlashAttention/SDPA)
│   │   ├── vision_encoder.py      # 6 configurable PatchEmbedding architectures + VisionEncoder
│   │   ├── connector.py           # Linear+GELU baseline & 2D Spatial Token Mixer + SwiGLU Projector
│   │   ├── decoder.py             # Causal Multimodal LanguageDecoder
│   │   └── nanovlm.py             # Top-level NanoVLM + repetition-penalized autoregressive generation
│   └── data/
│       ├── __init__.py
│       ├── dataset.py             # ShortDesc / LongDesc dataset loader & paper prompts
│       └── tokenizer.py           # Deterministic child-vocabulary word & punctuation tokenizer
├── scripts/
│   ├── verify_env.py              # L40S / CUDA / dataset integrity verification
│   ├── build_descriptions.py      # Crash-proof batched ShortDesc & LongDesc generator (Qwen3-VL-8B-Instruct)
│   ├── compare_patch_modes.py     # Parameter & shape breakdown across all 6 vision stems
│   ├── train.py                   # BF16 mixed-precision training with cosine schedule & live sample preview
│   ├── evaluate.py                # Held-out ROUGE-1 & 5-metric LLM-as-a-Judge evaluator
│   ├── run_full_matrix.py         # Automated 24-run Phase 1 matrix runner
│   ├── run_phase2_ideas.py        # Automated 18-run Phase 2 novel architecture runner
│   ├── eval_decoding_ablation.py  # Zero-retraining repetition-penalty decoding ablation on LongDesc
│   └── generate_plots.py          # Generates publication-ready PNG figures from results
├── results/
│   ├── summary_results.md         # 24-run Phase 1 baseline benchmark table
│   └── summary_phase2_comparison.md # Complete 42-run Phase 1 + Phase 2 benchmark table
├── RESEARCH_REPORT.md             # Full paper-style technical report & mathematical proofs
├── RUNBOOK.md                     # Exact sequential execution commands for L40S (`rama`)
└── requirements.txt
```
