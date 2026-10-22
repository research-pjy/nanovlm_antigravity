# NanoVLM Execution Runbook (`rama` L40S 44 GiB)

Follow these commands in exact sequential order on `rama`.

---

## Step 1: Activate Environment & Pull Latest Code on `rama`

```bash
conda activate qwen-vl
cd ~/projects/NanoVLM   # (or wherever you clone/pull this repository on rama)
git pull
```

---

## Step 2: Verify L40S GPU, Conda Packages, Disk Space & COCO 28K Manifest

Run the verification script to confirm PyTorch sees the L40S, BF16 is active, `Qwen/Qwen3-VL-8B-Instruct` is cached, and `~/datasets/image_selection.json` + `~/datasets/images/` are intact:

```bash
python scripts/verify_env.py --dataset-dir ~/datasets
```

---

## Step 3: Smoke-Test `ShortDesc` & `LongDesc` Generation (First 5 Images Only)

Before generating all 28,100 images, run a **15-second smoke test** (`--limit 5`) into a temporary test folder (`~/datasets/nanovlm_smoke_test`) to inspect the generated descriptions, word counts, and GPU memory usage:

```bash
python scripts/build_descriptions.py \
    --dataset-dir ~/datasets \
    --output-dir ~/datasets/nanovlm_smoke_test \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --mode both \
    --batch-size 16 \
    --limit 5
```

**What to check in the terminal output:**
- `ShortDesc` samples should be 2–3 simple, child-friendly sentences (~18–28 words) with no quotes or `"Oh/Wow/Look"` starts.
- `LongDesc` samples should be 5–6 simple, child-friendly sentences (~55–75 words).
- Check `~/datasets/nanovlm_smoke_test/shortdesc.json` and `longdesc.json`.

---

## Step 4: Full `ShortDesc` and `LongDesc` Dataset Creation (28,100 Images)

Once Step 3 looks good, run the full generation into a clean, dedicated folder (`~/datasets/nanovlm_clean`).
- **Crash-proof / Resume-safe:** Every batch of 64 is immediately flushed to `shortdesc_checkpoint.jsonl` and `longdesc_checkpoint.jsonl`. If interrupted for any reason, running the exact same command resumes from where it stopped.
- **Tip:** Run inside `tmux` or `screen` (e.g., `tmux new -s nanovlm_data`) so SSH disconnects never interrupt your run.

### 4A. Build `ShortDesc` (100 held_out + 25,200 train + 2,800 val = 28,100 total)

```bash
python scripts/build_descriptions.py \
    --dataset-dir ~/datasets \
    --output-dir ~/datasets/nanovlm_clean \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --mode short \
    --batch-size 64
```

### 4B. Build `LongDesc` (100 held_out + 25,200 train + 2,800 val = 28,100 total)

```bash
python scripts/build_descriptions.py \
    --dataset-dir ~/datasets \
    --output-dir ~/datasets/nanovlm_clean \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --mode long \
    --batch-size 64
```

*(Note: If `nvidia-smi` shows plenty of free VRAM at `--batch-size 64`, you can increase `--batch-size 96` or `128` for even faster throughput).*

---

## Step 5: Verify Parameter Counts Across All 4 Patch-vs-Conv Modes

While or after the dataset builds, verify the NanoVLM model architecture and compare parameter counts across `patch_then_conv`, `conv_then_patch_grid`, `conv_stem_strided`, and `standard_vit`:

```bash
python scripts/compare_patch_modes.py
```

---

## Step 6: End-to-End Training Smoke Test (on Smoke-Test Dataset)

Before launching full 28K training, run 2 epochs on the smoke-test dataset to verify the full pipeline (`DataLoader -> VisionEncoder -> Projector -> Decoder -> BF16 Loss -> Backward -> Checkpoint`):

```bash
python scripts/train.py \
    --variant mini \
    --patch-mode patch_then_conv \
    --train-json ~/datasets/nanovlm_smoke_test/shortdesc.json \
    --image-root ~/datasets \
    --epochs 2 \
    --batch-size 8 \
    --save-dir ~/nanovlm-runs/smoke_test
```

---

## Step 7: Full Training Runs (Testing Patch-then-Conv vs. Conv-then-Patch)

Once `~/datasets/nanovlm_clean/shortdesc.json` and `longdesc.json` are ready:

```bash
# 1. Patch-first then 2D Conv per patch (Paper baseline)
python scripts/train.py \
    --variant mini \
    --patch-mode patch_then_conv \
    --train-json ~/datasets/nanovlm_clean/shortdesc.json \
    --image-root ~/datasets \
    --epochs 20 \
    --batch-size 64 \
    --save-dir ~/nanovlm-runs/shortdesc_mini_patch_then_conv

# 2. 2D Conv on full 224x224 image first, then 16x16 grid patching
python scripts/train.py \
    --variant mini \
    --patch-mode conv_then_patch_grid \
    --train-json ~/datasets/nanovlm_clean/shortdesc.json \
    --image-root ~/datasets \
    --epochs 20 \
    --batch-size 64 \
    --save-dir ~/nanovlm-runs/shortdesc_mini_conv_then_patch

# 3. Strided 2D Conv stem (224x224 -> 14x14 tokens directly, lowest parameter count)
python scripts/train.py \
    --variant mini \
    --patch-mode conv_stem_strided \
    --train-json ~/datasets/nanovlm_clean/shortdesc.json \
    --image-root ~/datasets \
    --epochs 20 \
    --batch-size 64 \
    --save-dir ~/nanovlm-runs/shortdesc_mini_conv_stem_strided
```

---

## Step 8: Evaluate Trained Models (ROUGE-1 + Table 3 LLM-as-a-Judge)

Evaluate each trained model on the 25 held-out test samples:

```bash
# 1. Evaluate patch_then_conv
python scripts/evaluate.py \
    --checkpoint ~/nanovlm-runs/shortdesc_mini_patch_then_conv/nanovlm_mini_patch_then_conv_best.pt \
    --tokenizer ~/nanovlm-runs/shortdesc_mini_patch_then_conv/tokenizer.json \
    --dataset-json ~/datasets/nanovlm_clean/shortdesc.json \
    --image-root ~/datasets \
    --mode short \
    --run-judge

# 2. Evaluate conv_then_patch_grid
python scripts/evaluate.py \
    --checkpoint ~/nanovlm-runs/shortdesc_mini_conv_then_patch/nanovlm_mini_conv_then_patch_grid_best.pt \
    --tokenizer ~/nanovlm-runs/shortdesc_mini_conv_then_patch/tokenizer.json \
    --dataset-json ~/datasets/nanovlm_clean/shortdesc.json \
    --image-root ~/datasets \
    --mode short \
    --run-judge

# 3. Evaluate conv_stem_strided
python scripts/evaluate.py \
    --checkpoint ~/nanovlm-runs/shortdesc_mini_conv_stem_strided/nanovlm_mini_conv_stem_strided_best.pt \
    --tokenizer ~/nanovlm-runs/shortdesc_mini_conv_stem_strided/tokenizer.json \
    --dataset-json ~/datasets/nanovlm_clean/shortdesc.json \
    --image-root ~/datasets \
    --mode short \
    --run-judge
```

---

## Step 9: Automated Full-Matrix Runner (All 24 Experiments + Summary Report)

To train and evaluate the entire matrix across both datasets (`ShortDesc`, `LongDesc`), all 3 model sizes (`mini`, `base`, `large`), and all 4 patch modes (`patch_then_conv`, `conv_then_patch_grid`, `conv_stem_strided`, `standard_vit`):

- **Automatically skips** any run that already has a trained `*_best.pt` checkpoint and `eval_*.json` (so your completed `shortdesc_mini_*` runs are reused immediately!).
- **Loads `Qwen/Qwen3-VL-8B-Instruct` only ONCE** at the end to batch-grade all newly trained models.
- **Outputs:** `~/nanovlm-runs/summary_results.md`, `~/nanovlm-runs/summary_results.csv`, and `~/nanovlm-runs/summary_results.json`.

```bash
# Run the complete 24-experiment matrix (recommended inside tmux):
python scripts/run_full_matrix.py \
    --dataset-dir ~/datasets/nanovlm_clean \
    --image-root ~/datasets \
    --runs-dir ~/nanovlm-runs \
    --epochs 20 \
    --batch-size 64
```

*(Optional: If you want to finish `ShortDesc` across `mini`, `base`, and `large` first before `LongDesc`, pass `--datasets short` and then `--datasets long`.)*

---

## Step 10: Phase 2 — Test Our 3 Novel Architectural Ideas (`run_phase2_ideas.py`)

Now that Phase 1 (`patch_then_conv`, `conv_then_patch_grid`, `conv_stem_strided`, `standard_vit`) is complete, test our 3 custom architectural innovations designed to solve the 16-d bottleneck of `conv_stem_strided` while keeping the ~1.6M–2.0M parameter savings:

1. **`overlapping_conv_stem`**: Overlapping $7\times 7(\text{s}4) \to 5\times 5(\text{s}4)$ Conv stem ($3 \to 32 \to 96$) + BatchNorm + GELU.
2. **`convnext_stem`**: Overlapping stem + $7\times 7$ Depthwise-Separable ConvNeXt residual block across the $14\times 14$ patch grid.
3. **`convnext_swiglu`**: Combines `convnext_stem` with a **2D Spatial Depthwise Token Mixer + SwiGLU Gated Multimodal Projector** (`--connector-mode swiglu_token_mixer`).

```bash
# First check the parameter counts of the new modes:
python scripts/compare_patch_modes.py

# Run all Phase 2 novel architectures across ShortDesc and LongDesc (mini, base, large)
# and automatically generate ~/nanovlm-runs/summary_phase2_comparison.md:
python scripts/run_phase2_ideas.py \
    --dataset-dir ~/datasets/nanovlm_clean \
    --image-root ~/datasets \
    --runs-dir ~/nanovlm-runs \
    --epochs 20 \
    --batch-size 64
```
