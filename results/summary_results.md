# NanoVLM Full-Matrix Experimental Results (Phase 1 Baselines)

| Dataset | Variant | Patch Mode | Params (M) | PatchEmbed (M) | Best Val Loss | Loss Gap | ROUGE-1 | Grammar | Creativity | Consistency | Meaningfulness | Plot | **Total (/50)** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ShortDesc | `mini` | `patch_then_conv` | 5.58M | 1.648M | 2.9718 | 0.2770 | 0.4434 | 6.20 | 8.12 | 6.72 | 7.48 | 6.36 | **34.88** |
| ShortDesc | `mini` | `conv_then_patch_grid` | 5.58M | 1.648M | 2.9798 | 0.2791 | 0.4445 | 6.56 | 8.16 | 7.12 | 7.68 | 6.64 | **36.16** |
| ShortDesc | `mini` | `conv_stem_strided` | 3.94M | 0.009M | 2.9790 | 0.2884 | 0.4444 | 7.00 | 8.40 | 7.52 | 8.12 | 7.00 | **38.04** |
| ShortDesc | `mini` | `standard_vit` | 4.24M | 0.308M | 2.9844 | 0.2853 | 0.4291 | 6.68 | 8.16 | 7.16 | 7.80 | 6.60 | **36.40** |
| ShortDesc | `base` | `patch_then_conv` | 15.23M | 2.107M | 2.8729 | 0.6824 | 0.4356 | 6.96 | 8.56 | 7.80 | 8.28 | 7.16 | **38.76** |
| ShortDesc | `base` | `conv_then_patch_grid` | 15.23M | 2.107M | 2.8861 | 0.6775 | 0.4646 | 6.36 | 8.16 | 6.84 | 7.56 | 6.48 | **35.40** |
| ShortDesc | `base` | `conv_stem_strided` | 13.14M | 0.011M | 2.8640 | 0.6868 | 0.4467 | 7.16 | 8.56 | 7.80 | 8.44 | 7.12 | **39.08** |
| ShortDesc | `base` | `standard_vit` | 13.52M | 0.395M | 2.8739 | 0.6800 | 0.4286 | 6.56 | 8.32 | 7.16 | 7.84 | 6.68 | **36.56** |
| ShortDesc | `large` | `patch_then_conv` | 25.39M | 2.107M | 2.8835 | 1.5077 | 0.4253 | 6.96 | 8.20 | 7.52 | 8.16 | 6.84 | **37.68** |
| ShortDesc | `large` | `conv_then_patch_grid` | 25.39M | 2.107M | 2.8795 | 1.5335 | 0.4374 | 6.84 | 8.36 | 7.52 | 8.12 | 6.96 | **37.80** |
| ShortDesc | `large` | `conv_stem_strided` | 23.29M | 0.011M | 2.8735 | 1.4863 | 0.4519 | 6.64 | 8.20 | 7.16 | 7.92 | 6.76 | **36.68** |
| ShortDesc | `large` | `standard_vit` | 23.68M | 0.395M | 2.8844 | 1.5415 | 0.4512 | 6.60 | 8.28 | 7.24 | 7.88 | 6.72 | **36.72** |
| LongDesc | `mini` | `patch_then_conv` | 5.74M | 1.648M | 2.7350 | 0.0389 | 0.5606 | 5.72 | 8.40 | 6.32 | 7.28 | 6.24 | **33.96** |
| LongDesc | `mini` | `conv_then_patch_grid` | 5.74M | 1.648M | 2.7474 | 0.0379 | 0.5602 | 5.56 | 8.20 | 6.00 | 7.16 | 6.08 | **33.00** |
| LongDesc | `mini` | `conv_stem_strided` | 4.11M | 0.009M | 2.7542 | 0.0408 | 0.5734 | 5.52 | 8.16 | 6.04 | 7.08 | 6.04 | **32.84** |
| LongDesc | `mini` | `standard_vit` | 4.41M | 0.308M | 2.7445 | 0.0401 | 0.5714 | 5.32 | 7.88 | 5.60 | 6.72 | 5.64 | **31.16** |
| LongDesc | `base` | `patch_then_conv` | 15.46M | 2.107M | 2.5034 | 0.2421 | 0.5620 | 5.92 | 8.32 | 6.68 | 7.52 | 6.48 | **34.92** |
| LongDesc | `base` | `conv_then_patch_grid` | 15.46M | 2.107M | 2.5227 | 0.2397 | 0.5801 | 6.64 | 8.76 | 7.60 | 8.16 | 7.16 | **38.32** |
| LongDesc | `base` | `conv_stem_strided` | 13.36M | 0.011M | 2.5049 | 0.2366 | 0.5760 | 6.44 | 8.56 | 7.04 | 7.84 | 6.76 | **36.64** |
| LongDesc | `base` | `standard_vit` | 13.75M | 0.395M | 2.5101 | 0.2416 | 0.5686 | 6.48 | 8.64 | 7.28 | 7.88 | 6.92 | **37.20** |
| LongDesc | `large` | `patch_then_conv` | 25.73M | 2.107M | 2.4735 | 0.7289 | 0.5593 | 6.64 | 8.60 | 7.44 | 8.08 | 7.08 | **37.84** |
| LongDesc | `large` | `conv_then_patch_grid` | 25.73M | 2.107M | 2.4810 | 0.7250 | 0.5835 | 6.76 | 8.72 | 7.72 | 8.32 | 7.32 | **38.84** |
| LongDesc | `large` | `conv_stem_strided` | 23.63M | 0.011M | 2.4788 | 0.7310 | 0.5840 | 6.88 | 8.60 | 7.68 | 8.20 | 7.12 | **38.48** |
| LongDesc | `large` | `standard_vit` | 24.02M | 0.395M | 2.4743 | 0.7283 | 0.5787 | 6.84 | 8.80 | 7.68 | 8.28 | 7.24 | **38.84** |
