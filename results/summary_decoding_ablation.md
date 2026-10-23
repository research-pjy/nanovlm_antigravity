# LongDesc Repetition-Penalty Decoding Comparison (`temperature=0.5`, `repetition_penalty=1.15`)

Evaluating the exact same trained `LongDesc` (`base` and `large`) checkpoints with repetition-penalized decoding (`repetition_penalty=1.15`, `temperature=0.5`) versus unpenalized sampling (`temperature=0.7`, `repetition_penalty=1.0`):

| Variant | Architecture | Params (M) | PatchEmbed (M) | Best Val Loss | Baseline Score (`temp=0.7`) | Penalized Score (`rep=1.15`) | **Delta** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `base` | `patch_then_conv` *(Paper)* | 15.46M | 2.107M | 2.5034 | 34.92 | **38.88** | **+3.96** |
| `base` | `conv_then_patch_grid` | 15.46M | 2.107M | 2.5227 | 38.32 | **38.20** | -0.12 |
| `base` | `conv_stem_strided` | **13.36M** | **0.011M** | 2.5049 | 36.64 | **39.08** | **+2.44** |
| `base` | `standard_vit` | 13.75M | 0.395M | 2.5101 | 37.20 | **37.40** | +0.20 |
| `base` | `*overlapping_conv_stem*` | 13.48M | 0.132M | 2.5017 | 37.40 | **38.24** | +0.84 |
| `base` | `*convnext_stem*` *(Ours)* | **13.47M** | **0.118M** | 2.4993 | 37.12 | **39.28** | **+2.16** |
| `base` | **`*convnext_swiglu*` *(Ours)*** | **13.54M** | **0.118M** | **2.4909** | 37.00 | **39.64** *(Best Base)* | **+2.64** |
| `large` | `patch_then_conv` *(Paper)* | 25.73M | 2.107M | 2.4735 | 37.84 | **38.36** | +0.52 |
| `large` | **`conv_then_patch_grid`** | 25.73M | 2.107M | 2.4810 | 38.84 | **41.04** *(Best Overall)* | **+2.20** |
| `large` | `conv_stem_strided` | **23.63M** | **0.011M** | 2.4788 | 38.48 | **39.88** | **+1.40** |
| `large` | `standard_vit` | 24.02M | 0.395M | 2.4743 | 38.84 | **38.84** | +0.00 |
| `large` | `*overlapping_conv_stem*` | 23.75M | 0.132M | 2.4726 | 37.64 | **39.68** | **+2.04** |
| `large` | **`*convnext_stem*` *(Ours)*** | **23.74M** | **0.118M** | 2.4701 | 37.40 | **40.72** *(Best Compact)* | **+3.32** |
| `large` | `*convnext_swiglu*` *(Ours)* | 23.84M | 0.118M | **2.4664** | 36.80 | **38.60** | **+1.80** |
