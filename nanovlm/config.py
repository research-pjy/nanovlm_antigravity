"""Configuration definitions for NanoVLM models (mini, base, large) and experimental patch modes."""

from dataclasses import dataclass
from typing import Literal

PatchMode = Literal[
    "patch_then_conv",       # Divide 224x224 into 196x(3,16,16) patches -> 2x Conv2d per patch -> Flatten -> Linear
    "conv_then_patch_grid",  # 2x Conv2d on full 224x224 image -> Unfold into 196x(C,16,16) grids -> Flatten -> Linear
    "conv_stem_strided",     # 2x Strided Conv2d on full 224x224 image -> (C,14,14) feature map -> 196 tokens -> Linear
    "standard_vit",          # Standard ViT single Conv2d(3, img_embd_dim, kernel_size=16, stride=16) baseline
]


@dataclass
class NanoVLMConfig:
    """Hyperparameter configuration for NanoVLM.
    
    Matches Table 1 and Section 2.3 of:
    'NanoVLMs: How small can we go and still make coherent Vision Language Models?'
    """
    model_variant: str = "mini"
    
    # Image & Patch parameters (Section 2.3: fixed across all versions)
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    
    # Vision Encoder parameters (Table 1)
    n_blks: int = 1                # Number of transformer blocks in visual encoder
    img_embd_dim: int = 400        # Visual embedding dimension
    vis_n_head: int = 8            # Number of attention heads in visual encoder
    conv_hidden_channels: int = 8  # Output channels of 1st 2D conv layer
    conv_out_channels: int = 16    # Output channels of 2nd 2D conv layer
    patch_mode: PatchMode = "patch_then_conv"
    
    # Multimodal Projector parameters (Section 2.2.2)
    # To match Table 2 parameter ratios (14% mini, 8% base, 6% large),
    # the projector maps flattened visual representation or token projection into n_embd.
    projector_use_token_pooling: bool = False
    
    # Language Decoder parameters (Table 1)
    n_layer: int = 4               # Number of transformer layers in decoder
    n_head: int = 8                # Number of attention heads in decoder
    head_size: int = 12            # Size of each attention head in decoder (n_embd = n_head * head_size)
    n_embd: int = 96               # Textual embedding dimension
    vocab_size: int = 8192         # Vocabulary size for text token embeddings & LM head
    max_seq_len: int = 128         # Maximum text sequence length
    
    # Regularization & Training (Section 2.3)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    mlp_ratio: int = 4

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_vis_tokens(self) -> int:
        # 196 patch tokens + 1 prepended [CLS] token = 197 tokens (Section 2.2.1)
        return self.num_patches + 1


def get_config(
    variant: Literal["mini", "base", "large"] = "mini",
    patch_mode: PatchMode = "patch_then_conv",
    **overrides,
) -> NanoVLMConfig:
    """Return the paper's exact Table 1 configuration for 'mini', 'base', or 'large'."""
    presets = {
        "mini": dict(
            model_variant="mini",
            n_blks=1,
            n_layer=4,
            n_head=8,
            head_size=12,
            n_embd=96,
            img_embd_dim=400,
        ),
        "base": dict(
            model_variant="base",
            n_blks=3,
            n_layer=8,
            n_head=8,
            head_size=16,
            n_embd=128,
            img_embd_dim=512,
        ),
        "large": dict(
            model_variant="large",
            n_blks=5,
            n_layer=10,
            n_head=16,
            head_size=12,
            n_embd=192,
            img_embd_dim=512,
        ),
    }
    if variant not in presets:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(presets.keys())}.")
    
    params = {**presets[variant], "patch_mode": patch_mode, **overrides}
    return NanoVLMConfig(**params)
