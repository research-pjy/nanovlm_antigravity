"""Visual Encoder and multi-mode PatchEmbedding for NanoVLM.

Implements Section 2.2.1 (Visual Encoder), Figure 4, and Figure 5 of the NanoVLM paper,
with clean support for comparing the four patch extraction & convolution orderings:

1. `patch_then_conv` (Paper Recreation - Patch First, then Conv per patch):
   - Image (B, 3, 224, 224) is divided into 196 patches of (3, 16, 16).
   - Each patch passes through 2x 2D Conv layers + LayerNorm + ReLU.
   - Each patch's feature map (C_out, 16, 16) is flattened (4096-dim) and passed
     through a Fully Connected (Linear) layer -> `img_embd_dim` (196 tokens).

2. `conv_then_patch_grid` (Conv on full image first, keeping resolution, then slice into 196 grids):
   - Full image (B, 3, 224, 224) passes through 2x 2D Conv layers + LayerNorm + ReLU -> (B, C_out, 224, 224).
   - Feature map is partitioned into 14x14 = 196 spatial patches of (C_out, 16, 16).
   - Flattened (4096-dim) and passed through a Linear layer -> `img_embd_dim` (196 tokens).
   - Same parameter count as `patch_then_conv`, but convolutions share receptive fields across patch borders!

3. `conv_stem_strided` (Conv first with spatial downsampling to 14x14 grid, then flatten into 196 tokens):
   - Full image (B, 3, 224, 224) passes through 2 strided Conv2d layers (stride 4 each -> 14x14 spatial map).
   - Each of the 14x14 = 196 spatial cells is already a patch vector of size C_out!
   - Linear layer only maps `C_out -> img_embd_dim` (eliminates ~1.6M-2.1M parameters!).

4. `standard_vit` (Standard ViT single Conv2d baseline):
   - Single Conv2d(3, img_embd_dim, kernel_size=16, stride=16).
"""

import torch
import torch.nn as nn

from nanovlm.config import NanoVLMConfig
from nanovlm.models.attention import TransformerBlock


class PatchEmbedding(nn.Module):
    """Configurable Patch + Convolution Embedding module to test architectural hypotheses."""

    def __init__(self, config: NanoVLMConfig) -> None:
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.img_embd_dim = config.img_embd_dim
        self.patch_mode = config.patch_mode

        c1 = config.conv_hidden_channels  # default: 8
        c2 = config.conv_out_channels     # default: 16

        if self.patch_mode in ("patch_then_conv", "conv_then_patch_grid"):
            # Two 2D convolutional layers (Figure 4) preserving 16x16 per-patch resolution
            self.conv1 = nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
            self.act = nn.ReLU()

            # Flattened size per 16x16 patch = c2 * 16 * 16 (e.g. 16 * 16 * 16 = 4096)
            flat_dim = c2 * self.patch_size * self.patch_size
            self.norm = nn.LayerNorm(flat_dim)
            self.fc = nn.Linear(flat_dim, self.img_embd_dim)

        elif self.patch_mode == "conv_stem_strided":
            # Two strided 2D convolutions downsampling 224x224 -> 56x56 -> 14x14 (196 spatial locations)
            self.conv1 = nn.Conv2d(self.in_channels, c1, kernel_size=4, stride=4, padding=0)
            self.conv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=4, padding=0)
            self.act = nn.ReLU()

            # Each of the 14x14 = 196 locations has dimension `c2` (16)
            self.norm = nn.LayerNorm(c2)
            self.fc = nn.Linear(c2, self.img_embd_dim)

        elif self.patch_mode == "standard_vit":
            # Classic Dosovitskiy et al. ViT patch projection
            self.proj = nn.Conv2d(
                self.in_channels,
                self.img_embd_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            self.norm = nn.LayerNorm(self.img_embd_dim)

        elif self.patch_mode == "overlapping_conv_stem":
            # Idea 1: Overlapping Multi-Stage Conv Stem (solves the 16-d information bottleneck
            # of conv_stem_strided while still saving ~1.6M-2.0M parameters!).
            # Uses overlapping 7x7 (stride 4) -> 3x3 (stride 3/4) convolutions with richer channels
            # (3 -> 32 -> 96) so neighboring patches share spatial context AND each token gets 96 channels.
            stem_mid = 32
            stem_out = 96
            self.conv1 = nn.Conv2d(self.in_channels, stem_mid, kernel_size=7, stride=4, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_mid)
            self.conv2 = nn.Conv2d(stem_mid, stem_out, kernel_size=5, stride=4, padding=2, bias=False)
            self.bn2 = nn.BatchNorm2d(stem_out)
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(stem_out)
            self.fc = nn.Linear(stem_out, self.img_embd_dim)

        elif self.patch_mode == "convnext_stem":
            # Idea 2: Depthwise-Separable ConvNeXt-style Vision Stem
            # Stage 1: Overlapping downsampling stem to 14x14 spatial grid (64 channels)
            # Stage 2: 7x7 Depthwise Conv across the 14x14 grid (captures wide 7x7 patch neighborhood!)
            #          + Inverted Pointwise MLP (64 -> 192 -> 64) with residual connection
            stem_dim = 64
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=7, stride=4, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, stem_dim, kernel_size=5, stride=4, padding=2, bias=False),
                nn.BatchNorm2d(stem_dim),
                nn.GELU(),
            )
            # ConvNeXt-style 7x7 depthwise mixing across the 14x14 patch tokens
            self.dwconv = nn.Conv2d(stem_dim, stem_dim, kernel_size=7, stride=1, padding=3, groups=stem_dim)
            self.norm1 = nn.LayerNorm(stem_dim)
            self.pwconv1 = nn.Linear(stem_dim, stem_dim * 3)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(stem_dim * 3, stem_dim)
            self.norm2 = nn.LayerNorm(stem_dim)
            self.fc = nn.Linear(stem_dim, self.img_embd_dim)

        else:
            raise ValueError(f"Unsupported patch_mode: {self.patch_mode}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Transform input images into 196 patch embeddings.

        Args:
            images: Tensor of shape (B, 3, 224, 224).

        Returns:
            Patch tokens of shape (B, 196, img_embd_dim).
        """
        B, C, H, W = images.shape
        P = self.patch_size
        assert H == self.image_size and W == self.image_size, (
            f"Expected image size ({self.image_size}, {self.image_size}), got ({H}, {W})"
        )

        if self.patch_mode == "patch_then_conv":
            # Step 1: Divide image into 196 non-overlapping 16x16 patches FIRST
            # (B, C, nH*P, nW*P) -> (B * 196, C, 16, 16)
            nh, nw = H // P, W // P
            patches = (
                images.view(B, C, nh, P, nw, P)
                .permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(B * nh * nw, C, P, P)
            )
            # Step 2: Apply two 2D convolutions independently to each patch
            feat = self.act(self.conv1(patches))
            feat = self.act(self.conv2(feat))  # (B * 196, c2, 16, 16)

            # Step 3: Flatten per-patch features, normalize, and project via FC layer
            feat_flat = feat.view(B, nh * nw, -1)
            tokens = self.fc(self.act(self.norm(feat_flat)))  # (B, 196, img_embd_dim)
            return tokens

        elif self.patch_mode == "conv_then_patch_grid":
            # Step 1: Apply two 2D convolutions to the FULL 224x224 image FIRST
            feat = self.act(self.conv1(images))
            feat = self.act(self.conv2(feat))  # (B, c2, 224, 224)

            # Step 2: Slice the resulting feature map into 196 patches of (c2, 16, 16)
            c2 = feat.size(1)
            nh, nw = H // P, W // P
            feat_flat = (
                feat.view(B, c2, nh, P, nw, P)
                .permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(B, nh * nw, c2 * P * P)
            )
            # Step 3: Normalize and project via FC layer
            tokens = self.fc(self.act(self.norm(feat_flat)))  # (B, 196, img_embd_dim)
            return tokens

        elif self.patch_mode == "conv_stem_strided":
            # Step 1: Strided 2D convolutions downsample (B, 3, 224, 224) -> (B, c2, 14, 14)
            feat = self.act(self.conv1(images))
            feat = self.act(self.conv2(feat))  # (B, c2, 14, 14)

            # Step 2: Rearrange 14x14 spatial grid into 196 tokens of dim c2
            feat_tokens = feat.flatten(2).transpose(1, 2).contiguous()  # (B, 196, c2)
            tokens = self.fc(self.act(self.norm(feat_tokens)))          # (B, 196, img_embd_dim)
            return tokens

        elif self.patch_mode == "standard_vit":
            feat = self.proj(images)  # (B, img_embd_dim, 14, 14)
            tokens = feat.flatten(2).transpose(1, 2).contiguous()
            return self.norm(tokens)

        elif self.patch_mode == "overlapping_conv_stem":
            feat = self.act(self.bn1(self.conv1(images)))  # (B, 32, 56, 56)
            feat = self.act(self.bn2(self.conv2(feat)))    # (B, 96, 14, 14)
            feat_tokens = feat.flatten(2).transpose(1, 2).contiguous()  # (B, 196, 96)
            tokens = self.fc(self.norm(feat_tokens))                    # (B, 196, img_embd_dim)
            return tokens

        elif self.patch_mode == "convnext_stem":
            feat = self.stem(images)                       # (B, 64, 14, 14)
            residual = feat
            feat = self.dwconv(feat)                       # (B, 64, 14, 14)
            feat_tokens = feat.flatten(2).transpose(1, 2).contiguous()  # (B, 196, 64)
            feat_tokens = self.pwconv2(self.act(self.pwconv1(self.norm1(feat_tokens))))
            res_tokens = residual.flatten(2).transpose(1, 2).contiguous()
            feat_tokens = res_tokens + feat_tokens         # Residual connection
            tokens = self.fc(self.norm2(feat_tokens))      # (B, 196, img_embd_dim)
            return tokens

        raise RuntimeError("Unreachable")


class VisionEncoder(nn.Module):
    """NanoVLM Visual Encoder (Section 2.2.1 & Figure 5).

    Pipeline:
    1. PatchEmbedding -> 196 tokens of `img_embd_dim`
    2. Prepend learnable `[CLS]` token -> 197 tokens
    3. Add learnable Positional Encoding + LayerNorm + Dropout
    4. Pass through `n_blks` Vision Transformer blocks (bidirectional self-attention)
    5. Final LayerNorm
    """

    def __init__(self, config: NanoVLMConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        # Learnable [CLS] token (Section 2.2.1: "A [CLS] token is then prepended, making the sequence 197 tokens")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.img_embd_dim))

        # Positional encoding for 197 tokens followed by normalization
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_vis_tokens, config.img_embd_dim)
        )
        self.pos_norm = nn.LayerNorm(config.img_embd_dim)
        self.pos_drop = nn.Dropout(config.dropout)

        # Transformer Encoder blocks (`n_blks` = 1 for mini, 3 for base, 5 for large)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embd_dim=config.img_embd_dim,
                n_head=config.vis_n_head,
                head_size=config.img_embd_dim // config.vis_n_head,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                is_causal=False,
            )
            for _ in range(config.n_blks)
        ])
        self.final_norm = nn.LayerNorm(config.img_embd_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual token representations.

        Args:
            images: Tensor of shape (B, 3, 224, 224).

        Returns:
            Visual embeddings of shape (B, 197, img_embd_dim).
        """
        B = images.shape[0]
        x = self.patch_embed(images)  # (B, 196, img_embd_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, img_embd_dim)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 197, img_embd_dim)

        x = self.pos_drop(self.pos_norm(x + self.pos_embed))

        for blk in self.blocks:
            x = blk(x)

        x = self.final_norm(x)
        return x
