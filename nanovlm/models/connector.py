"""Visual-Textual Connector (Multimodal Projector) for NanoVLM.

Supports:
1. `linear_gelu` (Paper Section 2.2.2 baseline):
   Single learnable Linear(img_embd_dim, n_embd) followed by GELU.
2. `swiglu_token_mixer` (Phase 2 Novel Idea 3):
   - Applies a lightweight 3x3 depthwise 2D spatial convolution across the 14x14 patch tokens
     (preserving the [CLS] token) to refine local spatial coherence after the ViT blocks.
   - Projects into `n_embd` using a Gated Linear Unit (SwiGLU: SiLU(W_gate * x) * (W_val * x))
     followed by LayerNorm, filtering out noisy visual features before entering the small decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovlm.config import NanoVLMConfig


class MultimodalProjector(nn.Module):
    """Bridges visual encoder outputs (`img_embd_dim`) into the decoder's `n_embd` space."""

    def __init__(self, config: NanoVLMConfig) -> None:
        super().__init__()
        self.config = config
        self.use_token_pooling = config.projector_use_token_pooling
        self.connector_mode = getattr(config, "connector_mode", "linear_gelu")

        if self.use_token_pooling:
            in_features = config.num_vis_tokens * config.img_embd_dim
            self.proj = nn.Linear(in_features, config.n_embd)
            self.act = nn.GELU()
        elif self.connector_mode == "linear_gelu":
            # Paper baseline: single Linear projection + GELU
            self.proj = nn.Linear(config.img_embd_dim, config.n_embd)
            self.act = nn.GELU()
        elif self.connector_mode == "swiglu_token_mixer":
            # Idea 3: Spatial Depthwise Token Mixer + SwiGLU Gated Multimodal Projector
            dim_in = config.img_embd_dim
            dim_out = config.n_embd
            self.spatial_dw = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, groups=dim_in)
            self.pre_norm = nn.LayerNorm(dim_in)
            self.w_gate = nn.Linear(dim_in, dim_out)
            self.w_val = nn.Linear(dim_in, dim_out)
            self.out_norm = nn.LayerNorm(dim_out)
        else:
            raise ValueError(f"Unsupported connector_mode: {self.connector_mode}")

    def forward(self, vis_embeds: torch.Tensor) -> torch.Tensor:
        """Project visual embeddings to the textual embedding space.

        Args:
            vis_embeds: Tensor of shape (B, 197, img_embd_dim).

        Returns:
            Projected visual tokens of shape (B, N_vis_out, n_embd).
        """
        if self.use_token_pooling:
            B = vis_embeds.shape[0]
            flat = vis_embeds.reshape(B, 1, -1)
            return self.act(self.proj(flat))  # (B, 1, n_embd)

        if self.connector_mode == "linear_gelu":
            return self.act(self.proj(vis_embeds))  # (B, 197, n_embd)

        if self.connector_mode == "swiglu_token_mixer":
            B, N, D = vis_embeds.shape
            cls_tok = vis_embeds[:, :1, :]      # (B, 1, D)
            patch_toks = vis_embeds[:, 1:, :]   # (B, 196, D)
            grid_size = int(math.isqrt(N - 1))  # 14

            # 2D spatial depthwise residual mixing across the 14x14 visual tokens
            spatial = patch_toks.transpose(1, 2).contiguous().view(B, D, grid_size, grid_size)
            spatial = spatial + self.spatial_dw(spatial)
            patch_mixed = spatial.flatten(2).transpose(1, 2).contiguous()

            x = torch.cat([cls_tok, patch_mixed], dim=1)  # (B, 197, D)
            x = self.pre_norm(x)
            # SwiGLU gated projection
            gated = F.silu(self.w_gate(x)) * self.w_val(x)
            return self.out_norm(gated)

        raise RuntimeError("Unreachable")
