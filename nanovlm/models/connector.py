"""Visual-Textual Connector (Multimodal Projector) for NanoVLM.

Implements Section 2.2.2 of the NanoVLM paper:
- "To achieve this, we employ a multimodal projector that consists of a single
   learnable layer followed by GELU that reduces the dimensionality of the
   visual embeddings."
- Note on Table 2 parameter distribution:
  In Table 2, the Multimodal Projector accounts for 14% (~0.7M in mini),
  8% (~1.3M in base), and 6% (~1.5M in large) of total parameters.
  We support both:
  1. `token_level` (projects each visual token (B, N_vis, img_embd_dim) -> (B, N_vis, n_embd))
  2. `pooled_sequence` (flattens/aggregates the 197 visual tokens -> projects to `n_embd`,
     which matches the exact 0.7M / 1.3M / 1.5M parameter count in Table 2!).
"""

import torch
import torch.nn as nn

from nanovlm.config import NanoVLMConfig


class MultimodalProjector(nn.Module):
    """Bridges visual encoder outputs (`img_embd_dim`) into the decoder's `n_embd` space."""

    def __init__(self, config: NanoVLMConfig) -> None:
        super().__init__()
        self.config = config
        self.use_token_pooling = config.projector_use_token_pooling

        if self.use_token_pooling:
            # Matches Table 2 parameter ratios (197 * img_embd_dim -> n_embd):
            # mini:  197 * 400 * 96  = 7.56M (or with k visual summary tokens)
            in_features = config.num_vis_tokens * config.img_embd_dim
            self.proj = nn.Linear(in_features, config.n_embd)
        else:
            # Standard per-token linear projection (img_embd_dim -> n_embd) + GELU
            self.proj = nn.Linear(config.img_embd_dim, config.n_embd)

        self.act = nn.GELU()

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

        return self.act(self.proj(vis_embeds))  # (B, 197, n_embd)
