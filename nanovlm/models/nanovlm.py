"""Top-level NanoVLM model combining VisionEncoder, MultimodalProjector, and LanguageDecoder.

Matches Figure 1 (Root level architecture of VLM) of the NanoVLM paper.
Includes built-in parameter accounting across the three core modules to verify Table 1 & Table 2.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovlm.config import NanoVLMConfig
from nanovlm.models.vision_encoder import VisionEncoder
from nanovlm.models.connector import MultimodalProjector
from nanovlm.models.decoder import LanguageDecoder


class NanoVLM(nn.Module):
    """Complete Vision-Language Model from 'NanoVLMs: How small can we go...'."""

    def __init__(self, config: NanoVLMConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.connector = MultimodalProjector(config)
        self.decoder = LanguageDecoder(config)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of NanoVLM.

        Args:
            images: Input image batch of shape (B, 3, 224, 224).
            input_ids: Input partial/prompt token IDs of shape (B, T_text).
            targets: Optional target token IDs of shape (B, T_text) for training loss.

        Returns:
            (logits, loss)
        """
        vis_features = self.vision_encoder(images)       # (B, 197, img_embd_dim)
        vis_tokens = self.connector(vis_features)        # (B, N_vis, n_embd)
        logits, loss = self.decoder(vis_tokens, input_ids, targets=targets)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively complete partial text descriptions conditioned on an image (Section 3)."""
        self.eval()
        vis_features = self.vision_encoder(images)
        vis_tokens = self.connector(vis_features)

        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.config.max_seq_len:
                break

            logits, _ = self.decoder(vis_tokens, generated)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def parameter_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Compute parameter counts and percentages across the three core modules (Table 2)."""
        vis_params = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        patch_params = sum(
            p.numel() for p in self.vision_encoder.patch_embed.parameters() if p.requires_grad
        )
        conn_params = sum(p.numel() for p in self.connector.parameters() if p.requires_grad)
        dec_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total = vis_params + conn_params + dec_params

        return {
            "Patch Embedding (inside Visual Encoder)": {
                "params": patch_params,
                "millions": patch_params / 1e6,
                "pct": (patch_params / total) * 100.0 if total else 0.0,
            },
            "Visual Encoder (Total)": {
                "params": vis_params,
                "millions": vis_params / 1e6,
                "pct": (vis_params / total) * 100.0 if total else 0.0,
            },
            "Multimodal Projector": {
                "params": conn_params,
                "millions": conn_params / 1e6,
                "pct": (conn_params / total) * 100.0 if total else 0.0,
            },
            "Language Decoder": {
                "params": dec_params,
                "millions": dec_params / 1e6,
                "pct": (dec_params / total) * 100.0 if total else 0.0,
            },
            "Total": {
                "params": total,
                "millions": total / 1e6,
                "pct": 100.0,
            },
        }
