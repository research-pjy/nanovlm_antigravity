"""Autoregressive Language Decoder for NanoVLM.

Implements Section 2.2.3 (Decoder Block) of the NanoVLM paper:
- Embeds input text tokens into `n_embd`
- Concatenates projected visual embeddings with textual embeddings to form a multimodal sequence
- Applies learnable positional embeddings encoding token order
- Processes through `n_layer` causal Transformer blocks (`n_head` heads of `head_size` each)
- Final LayerNorm + Linear projection to `vocab_size` logits
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovlm.config import NanoVLMConfig
from nanovlm.models.attention import TransformerBlock


class LanguageDecoder(nn.Module):
    """Causal Transformer Decoder operating on concatenated [Visual, Textual] tokens."""

    def __init__(self, config: NanoVLMConfig) -> None:
        super().__init__()
        self.config = config

        # Text token embedding table
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)

        # Total sequence capacity = visual tokens + max text tokens
        vis_len = 1 if config.projector_use_token_pooling else config.num_vis_tokens
        self.max_total_len = vis_len + config.max_seq_len

        self.pos_embed = nn.Embedding(self.max_total_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Causal Transformer Decoder Layers (`n_layer` = 4 for mini, 8 for base, 10 for large)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embd_dim=config.n_embd,
                n_head=config.n_head,
                head_size=config.head_size,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                is_causal=True,
            )
            for _ in range(config.n_layer)
        ])

        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        vis_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the multimodal language decoder.

        Args:
            vis_tokens: Projected visual tokens of shape (B, N_vis, n_embd).
            input_ids: Text token IDs of shape (B, T_text).
            targets: Optional target token IDs of shape (B, T_text) for cross-entropy loss.

        Returns:
            Tuple of (text_logits of shape (B, T_text, vocab_size), loss if targets provided).
        """
        B, N_vis, _ = vis_tokens.shape
        _, T_text = input_ids.shape
        total_len = N_vis + T_text
        assert total_len <= self.max_total_len, (
            f"Sequence length {total_len} exceeds maximum {self.max_total_len}"
        )

        # 1. Text embeddings (Section 2.2.2)
        text_embeds = self.token_embed(input_ids)  # (B, T_text, n_embd)

        # 2. Concatenate visual and textual embeddings into multimodal sequence
        multimodal_embeds = torch.cat([vis_tokens, text_embeds], dim=1)  # (B, N_vis + T_text, n_embd)

        # 3. Add positional embeddings (Section 2.2.3)
        positions = torch.arange(0, total_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.drop(multimodal_embeds + self.pos_embed(positions))

        # 4. Pass through causal transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # 5. Slice out the textual positions to predict next text tokens
        text_hidden = x[:, N_vis:, :]  # (B, T_text, n_embd)
        logits = self.lm_head(text_hidden)  # (B, T_text, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss
