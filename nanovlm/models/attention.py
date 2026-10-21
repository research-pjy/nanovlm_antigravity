"""Shared Transformer building blocks (Multi-Head Attention and Pre-LN Transformer Block).

Matches Section 2.2 and Figure 5 of the NanoVLM paper:
- Pre-LayerNorm applied prior to Multi-Head Attention and MLP layers
- Optional causal self-attention masking (used in the Language Decoder)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional causal masking."""

    def __init__(
        self,
        embd_dim: int,
        n_head: int,
        head_size: Optional[int] = None,
        dropout: float = 0.1,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.head_size = head_size if head_size is not None else (embd_dim // n_head)
        self.inner_dim = self.n_head * self.head_size
        self.is_causal = is_causal
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embd_dim, self.inner_dim)
        self.k_proj = nn.Linear(embd_dim, self.inner_dim)
        self.v_proj = nn.Linear(embd_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, embd_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Multi-Head Attention.

        Args:
            x: Input tensor of shape (B, T, C).
            attention_mask: Optional boolean/additive mask of shape (B, 1, T, T) or (B, T).
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # Use PyTorch 2.0+ Flash/SDPA attention for speed on L40S / A100 GPUs
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_causal and (attention_mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """Two-layer Multi-Layer Perceptron (MLP) inside each Transformer block."""

    def __init__(
        self,
        embd_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = embd_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embd_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embd_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block as depicted in Figure 5 (right) of the NanoVLM paper."""

    def __init__(
        self,
        embd_dim: int,
        n_head: int,
        head_size: Optional[int] = None,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.attn = MultiHeadAttention(
            embd_dim=embd_dim,
            n_head=n_head,
            head_size=head_size,
            dropout=dropout,
            is_causal=is_causal,
        )
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = FeedForward(
            embd_dim=embd_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x
