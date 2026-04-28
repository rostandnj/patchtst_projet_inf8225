"""
src/models/transformer.py

Encoder Transformer (style pré-LayerNorm), utilisé comme backbone PatchTST.

Pré-norm : LayerNorm avant chaque sous-couche (self-attn et FFN). Plus
stable à entraîner sur petit n que post-norm. C'est le choix par défaut
dans la plupart des Transformers récents (et utilisé dans PatchTST).

On utilise nn.MultiheadAttention de PyTorch (batch_first=True) — pas de
mask causale ici (l'encodeur PatchTST est non-causal, on a accès à tous
les patches).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """FFN à deux couches : Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class TransformerEncoderLayer(nn.Module):
    """Une couche d'encodeur Transformer pré-norm.

    Forward:
        x : (B, N, D)
        return : (B, N, D)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # Pre-norm FFN
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack de n_layers d'encodeur Transformer pré-norm."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, attn_dropout=attn_dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)
