"""
src/models/patching.py

Découpage d'une séquence temporelle en patches, et projection linéaire
vers un espace d'embeddings.

Convention :
    Input  : (B, L) — B batch, L longueur de séquence
    Output : (B, N, D) — N patches, D dimension du modèle

Avec :
    P = patch_len, S = stride
    L_padded = max(L, P)  # on garantit au moins 1 patch
    N = floor((L_padded - P) / S) + 1 + 1
    Le dernier patch est obtenu par REPEAT-PAD : on duplique la dernière
    valeur de la séquence pour combler. Cette stratégie est celle du
    papier PatchTST original (Sec. 3.1).

Positional encoding :
    On utilise un PE appris (nn.Embedding) sur les N positions de patch.
    PE absolu suffit ici car N est petit (typiquement N <= 12 pour Exp A).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_n_patches(seq_len: int, patch_len: int, stride: int) -> int:
    """Calcule N pour donnée seq_len, patch_len, stride.

    On applique un repeat-pad final pour garantir que le dernier patch
    "déborde" sur des valeurs répliquées plutôt que tronquées.

    N = floor((L - P) / S) + 1   (patches qui rentrent entièrement)
        + 1                       (un patch supplémentaire qui couvre la fin
                                   avec repeat-pad)
    """
    if seq_len < patch_len:
        return 1  # un seul patch fait par repeat-pad complet
    n_full = (seq_len - patch_len) // stride + 1
    # On ajoute un patch supplémentaire couvrant les S derniers points
    # uniquement si nécessaire pour couvrir toute la séquence
    last_covered = (n_full - 1) * stride + patch_len
    if last_covered < seq_len:
        return n_full + 1
    return n_full


def make_patches(
    x: torch.Tensor,
    patch_len: int,
    stride: int,
) -> torch.Tensor:
    """Découpe (B, L) en (B, N, P) avec repeat-pad final si besoin.

    Args:
        x : (B, L)
        patch_len : longueur d'un patch P
        stride : stride S

    Returns:
        patches : (B, N, P)
    """
    B, L = x.shape
    N = compute_n_patches(L, patch_len, stride)

    # Étape 1 : repeat-pad si nécessaire
    last_covered = (N - 1) * stride + patch_len
    if last_covered > L:
        pad_len = last_covered - L
        # Réplique la dernière valeur le long de l'axe temps
        last_val = x[:, -1:].expand(B, pad_len)  # (B, pad_len)
        x_padded = torch.cat([x, last_val], dim=-1)  # (B, L + pad_len)
    else:
        x_padded = x

    # Étape 2 : extraction des patches via unfold
    # (B, L_padded) -> (B, N, P)
    patches = x_padded.unfold(dimension=-1, size=patch_len, step=stride)
    return patches


class PatchEmbedding(nn.Module):
    """Patching + projection linéaire + positional encoding.

    Forward:
        x : (B, L)
        return : (B, N, d_model)
    """

    def __init__(
        self,
        patch_len: int,
        stride: int,
        max_n_patches: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        # Projection linéaire P -> D
        self.projection = nn.Linear(patch_len, d_model)
        # Positional encoding appris (PE absolu)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L)
        patches = make_patches(x, self.patch_len, self.stride)  # (B, N, P)
        x_proj = self.projection(patches)                         # (B, N, D)
        N = x_proj.shape[1]
        if N > self.pos_embedding.shape[1]:
            raise RuntimeError(
                f"PatchEmbedding: got N={N} patches but max_n_patches="
                f"{self.pos_embedding.shape[1]}. Increase max_n_patches."
            )
        x_proj = x_proj + self.pos_embedding[:, :N, :]            # broadcast (1, N, D)
        return self.dropout(x_proj)
