"""
src/models/revin.py

Reversible Instance Normalization (Kim et al., 2022).

Normalisation par INSTANCE (chaque échantillon × chaque canal indépendamment).
À la différence de BatchNorm/LayerNorm, RevIN est PARAMÉTRÉE PAR FORWARD :
    - mode='norm'   : normalise et MÉMORISE les stats (mean, std).
    - mode='denorm' : applique la transformation inverse avec les stats mémorisées.

Pour PatchTST classification, on n'a pas besoin de denorm (la sortie est un
logit, pas une série temporelle). Mais on garde l'API complète pour la
phase 2 du mémoire (forecasting / drift detection).

Affine=True ajoute des paramètres apprenables (gamma, beta) par canal.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization sur le dernier axe (temps).

    Forward signature:
        x : tenseur (B, M, L) — B batch, M canaux, L temps
        mode : 'norm' ou 'denorm'

    Lorsque mode='norm', stocke les stats par (batch, canal) dans self.mean
    et self.stdev pour utilisation ultérieure en mode 'denorm'.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # Stats mémorisées (set par forward 'norm')
        self.register_buffer("mean", None, persistent=False)
        self.register_buffer("stdev", None, persistent=False)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"RevIN: mode must be 'norm' or 'denorm', got {mode!r}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, M, L)
        # Stats sur l'axe temps, par (batch, canal)
        mean = x.mean(dim=-1, keepdim=True).detach()           # (B, M, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False).detach()
        stdev = torch.sqrt(var + self.eps)
        self.mean = mean
        self.stdev = stdev
        x_norm = (x - mean) / stdev
        if self.affine:
            # Broadcast (M,) sur (B, M, L)
            x_norm = x_norm * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)
        return x_norm

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.stdev is None:
            raise RuntimeError(
                "RevIN: must call forward(mode='norm') before forward(mode='denorm')"
            )
        if self.affine:
            x = (x - self.affine_bias.view(1, -1, 1)) / (
                self.affine_weight.view(1, -1, 1) + self.eps
            )
        x = x * self.stdev + self.mean
        return x
