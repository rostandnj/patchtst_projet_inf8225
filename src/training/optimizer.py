"""
src/training/optimizer.py

Factory pour AdamW + scheduler cosine warmup.

Le scheduler fonctionne par EPOCH (pas par batch), parce que sur petit n
on n'a que quelques batches par epoch et le warmup au niveau epoch est
plus stable.

Schéma de LR :
    epochs 0..warmup_epochs-1 : linear ramp de 0 à lr_max
    epochs warmup_epochs..max_epochs-1 : cosine decay de lr_max à min_lr
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """Construit AdamW avec les hyperparams de cfg.

    On exclut les biais et LayerNorm/RevIN params du weight_decay (best practice
    pour les Transformers).
    """
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Pas de decay sur biais, normalisations, et embeddings positionnels
        if (p.ndim == 1 or
            "bias" in name or
            "norm" in name.lower() or
            "pos_embedding" in name or
            "mask_token" in name or
            "affine_weight" in name or
            "affine_bias" in name):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimConfig,
    total_epochs: int,
):
    """Cosine warmup scheduler à appliquer après chaque epoch.

    Utilise LambdaLR pour une formulation simple :
        lr_factor(epoch) = epoch / warmup    if epoch < warmup
                         = 0.5 * (1 + cos(pi * progress))   sinon
                           où progress = (epoch - warmup) / (total - warmup)

    Le min_lr est atteint asymptotiquement via floor sur le facteur.
    """
    warmup = cfg.warmup_epochs
    min_factor = cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            # Linear warmup ; +1 pour démarrer >0 à epoch=0
            return float(epoch + 1) / float(max(1, warmup))
        # Cosine decay
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_factor, cosine_factor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip les gradients in-place et retourne la norme avant clipping."""
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
