"""
src/training/checkpoint.py

Sauvegarde et chargement de checkpoints PyTorch.

Format : un dict avec model_state_dict, optimizer_state_dict, epoch, métriques.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Sauvegarde un checkpoint complet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Any = None,
) -> dict:
    """Charge un checkpoint dans le model (et optimizer si fourni).

    Returns:
        Le payload complet (utile pour récupérer epoch, metrics, extra).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
