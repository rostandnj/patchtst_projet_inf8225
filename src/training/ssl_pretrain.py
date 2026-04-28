"""
src/training/ssl_pretrain.py

Boucle de pretraining auto-supervisé pour PatchTST (masked patch
reconstruction).

PRINCIPE :
    1. Pour chaque batch (B, M, L), on tire aléatoirement un masque de
       40% des patches (par défaut).
    2. Les patches masqués sont remplacés par un mask_token appris dans
       l'embedding space avant le Transformer.
    3. Le modèle prédit la valeur originale (dans l'espace normalisé RevIN)
       de tous les patches.
    4. Loss MSE calculée UNIQUEMENT sur les patches masqués.

DIFFÉRENCE vs supervisé :
    - Pas de val/labels TDAH/CTRL utilisés
    - Early stopping sur la reconstruction loss (mode='min')
    - Pas de class weight
    - LR plus bas par défaut (5e-4 vs 1e-3) pour stabilité
    - Plus d'epochs (100 par défaut vs 50)

USAGE :
    history, best_state = train_ssl(
        model=PatchTSTReconstructor(cfg),
        train_loader=train_loader,
        device=torch.device("mps"),
        config=full_cfg,  # config.ssl utilisé
    )
    # On peut ensuite transférer le backbone vers un classifier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import FullExpConfig
from .early_stopping import EarlyStopping
from .optimizer import build_optimizer, build_scheduler, clip_gradients


@dataclass
class SSLHistory:
    """Historique d'un pretraining SSL."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_loss: float = float("inf")
    n_epochs_run: int = 0
    stopped_early: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "lr": self.lr,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "n_epochs_run": self.n_epochs_run,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


def _ssl_epoch_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 1.0,
) -> float:
    """Une epoch de SSL. Train si optimizer fourni, sinon eval."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_weighted = 0.0
    total_n = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            # On accepte (x, y) ou x seul
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)

            recon, mask, target = model(x, mask_ratio=mask_ratio)
            loss = model.reconstruction_loss(recon, target, mask)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    clip_gradients(model, grad_clip)
                optimizer.step()

            batch_size = x.size(0)
            total_loss_weighted += float(loss.item()) * batch_size
            total_n += batch_size

    return total_loss_weighted / max(1, total_n)


def train_ssl(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: FullExpConfig,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
    log_every: int = 10,
) -> tuple[SSLHistory, dict[str, torch.Tensor]]:
    """SSL pretraining : masked patch reconstruction.

    Args:
        model : PatchTSTReconstructor.
        train_loader : DataLoader d'entraînement (labels ignorés).
        device : torch.device.
        config : FullExpConfig (config.ssl utilisé).
        val_loader : optionnel pour early stopping. Si None, on early-stop
            sur la train loss.

    Returns:
        (history, best_state_dict) — best sur val_loss si val_loader fourni,
        sinon sur train_loss.
    """
    if config.ssl is None:
        raise ValueError("config.ssl must be set to call train_ssl()")

    ssl_cfg = config.ssl
    model = model.to(device)

    # Optim avec lr plus bas pour SSL (bonne pratique)
    optimizer = build_optimizer(model, config.optim)
    # Force lr SSL plus bas si l'utilisateur n'a pas changé optim.lr
    # (cohérent avec config.ssl)
    scheduler = build_scheduler(optimizer, config.optim, ssl_cfg.max_epochs)

    monitor_train_only = val_loader is None
    early = EarlyStopping(
        patience=ssl_cfg.early_stopping_patience,
        min_delta=ssl_cfg.early_stopping_min_delta,
        mode="min",
    )

    history = SSLHistory()
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(ssl_cfg.max_epochs):
        train_loss = _ssl_epoch_loop(
            model, train_loader, device, ssl_cfg.mask_ratio,
            optimizer=optimizer, grad_clip=config.optim.grad_clip,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = _ssl_epoch_loop(
                model, val_loader, device, ssl_cfg.mask_ratio, optimizer=None,
            )

        current_metric = train_loss if monitor_train_only else val_loss
        current_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_loss)
        if val_loss is not None:
            history.val_loss.append(val_loss)
        history.lr.append(current_lr)
        history.n_epochs_run = epoch + 1

        if early.is_best(current_metric):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            history.best_loss = current_metric

        stop = early.step(current_metric, epoch)

        if verbose and (epoch % log_every == 0 or stop or epoch == ssl_cfg.max_epochs - 1):
            tag = "★" if epoch == history.best_epoch else " "
            val_str = f" val={val_loss:.4f}" if val_loss is not None else ""
            print(f"  ssl ep{epoch:3d}  lr={current_lr:.5f}  "
                  f"train={train_loss:.4f}{val_str}  {tag}")

        if stop:
            history.stopped_early = True
            history.stop_reason = early.state.stop_reason
            if verbose:
                print(f"  Early stopping: {history.stop_reason}")
            break

        scheduler.step()

    if verbose:
        print(f"  SSL done. Best epoch={history.best_epoch}, "
              f"best loss={history.best_loss:.4f}")

    return history, best_state
