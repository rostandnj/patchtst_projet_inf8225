"""
src/training/supervised.py

Boucle d'entraînement supervisée pour PatchTST classification.

API principale : train_supervised()
    Entraîne un PatchTSTClassifier avec :
        - AdamW + cosine warmup
        - Early stopping sur val_loss
        - Best checkpoint sur val_loss
        - Class weight balanced (en option)
        - Gradient clipping

Retourne : history (dict) + best_state_dict (à recharger pour inférence)

Usage typique (1 inner split) :
    history, best_state = train_supervised(
        model=PatchTSTClassifier(cfg, n_classes=2),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("mps"),
        config=full_cfg,
    )
    model.load_state_dict(best_state)
    # -> on peut maintenant prédire sur le test set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import FullExpConfig
from .device import safe_int_to_float
from .early_stopping import EarlyStopping
from .optimizer import build_optimizer, build_scheduler, clip_gradients


@dataclass
class TrainHistory:
    """Historique d'un entraînement complet."""
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    n_epochs_run: int = 0
    stopped_early: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "lr": self.lr,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "n_epochs_run": self.n_epochs_run,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


def _compute_class_weights(
    train_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Calcule les poids inverses de classe pour CrossEntropyLoss."""
    n_per_class = torch.zeros(2, dtype=torch.float64)
    for _, y in train_loader:
        for c in (0, 1):
            n_per_class[c] += (y == c).sum().item()
    total = n_per_class.sum().clamp(min=1.0)
    # weight = total / (n_classes * count_class)
    weights = total / (2.0 * n_per_class.clamp(min=1.0))
    return weights.float().to(device)


def _epoch_loop(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """Une epoch complète. Train si optimizer fourni, sinon eval.

    Returns:
        (avg_loss, accuracy)
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_weighted = 0.0
    total_n = 0
    total_correct = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)                                   # (B, 2)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    clip_gradients(model, grad_clip)
                optimizer.step()

            batch_size = y.size(0)
            total_loss_weighted += float(loss.item()) * batch_size
            total_n += batch_size
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == y).sum().item())

    avg_loss = total_loss_weighted / max(1, total_n)
    accuracy = total_correct / max(1, total_n)
    return avg_loss, accuracy


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: FullExpConfig,
    verbose: bool = True,
    log_every: int = 5,
) -> tuple[TrainHistory, dict[str, torch.Tensor]]:
    """Entraîne PatchTST classification avec early stopping sur val_loss.

    Returns:
        (history, best_state_dict)
        Le best_state_dict correspond au modèle à l'epoch avec le meilleur
        val_loss (ou val_accuracy selon config.training.monitor).
    """
    model = model.to(device)
    optimizer = build_optimizer(model, config.optim)
    scheduler = build_scheduler(optimizer, config.optim, config.training.max_epochs)

    # Class weights sur le train set
    if config.training.use_class_weight:
        class_weights = _compute_class_weights(train_loader, device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    monitor = config.training.monitor
    es_mode = "min" if monitor == "val_loss" else "max"
    early = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta,
        mode=es_mode,
    )

    history = TrainHistory()
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(config.training.max_epochs):
        # Train
        train_loss, train_acc = _epoch_loop(
            model, train_loader, device, criterion,
            optimizer=optimizer, grad_clip=config.optim.grad_clip,
        )
        # Val
        val_loss, val_acc = _epoch_loop(
            model, val_loader, device, criterion, optimizer=None,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.lr.append(current_lr)
        history.n_epochs_run = epoch + 1

        # Tracking du best
        current_metric = val_loss if monitor == "val_loss" else val_acc
        if early.is_best(current_metric):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            history.best_val_loss = val_loss
            history.best_val_acc = val_acc

        # Early stopping
        stop = early.step(current_metric, epoch)

        if verbose and (epoch % log_every == 0 or stop or epoch == config.training.max_epochs - 1):
            print(f"  ep{epoch:3d}  lr={current_lr:.5f}  "
                  f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}  "
                  f"{'★' if epoch == history.best_epoch else ''}")

        if stop:
            history.stopped_early = True
            history.stop_reason = early.state.stop_reason
            if verbose:
                print(f"  Early stopping: {history.stop_reason}")
            break

        # Step scheduler à la fin de l'epoch
        scheduler.step()

    if verbose:
        print(f"  Done. Best epoch={history.best_epoch}, "
              f"best val_loss={history.best_val_loss:.4f}, "
              f"best val_acc={history.best_val_acc:.3f}")

    return history, best_state


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Inférence : retourne P(class=1) pour chaque sample du loader.

    Returns:
        np.ndarray shape (n_samples,) avec les probas P(ADHD=1).
    """
    model.eval()
    all_probas = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probas = F.softmax(logits, dim=-1)[:, 1]   # P(class=1)
        all_probas.append(probas.detach().cpu().numpy())
    return np.concatenate(all_probas, axis=0)
