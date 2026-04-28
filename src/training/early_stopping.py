"""
src/training/early_stopping.py

Early stopping sur métrique de validation avec patience.

Permet de monitorer 'val_loss' (minimisation) ou 'val_accuracy' (maximisation).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    """État d'un mécanisme d'early stopping."""
    best_metric: float
    epochs_since_improvement: int = 0
    best_epoch: int = -1
    stopped: bool = False
    stop_reason: str = ""


class EarlyStopping:
    """Early stopping en surveillant une métrique.

    Args:
        patience: nombre d'epochs sans amélioration avant arrêt.
        min_delta: amélioration minimale considérée comme significative.
        mode: 'min' (val_loss) ou 'max' (val_accuracy).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        init_best = float("inf") if mode == "min" else float("-inf")
        self.state = EarlyStoppingState(best_metric=init_best)

    def step(self, current_metric: float, epoch: int) -> bool:
        """Met à jour l'état avec la métrique courante.

        Returns:
            True si on doit s'arrêter, False sinon.
        """
        if self.mode == "min":
            improved = current_metric < (self.state.best_metric - self.min_delta)
        else:
            improved = current_metric > (self.state.best_metric + self.min_delta)

        if improved:
            self.state.best_metric = current_metric
            self.state.best_epoch = epoch
            self.state.epochs_since_improvement = 0
        else:
            self.state.epochs_since_improvement += 1
            if self.state.epochs_since_improvement >= self.patience:
                self.state.stopped = True
                self.state.stop_reason = (
                    f"No improvement for {self.patience} epochs "
                    f"(best={self.state.best_metric:.5f} at epoch {self.state.best_epoch})"
                )

        return self.state.stopped

    def is_best(self, current_metric: float) -> bool:
        """Indique si current_metric est meilleur que le best stocké (sans modifier l'état).

        À utiliser pour décider si on sauvegarde un checkpoint.
        """
        if self.mode == "min":
            return current_metric < (self.state.best_metric - self.min_delta) or \
                   self.state.best_epoch == -1
        return current_metric > (self.state.best_metric + self.min_delta) or \
               self.state.best_epoch == -1
