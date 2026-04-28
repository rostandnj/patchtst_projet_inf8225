"""
src/training/config.py

Dataclasses de configuration pour l'entraînement supervisé et SSL.
Sérialisable en JSON pour reproductibilité (sauvegardé avec chaque run).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class OptimConfig:
    """Configuration de l'optimiseur et du scheduler."""
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 1.0
    # Scheduler cosine warmup
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    """Hyperparamètres d'entraînement supervisé."""
    max_epochs: int = 50
    batch_size: int = 19          # = 19 enfants train si séquence
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    monitor: str = "val_loss"     # 'val_loss' ou 'val_accuracy'
    use_class_weight: bool = True
    seed: int = 42

    # Trial subsampling stratifié (pour A.1/A.3 séquence)
    use_trial_subsampling: bool = True
    subsample_strategy: str = "stratified"  # 'stratified' (début/milieu/fin)
                                            # ou 'random' (uniforme)
                                            # ou 'none' (offset fixe)


@dataclass
class SSLConfig:
    """Hyperparamètres du pretraining SSL (masked patch reconstruction)."""
    max_epochs: int = 100
    batch_size: int = 19
    mask_ratio: float = 0.4
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    seed: int = 42

    # Trial subsampling pour SSL également
    use_trial_subsampling: bool = True
    subsample_strategy: str = "stratified"


@dataclass
class FullExpConfig:
    """Config complète d'une expérience (modèle + train + optim)."""
    # Identification
    exp_name: str = "A1_seq_supervised"
    description: str = ""

    # Optim
    optim: OptimConfig = field(default_factory=OptimConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # SSL (utilisé uniquement pour A.3, A.4)
    ssl: Optional[SSLConfig] = None

    # PatchTST hyperparams (les autres viennent du modèle directement)
    patch_len: int = 4
    stride: int = 2
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.2
    attn_dropout: float = 0.1

    # Données
    seq_len: int = 20             # L pour Exp A.1/A.3 ; 10 pour A.2/A.4
    n_channels: int = 14

    # Inner CV
    inner_n_splits: int = 5
    inner_n_repeats: int = 3
    inner_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
