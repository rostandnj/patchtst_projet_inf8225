"""
src/training/inner_cv.py

Inner cross-validation : RepeatedStratifiedKFold pour les 23 enfants
disponibles dans chaque fold LOSO externe.

Génère 5 splits × 3 répétitions = 15 (train_inner, val_inner) splits.

Chaque split donne :
    train_inner_ids : ~18-19 enfants pour entraîner
    val_inner_ids   : ~4-5 enfants pour valider/early-stop

Stratification par label (CTRL/ADHD) pour avoir un ratio équilibré dans
le val set même quand le train du fold LOSO externe a 12-11 ou 11-12
selon l'enfant testé.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


@dataclass
class InnerSplit:
    """Un split de validation interne."""
    repeat_index: int          # 0..n_repeats-1
    fold_index: int            # 0..n_splits-1
    train_inner_ids: list[str]
    val_inner_ids: list[str]
    global_index: int          # repeat_index * n_splits + fold_index


def inner_cv_splits(
    train_pool_ids: list[str],
    labels_per_child: dict[str, int],
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> Iterator[InnerSplit]:
    """Itère sur les n_splits × n_repeats splits internes.

    Args:
        train_pool_ids : 23 enfants disponibles (le test LOSO est exclu).
        labels_per_child : dict pour stratification.
        n_splits : K dans le KFold (défaut 5)
        n_repeats : nombre de répétitions (défaut 3)
        seed : reproductibilité.

    Yields:
        InnerSplit avec les IDs de chaque côté.
    """
    ids = np.array(train_pool_ids)
    y = np.array([labels_per_child[c] for c in ids])

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    for global_idx, (train_idx, val_idx) in enumerate(rskf.split(ids, y)):
        repeat_index = global_idx // n_splits
        fold_index = global_idx % n_splits
        yield InnerSplit(
            repeat_index=repeat_index,
            fold_index=fold_index,
            train_inner_ids=ids[train_idx].tolist(),
            val_inner_ids=ids[val_idx].tolist(),
            global_index=global_idx,
        )


def n_inner_splits(n_splits: int = 5, n_repeats: int = 3) -> int:
    """Nombre total de splits internes."""
    return n_splits * n_repeats
