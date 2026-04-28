"""
src/evaluation/loso.py

Cross-validation Leave-One-Subject-Out (LOSO) pour les 24 enfants.

À chaque fold :
    - 1 enfant en test (tous ses traits)
    - 23 enfants en train (tous leurs traits)

Le split est strict : aucun trait de l'enfant test n'apparaît dans le train.
C'est CRUCIAL pour éviter le data leakage (sinon le modèle apprend "le style
de l'enfant" plutôt que "TDAH vs CTRL").

Format de sortie d'un fold :
    LOSOFold(test_child_id, train_child_ids)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class LOSOFold:
    """Un fold LOSO."""
    fold_index: int
    test_child_id: str
    train_child_ids: list[str]

    @property
    def n_train_children(self) -> int:
        return len(self.train_child_ids)


def loso_splits(child_ids: list[str]) -> Iterator[LOSOFold]:
    """Itère sur les folds LOSO de manière déterministe.

    Args:
        child_ids: liste des identifiants enfants (déduplique et trie).

    Yields:
        LOSOFold pour chaque enfant à tour de rôle (ordre alphabétique).
    """
    unique_ids = sorted(set(child_ids))
    for i, test_id in enumerate(unique_ids):
        train_ids = [c for c in unique_ids if c != test_id]
        yield LOSOFold(
            fold_index=i,
            test_child_id=test_id,
            train_child_ids=train_ids,
        )


def n_folds(child_ids: list[str]) -> int:
    """Nombre de folds LOSO = nombre d'enfants uniques."""
    return len(set(child_ids))
