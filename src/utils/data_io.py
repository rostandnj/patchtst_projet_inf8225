"""
src/utils/data_io.py

Chargement du dataset déjà préprocessé (params.npz, traces.npz, metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ProcessedDataset:
    """Dataset complet déjà préprocessé.

    Attributs :
        child_ids: liste des 24 IDs (ordonnée).
        labels_per_child: dict[child_id -> 0/1].
        params_per_child: dict[child_id -> matrix (n_strokes_i, 14)].
        trial_ids_per_child: dict[child_id -> array of trial_ids].
        param_names: liste des 14 noms.
        metadata: DataFrame des métadonnées (un row par trait).
    """
    child_ids: list[str]
    labels_per_child: dict[str, int]
    params_per_child: dict[str, np.ndarray]
    trial_ids_per_child: dict[str, np.ndarray]
    param_names: list[str]
    metadata: pd.DataFrame

    def n_strokes(self, child_id: str) -> int:
        return self.params_per_child[child_id].shape[0]

    @property
    def total_strokes(self) -> int:
        return sum(self.n_strokes(c) for c in self.child_ids)


def load_processed_dataset(processed_dir: Path) -> ProcessedDataset:
    """Charge tout depuis data/processed/."""
    processed_dir = Path(processed_dir)
    npz = np.load(processed_dir / "params.npz", allow_pickle=False)

    child_ids = [str(c) for c in npz["child_ids"]]
    labels = npz["labels"]
    param_names = [str(n) for n in npz["param_names"]]

    labels_per_child = {cid: int(lbl) for cid, lbl in zip(child_ids, labels)}
    params_per_child = {cid: npz[f"{cid}_params"] for cid in child_ids}
    trial_ids_per_child = {cid: npz[f"{cid}_trial_ids"] for cid in child_ids}

    metadata = pd.read_parquet(processed_dir / "metadata.parquet")

    return ProcessedDataset(
        child_ids=child_ids,
        labels_per_child=labels_per_child,
        params_per_child=params_per_child,
        trial_ids_per_child=trial_ids_per_child,
        param_names=param_names,
        metadata=metadata,
    )


def stack_strokes(
    dataset: ProcessedDataset,
    child_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empile les traits d'un sous-ensemble d'enfants en matrices.

    Returns:
        X: (total_strokes, 14) concaténation de tous les traits.
        y: (total_strokes,) labels par trait.
        ids: (total_strokes,) child_id par trait.
    """
    X_list, y_list, id_list = [], [], []
    for cid in child_ids:
        params = dataset.params_per_child[cid]
        n = params.shape[0]
        X_list.append(params)
        y_list.append(np.full(n, dataset.labels_per_child[cid], dtype=int))
        id_list.append(np.full(n, cid, dtype=object))
    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(id_list, axis=0),
    )
