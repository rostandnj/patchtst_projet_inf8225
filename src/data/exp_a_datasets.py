"""
src/data/exp_a_datasets.py

Datasets PyTorch pour Exp A : classification TDAH/CTRL à partir des
séquences inter-traits de paramètres sigma-lognormaux.

DEUX FORMATS :

    SequenceChildDataset (Exp A.1, A.3 — séquence complète)
        - Un échantillon = un enfant
        - Input  : (M=14, L_target) après padding/truncation
        - Label  : 0/1
        - **Trial subsampling** : pour augmenter virtuellement le nombre
          d'échantillons d'entraînement, on tire à chaque __getitem__ une
          sous-séquence consécutive de longueur L_target parmi les N traits
          de l'enfant. Trois stratégies :
            - 'none' : offset fixe = 0 (toujours les L premiers)
            - 'random' : offset uniformément aléatoire dans [0, N - L]
            - 'stratified' : offset stratifié en 3 zones (début/milieu/fin)
          Subsampling activé uniquement quand is_train=True.

    SlidingWindowDataset (Exp A.2, A.4 — fenêtres glissantes)
        - Pas besoin de subsampling : chaque fenêtre de L=10 traits est déjà
          un échantillon distinct, le dataset est déjà augmenté par construction.

NORMALISATION :
    On NE NORMALISE PAS dans le Dataset (RevIN s'en charge à l'intérieur
    du modèle PatchTST). Les paramètres sigma-lognormaux sont fournis
    bruts dans leurs unités natives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data_io import ProcessedDataset


# Pour Exp A on travaille sur 14 paramètres
DEFAULT_N_PARAMS = 14

# Stratégies de subsampling
SUBSAMPLING_STRATEGIES = ("none", "random", "stratified")


# ---------------------------------------------------------------------------
# Helpers de subsampling
# ---------------------------------------------------------------------------


def _stratified_offset(
    n_strokes: int,
    seq_len: int,
    rng: np.random.Generator,
) -> int:
    """Tire un offset stratifié en 3 zones (début/milieu/fin).

    Si n_strokes <= seq_len : retourne 0 (pas de marge).

    Sinon, max_offset = n_strokes - seq_len. On découpe [0, max_offset] en
    3 intervalles de tailles égales et on tire uniformément dans l'un des
    trois (choisi avec proba 1/3).
    """
    if n_strokes <= seq_len:
        return 0
    max_offset = n_strokes - seq_len  # inclusif
    # Trois zones : [0, max_offset/3], [max_offset/3, 2*max_offset/3],
    # [2*max_offset/3, max_offset]
    zone = rng.integers(0, 3)  # 0, 1, ou 2
    z_start = (zone * max_offset) // 3
    z_end = ((zone + 1) * max_offset) // 3
    if z_end < z_start:
        z_end = z_start
    # Tire uniformément dans [z_start, z_end] inclus
    if z_end == z_start:
        return int(z_start)
    return int(rng.integers(z_start, z_end + 1))


def _random_offset(
    n_strokes: int,
    seq_len: int,
    rng: np.random.Generator,
) -> int:
    """Tire un offset uniformément aléatoire dans [0, max_offset]."""
    if n_strokes <= seq_len:
        return 0
    max_offset = n_strokes - seq_len
    return int(rng.integers(0, max_offset + 1))


# ---------------------------------------------------------------------------
# Sequence-level dataset (Exp A.1, A.3)
# ---------------------------------------------------------------------------


@dataclass
class SequenceItem:
    """Un échantillon retourné par SequenceChildDataset (debug only)."""
    x: torch.Tensor                # (M, L_target)
    y: int                         # label 0/1
    child_id: str
    n_strokes_actual: int          # nombre de traits réels avant padding
    pad_mask: torch.Tensor         # (L_target,) bool, True = position paddée
    offset_used: int               # offset effectivement utilisé


class SequenceChildDataset(Dataset):
    """Un échantillon = un enfant (séquence ordonnée de traits).

    Args:
        dataset : ProcessedDataset chargé.
        child_ids : sous-ensemble d'enfants à utiliser.
        seq_len : longueur cible L de la séquence (ex: 20).
        is_train : si True, applique le trial subsampling à chaque
            __getitem__. Si False, utilise un offset déterministe (0).
        subsample_strategy : 'none', 'random', ou 'stratified'.
        pad_strategy : 'zero' (zero-pad), 'edge', 'mean'. Utilisé seulement
            quand n_strokes < seq_len.
        seed : seed pour le RNG du subsampling. Différent par dataset (train
            vs val) pour ne pas avoir des choix corrélés.
        return_dict : si True, retourne dict complet (debug).
    """

    def __init__(
        self,
        dataset: ProcessedDataset,
        child_ids: list[str],
        seq_len: int = 20,
        is_train: bool = True,
        subsample_strategy: str = "stratified",
        pad_strategy: str = "zero",
        seed: int = 0,
        return_dict: bool = False,
    ):
        if subsample_strategy not in SUBSAMPLING_STRATEGIES:
            raise ValueError(
                f"subsample_strategy must be one of {SUBSAMPLING_STRATEGIES}, "
                f"got {subsample_strategy!r}"
            )
        if pad_strategy not in ("zero", "edge", "mean"):
            raise ValueError("pad_strategy must be 'zero', 'edge' or 'mean'")

        self.dataset = dataset
        self.child_ids = list(child_ids)
        self.seq_len = seq_len
        self.is_train = is_train
        self.subsample_strategy = subsample_strategy
        self.pad_strategy = pad_strategy
        self.return_dict = return_dict
        self._rng = np.random.default_rng(seed)

    @property
    def n_channels(self) -> int:
        return self.dataset.params_per_child[self.child_ids[0]].shape[1]

    def _pick_offset(self, n_strokes: int) -> int:
        """Détermine l'offset selon le mode train/eval et la stratégie."""
        if not self.is_train or self.subsample_strategy == "none":
            return 0
        if self.subsample_strategy == "random":
            return _random_offset(n_strokes, self.seq_len, self._rng)
        if self.subsample_strategy == "stratified":
            return _stratified_offset(n_strokes, self.seq_len, self._rng)
        return 0

    def _build_item(self, cid: str) -> SequenceItem:
        params_full = self.dataset.params_per_child[cid]   # (n_strokes, M)
        n_actual = params_full.shape[0]
        M = params_full.shape[1]

        offset = self._pick_offset(n_actual)

        if n_actual >= self.seq_len:
            params = params_full[offset : offset + self.seq_len]
            n_kept = self.seq_len
            pad_mask = np.zeros(self.seq_len, dtype=bool)
        else:
            # Padding nécessaire (offset doit être 0)
            n_pad = self.seq_len - n_actual
            if self.pad_strategy == "zero":
                pad_block = np.zeros((n_pad, M), dtype=params_full.dtype)
            elif self.pad_strategy == "edge":
                pad_block = np.tile(params_full[-1:], (n_pad, 1))
            else:  # mean
                pad_block = np.tile(params_full.mean(axis=0, keepdims=True), (n_pad, 1))
            params = np.concatenate([params_full, pad_block], axis=0)
            n_kept = n_actual
            pad_mask = np.zeros(self.seq_len, dtype=bool)
            pad_mask[n_actual:] = True

        # (L, M) -> (M, L) pour PatchTST (channel-first)
        x = torch.from_numpy(np.ascontiguousarray(params.T)).float()
        y = self.dataset.labels_per_child[cid]
        pad_mask_t = torch.from_numpy(pad_mask)
        return SequenceItem(
            x=x, y=int(y), child_id=cid, n_strokes_actual=n_kept,
            pad_mask=pad_mask_t, offset_used=offset,
        )

    def __len__(self) -> int:
        return len(self.child_ids)

    def __getitem__(self, idx: int):
        cid = self.child_ids[idx]
        item = self._build_item(cid)
        if self.return_dict:
            return {
                "x": item.x,
                "y": torch.tensor(item.y, dtype=torch.long),
                "child_id": item.child_id,
                "pad_mask": item.pad_mask,
                "n_strokes_actual": item.n_strokes_actual,
                "offset_used": item.offset_used,
            }
        return item.x, torch.tensor(item.y, dtype=torch.long)


# ---------------------------------------------------------------------------
# Sliding window dataset (Exp A.2, A.4)
# ---------------------------------------------------------------------------


@dataclass
class WindowItem:
    """Un échantillon de SlidingWindowDataset."""
    x: torch.Tensor
    y: int
    child_id: str
    window_start: int
    window_end: int


class SlidingWindowDataset(Dataset):
    """Un échantillon = une fenêtre de window_len traits consécutifs.

    Génère toutes les fenêtres possibles de taille window_len avec stride
    `stride`. Si un enfant a < window_len traits, il est ignoré (warning).

    Pas de trial subsampling ici : les fenêtres SONT déjà l'augmentation.

    Pour l'inférence au niveau enfant : voir aggregate_window_predictions().
    """

    def __init__(
        self,
        dataset: ProcessedDataset,
        child_ids: list[str],
        window_len: int = 10,
        stride: int = 1,
        return_dict: bool = False,
    ):
        if window_len <= 0 or stride <= 0:
            raise ValueError("window_len and stride must be positive")

        self.dataset = dataset
        self.child_ids = list(child_ids)
        self.window_len = window_len
        self.stride = stride
        self.return_dict = return_dict

        self._items: list[WindowItem] = []
        self._skipped_children: list[str] = []
        for cid in self.child_ids:
            self._items.extend(self._windows_for_child(cid))

    @property
    def n_channels(self) -> int:
        return self.dataset.params_per_child[self.child_ids[0]].shape[1]

    def _windows_for_child(self, cid: str) -> list[WindowItem]:
        params = self.dataset.params_per_child[cid]
        n_strokes = params.shape[0]
        if n_strokes < self.window_len:
            self._skipped_children.append(cid)
            return []
        items = []
        y = self.dataset.labels_per_child[cid]
        for start in range(0, n_strokes - self.window_len + 1, self.stride):
            end = start + self.window_len
            window = params[start:end]
            x = torch.from_numpy(np.ascontiguousarray(window.T)).float()
            items.append(WindowItem(
                x=x, y=int(y), child_id=cid,
                window_start=start, window_end=end,
            ))
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        item = self._items[idx]
        if self.return_dict:
            return {
                "x": item.x,
                "y": torch.tensor(item.y, dtype=torch.long),
                "child_id": item.child_id,
                "window_start": item.window_start,
                "window_end": item.window_end,
            }
        return item.x, torch.tensor(item.y, dtype=torch.long)

    def child_id_per_item(self) -> list[str]:
        return [it.child_id for it in self._items]


# ---------------------------------------------------------------------------
# Aggregation niveau fenêtre -> niveau enfant
# ---------------------------------------------------------------------------


def aggregate_window_predictions(
    child_ids_per_window: list[str],
    proba_per_window: np.ndarray,
    aggregation: str = "mean_proba",
) -> tuple[np.ndarray, np.ndarray]:
    """Agrège les prédictions de fenêtres en prédictions par enfant."""
    child_ids = np.asarray(child_ids_per_window)
    proba = np.asarray(proba_per_window)
    unique_ids = np.array(sorted(set(child_ids.tolist())))
    proba_per_child = np.zeros(len(unique_ids))
    for i, cid in enumerate(unique_ids):
        mask = child_ids == cid
        if aggregation == "mean_proba":
            proba_per_child[i] = float(np.mean(proba[mask]))
        elif aggregation == "majority_vote":
            preds = (proba[mask] >= 0.5).astype(int)
            n1 = int(preds.sum())
            n0 = len(preds) - n1
            if n1 > n0:
                proba_per_child[i] = 1.0
            elif n0 > n1:
                proba_per_child[i] = 0.0
            else:
                proba_per_child[i] = float(np.mean(proba[mask]))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return unique_ids, proba_per_child
