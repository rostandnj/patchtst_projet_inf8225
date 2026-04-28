"""
src/data/exp_b_datasets.py

Datasets PyTorch pour Exp B : classification TDAH/CTRL à partir des
traces brutes x(t), y(t) au niveau du trait individuel.

UN ÉCHANTILLON = UN TRAIT.
    - Input  : (n_channels, T) avec n_channels ∈ {2 (x,y) ou 3 (x,y,v)}
    - Label  : 0/1, hérité de l'enfant à qui appartient le trait
    - 663 échantillons train possibles (vs 24 en Exp A) → beaucoup de signal

LONGUEUR T :
    Les traits ont une longueur variable (typiquement 100-500 samples).
    On uniformise à T=200 par :
        - Padding zéro à droite si T_actual < 200
        - Truncation centrée si T_actual > 200
          (on garde le milieu, là où est typiquement le pic d'accélération)

Pourquoi pas resampling ? Resampler à 200 changerait la fréquence
d'échantillonnage et donc la sémantique des dérivées (vitesse, accélération).
Le padding/truncation préserve la structure temporelle native.

NORMALISATION :
    Pas de normalisation dans le dataset (RevIN dans le modèle s'en charge).
    On retire juste le centre de gravité (mean centering par trait) pour x,y
    afin d'enlever le biais d'origine du repère, qui ne porte pas d'info
    motrice.

AGRÉGATION ENFANT :
    Pour l'évaluation au niveau enfant : voir aggregate_stroke_predictions().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.traces_io import TracesDataset


VALID_CHANNELS = ("x", "y", "vx", "vy", "v")


# ---------------------------------------------------------------------------
# Helpers de transformation
# ---------------------------------------------------------------------------


def center_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retire la moyenne de x et y (mean centering)."""
    return x - x.mean(), y - y.mean()


def compute_velocity(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule vx, vy, v à partir de x, y, t par différence finie centrée.

    Utilise np.gradient avec edge_order=1 pour gérer les bords proprement.

    Returns:
        (vx, vy, v) chacun shape (T,)
    """
    if len(t) < 2:
        return np.zeros_like(x), np.zeros_like(y), np.zeros_like(x)
    # Garde-fou : si t a des valeurs identiques (très rare), on utilise un dt uniforme
    dt = np.diff(t)
    if np.any(dt <= 0):
        # Fallback : assume sample rate uniforme
        dt_uniform = np.mean(dt[dt > 0]) if np.any(dt > 0) else 1.0 / 200.0
        t = np.arange(len(x)) * dt_uniform
    vx = np.gradient(x, t, edge_order=1)
    vy = np.gradient(y, t, edge_order=1)
    v = np.sqrt(vx**2 + vy**2)
    return vx, vy, v


def pad_or_truncate(
    arr: np.ndarray,
    target_len: int,
    pad_value: float = 0.0,
    pad_strategy: str = "zero",
) -> tuple[np.ndarray, int]:
    """Pad à droite ou truncate au centre pour atteindre target_len.

    Args:
        arr: array 1D à padder/truncate.
        target_len: longueur cible.
        pad_value: valeur de remplissage si pad_strategy='zero'.
        pad_strategy: 'zero' (zero pad) ou 'edge' (répète la dernière valeur).
            'edge' est recommandé pour éviter les artefacts statistiques de
            normalisation (RevIN) sur des traits courts.

    Returns:
        (padded_or_truncated, n_actual) où n_actual ≤ target_len est le
        nombre d'échantillons réels avant padding.
    """
    n = len(arr)
    if n == target_len:
        return arr.copy(), n
    if n < target_len:
        if pad_strategy == "edge" and n >= 1:
            # Répète la dernière valeur
            out = np.full(target_len, arr[-1], dtype=arr.dtype)
            out[:n] = arr
        else:  # 'zero' ou cas dégénéré
            out = np.full(target_len, pad_value, dtype=arr.dtype)
            out[:n] = arr
        return out, n
    # n > target_len : truncate au milieu
    excess = n - target_len
    start = excess // 2
    end = start + target_len
    return arr[start:end].copy(), target_len


# ---------------------------------------------------------------------------
# Dataset principal : un échantillon = un trait
# ---------------------------------------------------------------------------


@dataclass
class StrokeItem:
    """Un échantillon retourné par RawTraceStrokeDataset (debug)."""
    x: torch.Tensor                # (n_channels, target_len)
    y: int                         # label 0/1
    child_id: str
    trial_id: int
    n_actual: int                  # T réel avant padding/truncation
    pad_mask: torch.Tensor         # (target_len,) bool, True = padded


class RawTraceStrokeDataset(Dataset):
    """Un trait par échantillon.

    Args:
        traces : TracesDataset chargé
        labels_per_child : dict cid -> 0/1
        child_ids : sous-ensemble d'enfants à utiliser
        target_len : longueur cible T (default 200)
        channels : tuple de canaux à inclure parmi VALID_CHANNELS
            ('x', 'y') = 2 canaux par défaut.
            ('x', 'y', 'v') = ajoute vélocité euclidienne.
            ('x', 'y', 'vx', 'vy') = canaux séparés.
        center_xy_flag : si True, retire la moyenne de x et y avant tout
            (recommandé : sinon RevIN va lutter avec les valeurs absolues)
        return_dict : si True, retourne dict complet (debug)
    """

    def __init__(
        self,
        traces: TracesDataset,
        labels_per_child: dict[str, int],
        child_ids: list[str],
        target_len: int = 200,
        channels: tuple[str, ...] = ("x", "y"),
        center_xy_flag: bool = True,
        return_dict: bool = False,
        # ---- Phase B fixes ----
        min_motion_std: float = 1.0,
        # Filtre les traits dont std(x) < min_motion_std OU std(y) < min_motion_std
        # (en mm). Évite les traits quasi-immobiles qui causent des problèmes
        # numériques en aval (RevIN amplifie). 0 = pas de filtrage.
        scale_xy: float = 1.0,
        # Divise x, y par scale_xy avant centering. Recommandé : 100.0 pour
        # ramener les coordonnées (en mm) dans une plage [-1.5, 1.5] adaptée
        # aux Transformers. 1.0 = pas de scaling.
        pad_strategy: str = "edge",
        # 'zero' (zero pad à droite, comme Phase B.1) ou 'edge' (répète la
        # dernière valeur, recommandé pour préserver les statistiques de RevIN).
    ):
        for ch in channels:
            if ch not in VALID_CHANNELS:
                raise ValueError(f"Unknown channel: {ch!r}, valid: {VALID_CHANNELS}")
        if pad_strategy not in ("zero", "edge"):
            raise ValueError(f"pad_strategy must be 'zero' or 'edge', got {pad_strategy!r}")

        self.traces = traces
        self.labels_per_child = labels_per_child
        self.child_ids = list(child_ids)
        self.target_len = target_len
        self.channels = tuple(channels)
        self.center_xy_flag = center_xy_flag
        self.return_dict = return_dict
        self.min_motion_std = float(min_motion_std)
        self.scale_xy = float(scale_xy)
        self.pad_strategy = pad_strategy

        # Construit l'index plat (cid, tid) pour __getitem__ rapide
        # Avec filtrage des traits dégénérés (mouvement quasi-nul)
        self._items: list[tuple[str, int]] = []
        self._n_filtered_motion = 0
        for cid in self.child_ids:
            if cid not in self.traces.trial_ids_per_child:
                continue
            for tid in self.traces.trial_ids_per_child[cid]:
                tr = self.traces.traces[(cid, tid)]
                if self.min_motion_std > 0:
                    sx = float(np.std(tr["x"]))
                    sy = float(np.std(tr["y"]))
                    if sx < self.min_motion_std or sy < self.min_motion_std:
                        self._n_filtered_motion += 1
                        continue
                self._items.append((cid, tid))

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_filtered_motion(self) -> int:
        """Nombre de traits exclus pour cause de mouvement insuffisant."""
        return self._n_filtered_motion

    def child_id_per_item(self) -> list[str]:
        return [cid for cid, _ in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def _build_channels(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        """Construit les canaux demandés à partir de (x, y, t)."""
        # Pre-scale x, y
        if self.scale_xy != 1.0:
            x = x / self.scale_xy
            y = y / self.scale_xy
        if self.center_xy_flag:
            x, y = center_xy(x, y)

        out: dict[str, np.ndarray] = {"x": x, "y": y}

        # Si on a besoin de vélocité, on la calcule (note : sur x, y déjà scalés)
        need_velocity = any(ch in ("vx", "vy", "v") for ch in self.channels)
        if need_velocity:
            vx, vy, v = compute_velocity(x, y, t)
            out["vx"] = vx
            out["vy"] = vy
            out["v"] = v

        return out

    def _build_item(self, cid: str, tid: int) -> StrokeItem:
        tr = self.traces.traces[(cid, tid)]
        x, y, t = tr["x"], tr["y"], tr["t"]

        chans_dict = self._build_channels(x, y, t)

        # Stack canaux puis pad/truncate
        channel_arrays = []
        n_actual = 0
        for ch in self.channels:
            arr = chans_dict[ch].astype(np.float32)
            arr_padded, n_actual = pad_or_truncate(
                arr, self.target_len,
                pad_value=0.0,
                pad_strategy=self.pad_strategy,
            )
            channel_arrays.append(arr_padded)

        x_tensor = torch.from_numpy(np.stack(channel_arrays, axis=0))  # (n_channels, target_len)

        # Pad mask : True où c'est du padding (utile pour attention masking si besoin)
        pad_mask = torch.zeros(self.target_len, dtype=torch.bool)
        if n_actual < self.target_len:
            pad_mask[n_actual:] = True

        label = int(self.labels_per_child[cid])

        return StrokeItem(
            x=x_tensor, y=label, child_id=cid, trial_id=tid,
            n_actual=n_actual, pad_mask=pad_mask,
        )

    def __getitem__(self, idx: int):
        cid, tid = self._items[idx]
        item = self._build_item(cid, tid)
        if self.return_dict:
            return {
                "x": item.x,
                "y": torch.tensor(item.y, dtype=torch.long),
                "child_id": item.child_id,
                "trial_id": item.trial_id,
                "n_actual": item.n_actual,
                "pad_mask": item.pad_mask,
            }
        return item.x, torch.tensor(item.y, dtype=torch.long)


# ---------------------------------------------------------------------------
# Aggregation niveau trait → niveau enfant
# ---------------------------------------------------------------------------


def aggregate_stroke_predictions(
    child_ids_per_stroke: list[str],
    proba_per_stroke: np.ndarray,
    aggregation: str = "mean_proba",
) -> tuple[np.ndarray, np.ndarray]:
    """Agrège les prédictions trait par trait en prédictions par enfant.

    Args:
        child_ids_per_stroke : liste des cid par trait dans le même ordre
            que proba_per_stroke
        proba_per_stroke : array (n_strokes,) avec P(ADHD) prédite
        aggregation : 'mean_proba' (moyenne des probas) ou 'majority_vote'

    Returns:
        (unique_child_ids, proba_per_child) ordonnés par cid
    """
    cids = np.asarray(child_ids_per_stroke)
    probs = np.asarray(proba_per_stroke)
    unique_cids = np.array(sorted(set(cids.tolist())))
    proba_per_child = np.zeros(len(unique_cids), dtype=np.float64)
    for i, cid in enumerate(unique_cids):
        mask = cids == cid
        if aggregation == "mean_proba":
            proba_per_child[i] = float(probs[mask].mean())
        elif aggregation == "majority_vote":
            preds = (probs[mask] >= 0.5).astype(int)
            n1 = int(preds.sum())
            n0 = len(preds) - n1
            if n1 > n0:
                proba_per_child[i] = 1.0
            elif n0 > n1:
                proba_per_child[i] = 0.0
            else:
                proba_per_child[i] = float(probs[mask].mean())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")
    return unique_cids, proba_per_child
