"""
src/data/exp_c_datasets.py

Datasets PyTorch pour Exp C : classification TDAH/CTRL à partir de la
COMBINAISON traces brutes + paramètres sigma-lognormaux.

DESIGN C.1 — Concaténation 16 canaux
    Pour chaque trait :
        - Canaux 0-1 : x(t), y(t) (200 timesteps, comme Exp B)
        - Canaux 2-15 : les 14 paramètres sigma-lognormaux diffusés
                        sur les 200 timesteps (valeurs constantes)
    Total : tensor de shape (16, 200) par échantillon.

PatchTST en channel-independence va traiter chaque canal séparément avec
des poids partagés, ce qui marche pour les deux types :
    - Canaux dynamiques (x, y) : exploitation de la structure temporelle
    - Canaux constants (params) : extraction de la valeur (chaque patch
      reçoit la même valeur, RevIN standardise par canal)

NORMALISATION :
    - x, y : centrés par sample (comme Exp B), divisés par scale_xy=100
    - Paramètres sigma-lognormaux : passés bruts (RevIN normalise)

ALIGNMENT :
    Le mapping (cid, tid) → params est crucial. On utilise les params
    extraits dans Phase 1 (data/processed/params.npz via ProcessedDataset),
    et on les aligne avec les traits de traces.npz par leur (cid, tid).

LABELS :
    Hérités de l'enfant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.exp_b_datasets import (
    center_xy, compute_velocity, pad_or_truncate, VALID_CHANNELS,
)
from src.utils.data_io import ProcessedDataset
from src.utils.traces_io import TracesDataset


# Pour Exp A on a vu 14 paramètres dans l'ordre canonique Faci
DEFAULT_N_PARAMS = 14


@dataclass
class StrokeWithParamsItem:
    """Échantillon retourné par RawTraceWithParamsDataset."""
    x: torch.Tensor                # (n_total_channels, target_len)
    y: int                         # label 0/1
    child_id: str
    trial_id: int
    n_actual: int                  # T réel des traces avant padding
    pad_mask: torch.Tensor         # (target_len,) bool


class RawTraceWithParamsDataset(Dataset):
    """Un trait par échantillon, avec traces + paramètres concaténés.

    Args:
        traces : TracesDataset chargé.
        params_dataset : ProcessedDataset (avec params_per_child et la map
            stroke_index_per_child donnant l'ordre des trial_ids).
        labels_per_child : dict cid -> 0/1.
        child_ids : sous-ensemble d'enfants.
        target_len : longueur cible des traces (default 200).
        trace_channels : canaux des traces parmi VALID_CHANNELS
            (default ('x', 'y')).
        center_xy_flag : centrage des traces (default True).
        scale_xy : facteur de division des traces (default 100.0).
        pad_strategy : 'zero' ou 'edge' (default 'edge').
        min_motion_std : filtre mouvement minimum (default 1.0 mm en
            valeurs ORIGINALES, avant scale_xy).
        params_to_use : liste d'indices des params à utiliser (default
            None = tous les 14). Permet de tester ablations.
        return_dict : si True, retourne dict complet (debug).
    """

    def __init__(
        self,
        traces: TracesDataset,
        params_dataset: ProcessedDataset,
        labels_per_child: dict[str, int],
        child_ids: list[str],
        target_len: int = 200,
        trace_channels: tuple[str, ...] = ("x", "y"),
        center_xy_flag: bool = True,
        scale_xy: float = 100.0,
        pad_strategy: str = "edge",
        min_motion_std: float = 1.0,
        params_to_use: list[int] = None,
        return_dict: bool = False,
    ):
        for ch in trace_channels:
            if ch not in VALID_CHANNELS:
                raise ValueError(f"Unknown channel: {ch!r}")
        if pad_strategy not in ("zero", "edge"):
            raise ValueError(f"pad_strategy must be 'zero' or 'edge'")

        self.traces = traces
        self.params_dataset = params_dataset
        self.labels_per_child = labels_per_child
        self.child_ids = list(child_ids)
        self.target_len = target_len
        self.trace_channels = tuple(trace_channels)
        self.center_xy_flag = center_xy_flag
        self.scale_xy = float(scale_xy)
        self.pad_strategy = pad_strategy
        self.min_motion_std = float(min_motion_std)
        self.return_dict = return_dict

        # Indices des params à utiliser (default : tous)
        if params_to_use is None:
            params_to_use = list(range(DEFAULT_N_PARAMS))
        self.params_to_use = list(params_to_use)
        self.n_params = len(self.params_to_use)

        # Construit l'index plat (cid, tid) avec filtrage
        self._items: list[tuple[str, int]] = []
        self._n_filtered_motion = 0
        self._n_missing_params = 0
        for cid in self.child_ids:
            if cid not in self.traces.trial_ids_per_child:
                continue
            for tid in self.traces.trial_ids_per_child[cid]:
                tr = self.traces.traces[(cid, tid)]
                # Filtre traits dégénérés
                if self.min_motion_std > 0:
                    sx = float(np.std(tr["x"]))
                    sy = float(np.std(tr["y"]))
                    if sx < self.min_motion_std or sy < self.min_motion_std:
                        self._n_filtered_motion += 1
                        continue
                # Vérifie qu'on a des params pour ce trait
                params = self._get_params_for_stroke(cid, tid)
                if params is None:
                    self._n_missing_params += 1
                    continue
                self._items.append((cid, tid))

    @property
    def n_trace_channels(self) -> int:
        return len(self.trace_channels)

    @property
    def n_total_channels(self) -> int:
        """Nombre total de canaux (traces + params)."""
        return self.n_trace_channels + self.n_params

    @property
    def n_filtered_motion(self) -> int:
        return self._n_filtered_motion

    @property
    def n_missing_params(self) -> int:
        return self._n_missing_params

    def child_id_per_item(self) -> list[str]:
        return [cid for cid, _ in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def _get_params_for_stroke(self, cid: str, tid: int) -> np.ndarray:
        """Retourne le vecteur de 14 paramètres pour le trait (cid, tid).

        ProcessedDataset.params_per_child[cid] est un array (n_strokes, 14).
        On doit retrouver la position du tid dans cet array via
        stroke_indices_per_child[cid] qui liste les tids dans l'ordre.
        Returns None si introuvable.
        """
        if not hasattr(self.params_dataset, 'params_per_child'):
            return None
        if cid not in self.params_dataset.params_per_child:
            return None
        params_arr = self.params_dataset.params_per_child[cid]  # (n, 14)

        # Trouve l'indice du tid dans la liste des tids de cet enfant
        # Si ProcessedDataset expose stroke_indices_per_child (= liste de tids)
        if hasattr(self.params_dataset, 'stroke_indices_per_child'):
            tids_list = self.params_dataset.stroke_indices_per_child.get(cid)
            if tids_list is None:
                return None
            try:
                idx = list(tids_list).index(tid)
            except ValueError:
                return None
        else:
            # Fallback : on assume que tid-1 == position (0-indexed)
            # Cela suppose que les tids vont de 1 à N sans trous, ce qui
            # peut être incorrect si certains traits ont été filtrés en Phase 1.
            # On vérifie au moins que la taille match.
            tids_in_traces = self.traces.trial_ids_per_child.get(cid, [])
            if len(params_arr) != len(tids_in_traces):
                # Les tailles diffèrent : params filtrés différemment des traces
                return None
            try:
                idx = tids_in_traces.index(tid)
            except ValueError:
                return None

        if idx < 0 or idx >= len(params_arr):
            return None
        return params_arr[idx]  # (14,)

    def _build_traces(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[dict, int]:
        """Construit les canaux traces avec scale + center + pad."""
        # Pre-scale x, y
        if self.scale_xy != 1.0:
            x = x / self.scale_xy
            y = y / self.scale_xy
        if self.center_xy_flag:
            x, y = center_xy(x, y)

        chans_dict = {"x": x, "y": y}
        need_velocity = any(ch in ("vx", "vy", "v") for ch in self.trace_channels)
        if need_velocity:
            vx, vy, v = compute_velocity(x, y, t)
            chans_dict["vx"] = vx
            chans_dict["vy"] = vy
            chans_dict["v"] = v

        # Pad/truncate chaque canal à target_len
        out = {}
        n_actual = 0
        for ch in self.trace_channels:
            arr = chans_dict[ch].astype(np.float32)
            arr_p, n_actual = pad_or_truncate(
                arr, self.target_len, pad_value=0.0, pad_strategy=self.pad_strategy,
            )
            out[ch] = arr_p
        return out, n_actual

    def _build_item(self, cid: str, tid: int) -> StrokeWithParamsItem:
        # Traces
        tr = self.traces.traces[(cid, tid)]
        x, y, t = tr["x"], tr["y"], tr["t"]
        trace_chans, n_actual = self._build_traces(x, y, t)

        # Params
        params = self._get_params_for_stroke(cid, tid)  # (14,) ou None
        # Si None on a déjà filtré en init, donc ça ne devrait pas arriver
        params_selected = params[self.params_to_use].astype(np.float32)  # (n_params,)

        # Diffuse les params sur target_len
        param_channels = []
        for p_val in params_selected:
            param_channels.append(np.full(self.target_len, p_val, dtype=np.float32))

        # Stack tous les canaux : traces puis params
        all_channels = []
        for ch in self.trace_channels:
            all_channels.append(trace_chans[ch])
        all_channels.extend(param_channels)

        x_tensor = torch.from_numpy(np.stack(all_channels, axis=0))  # (n_total, T)

        pad_mask = torch.zeros(self.target_len, dtype=torch.bool)
        if n_actual < self.target_len:
            pad_mask[n_actual:] = True

        label = int(self.labels_per_child[cid])

        return StrokeWithParamsItem(
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
