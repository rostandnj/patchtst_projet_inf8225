"""
src/utils/traces_io.py

Chargement et accès aux traces brutes pour Exp B.

Format attendu de data/processed/traces.npz :
    Pour chaque trait : 3 arrays
        {cid}_t{tid:03d}_x : (T,) coordonnées x en mm
        {cid}_t{tid:03d}_y : (T,) coordonnées y en mm
        {cid}_t{tid:03d}_t : (T,) timestamps en s

Les T varient par trait (typiquement 100-500 samples après nettoyage des
plateaux statiques). Les arrays sont stockés tels quels (pas de padding).

API :
    load_traces(processed_dir) -> TracesDataset
    TracesDataset.get_trace(child_id, trial_id) -> (x, y, t)
    TracesDataset.list_trial_ids(child_id) -> list[int]
    TracesDataset.iter_strokes(child_id) -> iterator over (trial_id, x, y, t)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


KEY_PATTERN = re.compile(r"^(S\d+)_t(\d+)_(x|y|t)$")


@dataclass
class TracesDataset:
    """Dataset des traces brutes nettoyées.

    Attributes:
        child_ids: liste des enfants disponibles
        trial_ids_per_child: dict[cid -> sorted list of trial_ids]
        traces: dict[(cid, tid) -> dict avec keys 'x', 'y', 't' chacun array (T,)]
    """
    child_ids: list[str]
    trial_ids_per_child: dict[str, list[int]]
    traces: dict[tuple[str, int], dict[str, np.ndarray]]

    @property
    def total_strokes(self) -> int:
        return len(self.traces)

    def get_trace(self, child_id: str, trial_id: int) -> dict[str, np.ndarray]:
        """Retourne un dict avec keys 'x', 'y', 't' (chacun array 1D)."""
        return self.traces[(child_id, trial_id)]

    def list_trial_ids(self, child_id: str) -> list[int]:
        return self.trial_ids_per_child[child_id]

    def iter_strokes(self, child_id: str) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Itère sur les traits d'un enfant, dans l'ordre des trial_ids."""
        for tid in self.trial_ids_per_child[child_id]:
            tr = self.traces[(child_id, tid)]
            yield tid, tr["x"], tr["y"], tr["t"]

    def n_strokes(self, child_id: str) -> int:
        return len(self.trial_ids_per_child[child_id])

    def trace_lengths_stats(self) -> dict[str, float]:
        """Stats sur les longueurs de traits (min/max/mean/median/p25/p75)."""
        lengths = np.array([len(tr["x"]) for tr in self.traces.values()])
        return {
            "n_strokes": int(len(lengths)),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p25": float(np.percentile(lengths, 25)),
            "p75": float(np.percentile(lengths, 75)),
            "p95": float(np.percentile(lengths, 95)),
        }


def load_traces(processed_dir: Path) -> TracesDataset:
    """Charge data/processed/traces.npz et reconstruit la structure logique."""
    processed_dir = Path(processed_dir)
    npz = np.load(processed_dir / "traces.npz", allow_pickle=False)

    # Groupe les keys par (cid, tid)
    grouped: dict[tuple[str, int], dict[str, np.ndarray]] = defaultdict(dict)
    for key in npz.files:
        m = KEY_PATTERN.match(key)
        if m is None:
            continue
        cid, tid_str, channel = m.group(1), m.group(2), m.group(3)
        tid = int(tid_str)
        grouped[(cid, tid)][channel] = np.asarray(npz[key], dtype=np.float32)

    # Vérifie que chaque trait a bien x, y, t
    valid_traces: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    incomplete = []
    for key, channels in grouped.items():
        if set(channels.keys()) >= {"x", "y", "t"}:
            # Vérifie shapes cohérentes
            lx, ly, lt = len(channels["x"]), len(channels["y"]), len(channels["t"])
            if lx == ly == lt and lx >= 2:
                valid_traces[key] = channels
            else:
                incomplete.append(key)
        else:
            incomplete.append(key)

    if incomplete:
        print(f"[traces_io] Warning: {len(incomplete)} incomplete traits skipped")

    # Construit child_ids et trial_ids_per_child
    trial_ids_per_child: dict[str, list[int]] = defaultdict(list)
    for cid, tid in valid_traces.keys():
        trial_ids_per_child[cid].append(tid)
    for cid in trial_ids_per_child:
        trial_ids_per_child[cid].sort()

    child_ids = sorted(trial_ids_per_child.keys())

    return TracesDataset(
        child_ids=child_ids,
        trial_ids_per_child=dict(trial_ids_per_child),
        traces=dict(valid_traces),
    )
