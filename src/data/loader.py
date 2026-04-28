"""
src/data/loader.py

Chargement des fichiers SXX.json et assemblage en structures Python
exploitables pour l'entraînement et l'évaluation.

Un JSON = un enfant = liste ordonnée de trials.
Sortie canonique : `ChildRecord` qui contient tous les traits valides d'un
enfant dans l'ordre d'acquisition (trié par trialID).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .sigma_lognormal import (
    SigmaLognormalStroke,
    extract_sigma_lognormal_parameters,
    is_valid_faci,
    PARAMETER_NAMES,
)
from .raw_traces import (
    RawTrace,
    load_raw_from_trial,
    remove_plateaus,
)

logger = logging.getLogger(__name__)


@dataclass
class StrokeRecord:
    """Un trait valide : paramètres sigma-lognormaux + trace brute nettoyée."""
    child_id: str
    trial_id: int
    label: int                        # 0 = control (healthy), 1 = ADHD
    sigma_params: SigmaLognormalStroke
    raw_trace: RawTrace               # trace nettoyée (plateaux retirés)


@dataclass
class ChildRecord:
    """Tous les traits valides d'un enfant, dans l'ordre d'acquisition."""
    child_id: str
    label: int                        # 0 = control, 1 = ADHD
    strokes: list[StrokeRecord] = field(default_factory=list)

    @property
    def n_strokes(self) -> int:
        return len(self.strokes)

    def params_matrix(self) -> np.ndarray:
        """Retourne matrice (n_strokes, N_PARAMS) dans l'ordre canonique."""
        return np.stack([s.sigma_params.to_vector() for s in self.strokes], axis=0)

    def trial_ids(self) -> np.ndarray:
        return np.array([s.trial_id for s in self.strokes], dtype=np.int64)


@dataclass
class RejectionLog:
    """Trace détaillée des rejets logiciels pour rapport QC."""
    total_seen: int = 0
    kept: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    rejected_extraction_missing: int = 0
    rejected_no_raw: int = 0
    rejected_no_movement: int = 0

    def record(self, reason: str):
        self.rejected_by_reason[reason] = self.rejected_by_reason.get(reason, 0) + 1

    def summary(self) -> str:
        lines = [f"Trials seen: {self.total_seen}",
                 f"Trials kept: {self.kept}",
                 f"Rejected (sigma-lognormal extraction missing): "
                 f"{self.rejected_extraction_missing}",
                 f"Rejected (no raw data): {self.rejected_no_raw}",
                 f"Rejected (no movement detected): {self.rejected_no_movement}",
                 "Rejected by rules:"]
        for reason, n in sorted(self.rejected_by_reason.items()):
            lines.append(f"  {reason}: {n}")
        return "\n".join(lines)


def load_child(
    json_path: Path,
    snr_min: float = 15.0,
    d_max_mm: float = 500.0,
    duration_max_s: Optional[float] = None,
    velocity_threshold_mm_s: float = 5.0,
    log: Optional[RejectionLog] = None,
) -> ChildRecord:
    """Charge un fichier SXX.json, applique les filtres, retourne ChildRecord.

    Ordre d'application des filtres :
        1. Extraction SSVn (retourne None si incomplet)
        2. Filtres R1-R4 sur les paramètres (SNR, D, t0/mu, NaN)
        3. Chargement trace brute + nettoyage plateaux
        4. Filtre R5 sur durée nettoyée (si duration_max_s fourni)
    """
    with open(json_path) as f:
        trials = json.load(f)

    if not isinstance(trials, list) or len(trials) == 0:
        raise ValueError(f"{json_path.name}: file is empty or not a list")

    first_meta = trials[0].get("metaData", {}).get("participant", {})
    healthy = first_meta.get("health", {}).get("healthy")
    if healthy is None:
        raise ValueError(f"{json_path.name}: health.healthy not set")
    label = 0 if healthy is True else 1

    child_id = json_path.stem
    child = ChildRecord(child_id=child_id, label=label)

    if log is None:
        log = RejectionLog()

    trials_sorted = sorted(
        trials,
        key=lambda t: t.get("metaData", {}).get("trial", {}).get("trialID", 0),
    )

    for tr in trials_sorted:
        log.total_seen += 1

        tmeta = tr.get("metaData", {}).get("trial", {})
        trial_id = tmeta.get("trialID")
        if trial_id is None:
            log.record("missing_trialID")
            continue

        # --- Étape 1 : Extraction des paramètres sigma-lognormaux ---
        params = extract_sigma_lognormal_parameters(tr)
        if params is None:
            log.rejected_extraction_missing += 1
            continue

        # --- Étape 2 : Filtres R1-R4 (sans durée) ---
        valid, reason = is_valid_faci(
            params, snr_min=snr_min, d_max_mm=d_max_mm,
            duration_s=None, duration_max_s=None,
        )
        if not valid:
            log.record(reason)
            continue

        # --- Étape 3 : Trace brute + nettoyage ---
        raw = load_raw_from_trial(tr)
        if raw is None:
            log.rejected_no_raw += 1
            continue

        clean = remove_plateaus(raw, velocity_threshold_mm_s=velocity_threshold_mm_s)
        if clean is None or clean.n_samples < 10:
            log.rejected_no_movement += 1
            continue

        # --- Étape 4 : Filtre R5 sur durée (si demandé) ---
        if duration_max_s is not None:
            valid_r5, reason_r5 = is_valid_faci(
                params, snr_min=snr_min, d_max_mm=d_max_mm,
                duration_s=clean.duration_s, duration_max_s=duration_max_s,
            )
            if not valid_r5:
                log.record(reason_r5)
                continue

        stroke = StrokeRecord(
            child_id=child_id,
            trial_id=int(trial_id),
            label=label,
            sigma_params=params,
            raw_trace=clean,
        )
        child.strokes.append(stroke)
        log.kept += 1

    return child


def load_all_children(
    raw_dir: Path,
    **kwargs,
) -> tuple[list[ChildRecord], RejectionLog]:
    """Charge tous les S*.json d'un dossier. Retourne (liste ChildRecord, log)."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("S*.json"))
    if not files:
        raise FileNotFoundError(f"No S*.json files found in {raw_dir}")

    log = RejectionLog()
    children = []
    for jf in files:
        try:
            child = load_child(jf, log=log, **kwargs)
            children.append(child)
        except Exception as e:
            logger.warning("Failed to load %s: %s", jf.name, e)

    return children, log
