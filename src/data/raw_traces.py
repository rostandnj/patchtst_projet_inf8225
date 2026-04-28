"""
src/data/raw_traces.py

Nettoyage et traitement des traces brutes x(t), y(t) pour Exp B / Exp C.

Les fichiers JSON contiennent des traces avec des plateaux de position
constante au début (avant le mouvement) et à la fin (après que le stylo
s'est immobilisé). Ces plateaux ne portent pas d'information motrice et
dominent la longueur du signal. On les retire par détection de vitesse.

Canaux dérivés produits à partir de x(t), y(t) :
    - vx, vy : vitesses instantanées (différence centrée)
    - v      : vitesse euclidienne = sqrt(vx^2 + vy^2)
    - ax, ay : accélérations (dérivée seconde)
    - a      : module d'accélération

Option de resampling linéaire pour avoir une longueur fixe L par trait,
requise par PatchTST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Canaux retournés dans l'ordre canonique
RAW_CHANNELS_BASE = ["x", "y"]
RAW_CHANNELS_FULL = ["x", "y", "vx", "vy", "v", "ax", "ay", "a"]


@dataclass
class RawTrace:
    """Conteneur pour une trace brute nettoyée."""
    t: np.ndarray                    # timestamps en secondes, shape (L,)
    x: np.ndarray                    # shape (L,), en mm
    y: np.ndarray                    # shape (L,), en mm
    sample_rate_hz: float = 200.0    # fréquence d'échantillonnage

    @property
    def n_samples(self) -> int:
        return len(self.x)

    @property
    def duration_s(self) -> float:
        if self.n_samples < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])


def load_raw_from_trial(trial: dict) -> Optional[RawTrace]:
    """Extrait (t, x, y) d'un trial JSON. Retourne None si invalide."""
    raw = trial.get("rawData", {})
    t_list = raw.get("timeStamps", [[]])
    x_list = raw.get("x", [[]])
    y_list = raw.get("y", [[]])

    if not (t_list and x_list and y_list):
        return None
    if len(t_list) == 0 or len(x_list) == 0 or len(y_list) == 0:
        return None

    t = np.asarray(t_list[0], dtype=np.float64)
    x = np.asarray(x_list[0], dtype=np.float64)
    y = np.asarray(y_list[0], dtype=np.float64)

    if len(t) != len(x) or len(x) != len(y):
        return None
    if len(t) < 3:
        return None

    sr = trial.get("metaData", {}).get("device", {}).get("sampleRate", 200.0)
    return RawTrace(t=t, x=x, y=y, sample_rate_hz=float(sr))


def remove_plateaus(
    trace: RawTrace,
    velocity_threshold_mm_s: float = 5.0,
    min_run_samples: int = 3,
    pad_before: int = 2,
    pad_after: int = 2,
) -> Optional[RawTrace]:
    """Retire les plateaux statiques de début et de fin d'une trace.

    Principe : on calcule la vitesse euclidienne instantanée. Le mouvement
    est considéré actif quand la vitesse dépasse le seuil pendant au moins
    `min_run_samples` échantillons consécutifs.

    On coupe au premier et au dernier échantillon actif (avec un petit padding
    pour garder un peu du début et de la fin du mouvement).

    Paramètres:
        velocity_threshold_mm_s : seuil de vitesse (mm/s) pour considérer
            un échantillon comme "en mouvement". Typique: 5 mm/s pour éviter
            le bruit de digitizer.
        min_run_samples : nombre minimum d'échantillons consécutifs au-dessus
            du seuil pour valider le début/fin du mouvement. Évite les spikes.
        pad_before, pad_after : marge conservée avant/après.

    Retourne None si la trace n'a pas de mouvement détectable.
    """
    if trace.n_samples < 10:
        return None

    dt = np.diff(trace.t)
    # Dérivées finies d'ordre 1 (différence avant)
    dx = np.diff(trace.x)
    dy = np.diff(trace.y)
    # Garde contre dt nul
    safe_dt = np.where(dt > 0, dt, 1.0 / trace.sample_rate_hz)
    vx = dx / safe_dt
    vy = dy / safe_dt
    v = np.sqrt(vx**2 + vy**2)

    # v a longueur N-1. On aligne en 0-padding à gauche pour rester sur N.
    v_full = np.concatenate([[0.0], v])

    active = v_full > velocity_threshold_mm_s

    # Cherche le premier run de min_run_samples actifs consécutifs
    def first_run_index(mask: np.ndarray, min_run: int) -> Optional[int]:
        run = 0
        for i, a in enumerate(mask):
            if a:
                run += 1
                if run >= min_run:
                    return i - min_run + 1
            else:
                run = 0
        return None

    start = first_run_index(active, min_run_samples)
    if start is None:
        return None
    end = first_run_index(active[::-1], min_run_samples)
    if end is None:
        return None
    end = len(active) - 1 - end  # ré-indexation

    if end <= start + 1:
        return None

    # Padding
    start = max(0, start - pad_before)
    end = min(trace.n_samples - 1, end + pad_after)

    return RawTrace(
        t=trace.t[start:end + 1] - trace.t[start],  # re-zéro le temps
        x=trace.x[start:end + 1],
        y=trace.y[start:end + 1],
        sample_rate_hz=trace.sample_rate_hz,
    )


def compute_derived_channels(trace: RawTrace) -> dict[str, np.ndarray]:
    """Calcule vx, vy, v, ax, ay, a par différence centrée.

    Retourne un dict avec les 8 canaux (x, y inclus) tous de même longueur L.
    Les dérivées sont calculées par np.gradient (ordre 2 centré sauf aux
    bords où c'est ordre 1).
    """
    t = trace.t
    x, y = trace.x, trace.y

    vx = np.gradient(x, t, edge_order=1)
    vy = np.gradient(y, t, edge_order=1)
    v = np.sqrt(vx**2 + vy**2)

    ax = np.gradient(vx, t, edge_order=1)
    ay = np.gradient(vy, t, edge_order=1)
    a = np.sqrt(ax**2 + ay**2)

    return {"x": x, "y": y, "vx": vx, "vy": vy, "v": v,
            "ax": ax, "ay": ay, "a": a}


def resample_trace(
    channels: dict[str, np.ndarray],
    t_original: np.ndarray,
    target_length: int,
) -> dict[str, np.ndarray]:
    """Resample linéairement chaque canal à une longueur cible fixe.

    Préserve la forme temporelle relative tout en normalisant la durée,
    ce qui convient à PatchTST qui exige un L fixe.
    """
    if len(t_original) < 2:
        raise ValueError("t_original too short to resample")

    t_new = np.linspace(t_original[0], t_original[-1], target_length)
    out = {}
    for name, signal in channels.items():
        out[name] = np.interp(t_new, t_original, signal)
    return out


def trace_to_tensor(
    trace: RawTrace,
    channels: list[str],
    target_length: int,
) -> np.ndarray:
    """Construit le tenseur (M, L) attendu par PatchTST pour un trait.

    M = nombre de canaux sélectionnés, L = target_length.
    """
    all_channels = compute_derived_channels(trace)
    resampled = resample_trace(all_channels, trace.t, target_length)
    try:
        mat = np.stack([resampled[c] for c in channels], axis=0)
    except KeyError as e:
        raise ValueError(f"Unknown channel requested: {e}")
    return mat.astype(np.float32)
