"""
src/baselines/features.py

Feature engineering au niveau enfant pour les baselines classiques.

Trois schémas d'agrégation, du plus simple au plus riche :

    FACI4_MEAN  (4 features)
        - Moyenne par enfant des 4 features sélectionnées par t-test dans
          Faci et al. (2021) : nbLog, SNR/nbLog, t0, D.
        - REPRODUCTION EXACTE de l'article.

    ALL14_MEAN  (14 features)
        - Moyenne par enfant des 14 paramètres sigma-lognormaux.
        - Extension de Faci en utilisant tous les paramètres extraits.

    ALL14_MEAN_STD  (28 features)
        - Moyenne ET écart-type par enfant pour chacun des 14 paramètres.
        - Capture la variabilité intra-enfant (info que la moyenne seule
          détruit). Premier pas vers la temporalité.

    RICH_STATS  (~140 features)
        - Pour chaque paramètre : mean, std, min, max, median, q25, q75,
          skewness, kurtosis, trend_slope (régression linéaire sur la
          séquence de traits).
        - Capture statistiquement la dynamique inter-traits, mais sans
          PatchTST -- c'est le baseline le plus fort. Si PatchTST ne bat
          pas RICH_STATS, l'apport temporel n'est pas significatif.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from src.utils.data_io import ProcessedDataset


FeatureScheme = Literal["FACI4_MEAN", "ALL14_MEAN", "ALL14_MEAN_STD", "RICH_STATS"]

# Indices canoniques des 14 paramètres
# Ordre : SNR, nbLog, SNR_nbLog, t0, D, mu, sigma, theta_s, theta_e, M, m, t_bar, s, Ac
FACI_SELECTED = ["nbLog", "SNR_nbLog", "t0", "D"]


def _faci_indices(param_names: list[str]) -> list[int]:
    """Indices des 4 features sélectionnées par Faci dans le vecteur 14-D."""
    return [param_names.index(n) for n in FACI_SELECTED]


def _trend_slope(values: np.ndarray) -> float:
    """Pente de la régression linéaire de values en fonction de l'index.

    Capture la tendance "fatigue / apprentissage" au cours des traits.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    mean_x, mean_y = x.mean(), values.mean()
    num = ((x - mean_x) * (values - mean_y)).sum()
    den = ((x - mean_x) ** 2).sum()
    if den == 0:
        return 0.0
    return float(num / den)


def child_features(
    params_matrix: np.ndarray,
    param_names: list[str],
    scheme: FeatureScheme,
) -> tuple[np.ndarray, list[str]]:
    """Calcule le vecteur de features pour un enfant.

    Args:
        params_matrix: (n_strokes, 14) paramètres ordonnés par trial.
        param_names: 14 noms en ordre canonique.
        scheme: schéma d'agrégation.

    Returns:
        (feature_vector, feature_names)
    """
    if scheme == "FACI4_MEAN":
        idx = _faci_indices(param_names)
        sub = params_matrix[:, idx]
        vec = sub.mean(axis=0)
        names = [f"mean_{n}" for n in FACI_SELECTED]
        return vec.astype(np.float64), names

    if scheme == "ALL14_MEAN":
        vec = params_matrix.mean(axis=0)
        names = [f"mean_{n}" for n in param_names]
        return vec.astype(np.float64), names

    if scheme == "ALL14_MEAN_STD":
        means = params_matrix.mean(axis=0)
        stds = params_matrix.std(axis=0, ddof=1) if params_matrix.shape[0] > 1 else np.zeros(params_matrix.shape[1])
        vec = np.concatenate([means, stds])
        names = ([f"mean_{n}" for n in param_names] +
                 [f"std_{n}" for n in param_names])
        return vec.astype(np.float64), names

    if scheme == "RICH_STATS":
        feats = []
        names = []
        n_strokes = params_matrix.shape[0]
        for i, pname in enumerate(param_names):
            col = params_matrix[:, i]
            stats = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col, ddof=1)) if n_strokes > 1 else 0.0,
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "median": float(np.median(col)),
                "q25": float(np.percentile(col, 25)),
                "q75": float(np.percentile(col, 75)),
                "skew": float(skew(col)) if n_strokes > 2 else 0.0,
                "kurt": float(kurtosis(col)) if n_strokes > 3 else 0.0,
                "trend": _trend_slope(col),
            }
            for stat_name, val in stats.items():
                feats.append(val if np.isfinite(val) else 0.0)
                names.append(f"{stat_name}_{pname}")
        return np.array(feats, dtype=np.float64), names

    raise ValueError(f"Unknown scheme: {scheme}")


def build_child_feature_matrix(
    dataset: ProcessedDataset,
    child_ids: list[str],
    scheme: FeatureScheme,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construit la matrice (n_children, n_features) pour un sous-ensemble.

    Returns:
        X: (n_children, n_features)
        y: (n_children,) labels
        feature_names: noms des features
    """
    X_list = []
    y_list = []
    feat_names_ref = None
    for cid in child_ids:
        params = dataset.params_per_child[cid]
        vec, names = child_features(params, dataset.param_names, scheme)
        if feat_names_ref is None:
            feat_names_ref = names
        elif names != feat_names_ref:
            raise RuntimeError("Inconsistent feature names across children")
        X_list.append(vec)
        y_list.append(dataset.labels_per_child[cid])
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y, feat_names_ref or []
