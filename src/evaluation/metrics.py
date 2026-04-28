"""
src/evaluation/metrics.py

Métriques de classification calculées à deux granularités :

1. Niveau TRAIT (stroke-level) : chaque trait est un échantillon indépendant.
   Métrique = accuracy/F1/etc. sur les ~28 prédictions de l'enfant test.

2. Niveau ENFANT (child-level) : on agrège les prédictions des traits d'un
   enfant en une prédiction unique au niveau enfant (vote majoritaire ou
   moyenne des probabilités), puis métrique sur la prédiction unique.

C'est la métrique au niveau ENFANT qui est cliniquement pertinente et
comparable au papier de Faci et al. (qui rapporte 91.67% au niveau enfant).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


@dataclass
class ClassificationMetrics:
    """Métriques agrégées pour une classification binaire (0=CTRL, 1=ADHD)."""
    accuracy: float
    f1: float                     # F1 sur la classe positive (ADHD)
    sensitivity: float            # = recall_ADHD = TP / (TP + FN)
    specificity: float            # = TN / (TN + FP)
    auc: float = float("nan")     # AUC-ROC, NaN si scores non disponibles
    n_samples: int = 0
    n_correct: int = 0
    confusion: tuple = (0, 0, 0, 0)  # (TN, FP, FN, TP)

    def to_dict(self) -> dict:
        return asdict(self)


def stroke_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    """Métriques au niveau trait (un sample = un trait).

    Args:
        y_true: labels vrais (0/1), shape (n_strokes,).
        y_pred: prédictions (0/1).
        y_score: scores/probas pour l'AUC (optionnel). Si fourni, doit être
            la probabilité de la classe 1 (ADHD).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    # Spécificité = recall sur la classe 0
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    auc = float("nan")
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return ClassificationMetrics(
        accuracy=float(acc),
        f1=float(f1),
        sensitivity=float(sens),
        specificity=float(spec),
        auc=float(auc),
        n_samples=int(len(y_true)),
        n_correct=int((y_true == y_pred).sum()),
        confusion=(int(tn), int(fp), int(fn), int(tp)),
    )


def aggregate_to_child_level(
    child_ids: np.ndarray,
    y_true_strokes: np.ndarray,
    y_pred_strokes: np.ndarray,
    y_score_strokes: Optional[np.ndarray] = None,
    aggregation: str = "mean_proba",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Agrège les prédictions niveau trait en prédictions niveau enfant.

    Args:
        child_ids: ID enfant pour chaque trait, shape (n_strokes,).
        y_true_strokes: labels vrais au niveau trait. Tous les traits d'un
            enfant ont le même label (vérifié).
        y_pred_strokes: prédictions au niveau trait (0/1).
        y_score_strokes: probas P(ADHD=1) au niveau trait (optionnel).
        aggregation: stratégie d'agrégation:
            - 'mean_proba': moyenne des probas, seuil 0.5 (recommandé)
            - 'majority_vote': vote majoritaire des prédictions binaires

    Returns:
        unique_ids: IDs des enfants (uniques).
        y_true_child: label vrai par enfant (1 par enfant).
        y_pred_child: prédiction par enfant après agrégation.
        y_score_child: score moyen par enfant (NaN si y_score_strokes None).
    """
    if aggregation not in ("mean_proba", "majority_vote"):
        raise ValueError(f"Unknown aggregation: {aggregation}")

    child_ids = np.asarray(child_ids)
    y_true_strokes = np.asarray(y_true_strokes).astype(int)
    y_pred_strokes = np.asarray(y_pred_strokes).astype(int)

    unique_ids = np.array(sorted(np.unique(child_ids)))
    y_true_child = np.zeros(len(unique_ids), dtype=int)
    y_pred_child = np.zeros(len(unique_ids), dtype=int)
    y_score_child = np.full(len(unique_ids), np.nan)

    for i, cid in enumerate(unique_ids):
        mask = child_ids == cid
        labels = y_true_strokes[mask]
        # Sanity check : un enfant a un seul label
        unique_lbls = np.unique(labels)
        if len(unique_lbls) != 1:
            raise ValueError(f"Child {cid} has multiple labels: {unique_lbls}")
        y_true_child[i] = int(unique_lbls[0])

        if y_score_strokes is not None:
            mean_score = float(np.mean(y_score_strokes[mask]))
            y_score_child[i] = mean_score
            if aggregation == "mean_proba":
                y_pred_child[i] = int(mean_score >= 0.5)

        if aggregation == "majority_vote" or y_score_strokes is None:
            preds = y_pred_strokes[mask]
            # Tie-break : si 50/50, on regarde mean_score si dispo, sinon 0
            n1 = int((preds == 1).sum())
            n0 = int((preds == 0).sum())
            if n1 > n0:
                y_pred_child[i] = 1
            elif n0 > n1:
                y_pred_child[i] = 0
            else:
                y_pred_child[i] = int(y_score_child[i] >= 0.5) if not np.isnan(y_score_child[i]) else 0

    return unique_ids, y_true_child, y_pred_child, y_score_child


def child_level_metrics(
    child_ids: np.ndarray,
    y_true_strokes: np.ndarray,
    y_pred_strokes: np.ndarray,
    y_score_strokes: Optional[np.ndarray] = None,
    aggregation: str = "mean_proba",
) -> ClassificationMetrics:
    """Métriques au niveau enfant après agrégation des prédictions traits."""
    _, y_true_child, y_pred_child, y_score_child = aggregate_to_child_level(
        child_ids, y_true_strokes, y_pred_strokes, y_score_strokes, aggregation
    )
    score = y_score_child if y_score_strokes is not None else None
    return stroke_level_metrics(y_true_child, y_pred_child, score)
