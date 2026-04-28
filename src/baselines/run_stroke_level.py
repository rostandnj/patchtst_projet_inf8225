"""
src/baselines/run_stroke_level.py

Orchestration LOSO pour baselines CLASSIQUES au niveau trait.

À chaque fold LOSO :
    1. X_train = (n_strokes_train, 14) où chaque trait des 23 enfants train
       est un échantillon, label hérité de l'enfant.
    2. X_test = (n_strokes_test, 14) traits du seul enfant test.
    3. Entraîne classifieur, prédit sur tous les traits test.
    4. Agrège les ~28 prédictions de l'enfant test par moyenne des probas
       -> 1 prédiction au niveau enfant.

Cette baseline est conceptuellement parallèle à ce que fera PatchTST sur
les traces brutes (Exp B) : prédire au niveau trait, agréger au niveau
enfant. Elle donne le point de comparaison apples-to-apples pour mesurer
ce que PatchTST apporte au-delà du sigma-lognormal vu trait par trait.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.baselines.classifiers import fit_classifier, FittedModel
from src.evaluation.loso import loso_splits
from src.evaluation.metrics import (
    stroke_level_metrics,
    child_level_metrics,
    ClassificationMetrics,
)
from src.utils.data_io import ProcessedDataset, stack_strokes


@dataclass
class LosoStrokeResult:
    """Résultats d'une expérience LOSO au niveau trait."""
    method_name: str
    classifier: str

    # Concat de tous les traits test (24 enfants × ~28)
    all_child_ids: np.ndarray            # (n_total_strokes,)
    all_y_true: np.ndarray               # par trait
    all_y_pred: np.ndarray               # par trait
    all_y_proba: np.ndarray              # par trait

    # Métriques agrégées
    stroke_metrics: Optional[ClassificationMetrics] = None
    child_metrics: Optional[ClassificationMetrics] = None

    def to_summary_dict(self) -> dict:
        s, c = self.stroke_metrics, self.child_metrics
        return {
            "method_name": self.method_name,
            "classifier": self.classifier,
            "stroke_accuracy": s.accuracy if s else None,
            "stroke_f1": s.f1 if s else None,
            "child_accuracy": c.accuracy if c else None,
            "child_f1": c.f1 if c else None,
            "child_sensitivity": c.sensitivity if c else None,
            "child_specificity": c.specificity if c else None,
            "child_auc": c.auc if c else None,
        }


def run_stroke_level_loso(
    dataset: ProcessedDataset,
    classifier_name: str,
    method_label: str = "",
    cv_folds: int = 5,
    cv_repeats: int = 1,
    random_state: int = 42,
    verbose: bool = True,
) -> LosoStrokeResult:
    """LOSO classification au niveau trait sur les 14 paramètres."""
    method_name = method_label or f"STROKE_14/{classifier_name}"

    all_child_ids = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold in loso_splits(dataset.child_ids):
        X_train, y_train, _ = stack_strokes(dataset, fold.train_child_ids)
        X_test, y_test, ids_test = stack_strokes(dataset, [fold.test_child_id])

        model: FittedModel = fit_classifier(
            classifier_name=classifier_name,
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds,
            cv_repeats=cv_repeats,
            random_state=random_state,
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        all_child_ids.append(ids_test)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_proba.append(y_proba)

        if verbose:
            true_lbl = int(y_test[0])
            mean_proba = float(np.mean(y_proba))
            child_pred = int(mean_proba >= 0.5)
            ok = "✓" if child_pred == true_lbl else "✗"
            print(f"  [{method_name}] fold {fold.fold_index:2d} test={fold.test_child_id} "
                  f"true={true_lbl} child_pred={child_pred} "
                  f"mean_p={mean_proba:.3f} (n_strokes={len(y_test)}) {ok}")

    all_child_ids_arr = np.concatenate(all_child_ids)
    all_y_true_arr = np.concatenate(all_y_true)
    all_y_pred_arr = np.concatenate(all_y_pred)
    all_y_proba_arr = np.concatenate(all_y_proba)

    stroke_m = stroke_level_metrics(all_y_true_arr, all_y_pred_arr, all_y_proba_arr)
    child_m = child_level_metrics(
        child_ids=all_child_ids_arr,
        y_true_strokes=all_y_true_arr,
        y_pred_strokes=all_y_pred_arr,
        y_score_strokes=all_y_proba_arr,
        aggregation="mean_proba",
    )

    return LosoStrokeResult(
        method_name=method_name,
        classifier=classifier_name,
        all_child_ids=all_child_ids_arr,
        all_y_true=all_y_true_arr,
        all_y_pred=all_y_pred_arr,
        all_y_proba=all_y_proba_arr,
        stroke_metrics=stroke_m,
        child_metrics=child_m,
    )
