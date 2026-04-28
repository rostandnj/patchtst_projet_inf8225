"""
src/baselines/run_child_level.py

Orchestration LOSO pour baselines CLASSIQUES au niveau enfant.

À chaque fold LOSO :
    1. Construit la matrice de features (n_train_children, n_features) et la
       matrice de test (1 enfant, n_features) selon le schéma choisi.
    2. Entraîne le classifieur avec grid-search interne (CV stratifié sur le
       train).
    3. Prédit pour l'enfant test (1 prédiction).

Après les 24 folds, on a 24 prédictions au niveau enfant. On calcule alors
les métriques globales sur ces 24 prédictions :
    - accuracy = nb_correct / 24
    - sensitivité, spécificité, F1, AUC

NOTE IMPORTANTE :
    Au niveau enfant, chaque fold ne donne qu'UNE prédiction. On ne peut
    PAS calculer accuracy "par fold" (elle vaudrait 0 ou 1). Pour comparer
    deux méthodes par tests appariés, on utilise la **probabilité prédite**
    de la classe ADHD (continue) au lieu de la prédiction binaire.
    Cf. src/evaluation/stats.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.baselines.classifiers import fit_classifier, FittedModel
from src.baselines.features import build_child_feature_matrix, FeatureScheme
from src.evaluation.loso import loso_splits
from src.evaluation.metrics import stroke_level_metrics, ClassificationMetrics
from src.utils.data_io import ProcessedDataset


@dataclass
class LosoChildResult:
    """Résultats d'une expérience LOSO au niveau enfant."""
    method_name: str
    feature_scheme: str
    classifier: str

    # Pour chaque fold (= chaque enfant test)
    fold_test_child_ids: list[str]
    fold_y_true: np.ndarray         # (n_folds=24,)
    fold_y_pred: np.ndarray         # (n_folds=24,)
    fold_y_proba: np.ndarray        # (n_folds=24,) P(ADHD=1)
    fold_best_params: list[dict]
    fold_cv_scores: list[float]

    # Agrégés
    metrics: Optional[ClassificationMetrics] = None

    def to_summary_dict(self) -> dict:
        m = self.metrics
        return {
            "method_name": self.method_name,
            "feature_scheme": self.feature_scheme,
            "classifier": self.classifier,
            "accuracy": m.accuracy if m else None,
            "f1": m.f1 if m else None,
            "sensitivity": m.sensitivity if m else None,
            "specificity": m.specificity if m else None,
            "auc": m.auc if m else None,
            "n_folds": len(self.fold_y_true),
            "n_correct": int((self.fold_y_true == self.fold_y_pred).sum()),
        }


def run_child_level_loso(
    dataset: ProcessedDataset,
    feature_scheme: FeatureScheme,
    classifier_name: str,
    method_label: str = "",
    cv_folds: int = 5,
    cv_repeats: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> LosoChildResult:
    """Exécute LOSO complet pour un (scheme, classifier) au niveau enfant."""
    method_name = method_label or f"{feature_scheme}/{classifier_name}"

    fold_test_ids = []
    fold_y_true = []
    fold_y_pred = []
    fold_y_proba = []
    fold_best_params = []
    fold_cv_scores = []

    for fold in loso_splits(dataset.child_ids):
        # Construire X_train, y_train depuis les 23 enfants train
        X_train, y_train, train_feat_names = build_child_feature_matrix(
            dataset, fold.train_child_ids, feature_scheme,
        )
        # X_test : juste 1 enfant
        X_test, y_test, _ = build_child_feature_matrix(
            dataset, [fold.test_child_id], feature_scheme,
        )

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

        fold_test_ids.append(fold.test_child_id)
        fold_y_true.append(int(y_test[0]))
        fold_y_pred.append(int(y_pred[0]))
        fold_y_proba.append(float(y_proba[0]))
        fold_best_params.append(model.best_params)
        fold_cv_scores.append(model.cv_score)

        if verbose:
            ok = "✓" if y_pred[0] == y_test[0] else "✗"
            print(f"  [{method_name}] fold {fold.fold_index:2d} test={fold.test_child_id} "
                  f"true={int(y_test[0])} pred={int(y_pred[0])} "
                  f"p_adhd={float(y_proba[0]):.3f} {ok}")

    fold_y_true_arr = np.array(fold_y_true)
    fold_y_pred_arr = np.array(fold_y_pred)
    fold_y_proba_arr = np.array(fold_y_proba)

    metrics = stroke_level_metrics(
        y_true=fold_y_true_arr,
        y_pred=fold_y_pred_arr,
        y_score=fold_y_proba_arr,
    )

    return LosoChildResult(
        method_name=method_name,
        feature_scheme=feature_scheme,
        classifier=classifier_name,
        fold_test_child_ids=fold_test_ids,
        fold_y_true=fold_y_true_arr,
        fold_y_pred=fold_y_pred_arr,
        fold_y_proba=fold_y_proba_arr,
        fold_best_params=fold_best_params,
        fold_cv_scores=fold_cv_scores,
        metrics=metrics,
    )


def results_to_dataframe(results: list[LosoChildResult]) -> pd.DataFrame:
    """Tableau de synthèse de résultats LOSO."""
    rows = [r.to_summary_dict() for r in results]
    return pd.DataFrame(rows)
