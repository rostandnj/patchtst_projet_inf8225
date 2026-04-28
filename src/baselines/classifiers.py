"""
src/baselines/classifiers.py

Wrappers autour des classifieurs scikit-learn.

Améliorations vs version naive :
    1. Standardisation des features (StandardScaler)
    2. PCA optionnel en haute dimension (>=PCA_TRIGGER_DIM features)
       pour éviter l'overfit avec n_train=23 enfants
    3. class_weight='balanced' pour les classifieurs qui le supportent
    4. RepeatedStratifiedKFold pour estimation CV plus stable
    5. Grilles d'hyperparamètres plus restrictives (évite les configs
       overfit comme C=100, gamma=1)

Important pour le LOSO : à chaque fold externe, on a 23 enfants en train.
La grid-search interne fait du RepeatedStratifiedKFold (5 splits × 3 repeats)
sur ces 23 enfants, choisit les hyperparams, puis entraîne le modèle final.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


AVAILABLE_CLASSIFIERS = ["LDA", "SVM_RBF", "SVM_LINEAR", "KNN", "RF", "GB", "LOGREG"]

# Si n_features >= ce seuil, on insère un PCA pour réduire la dim avant le clf.
# n_train_children est typiquement 23, donc 30 features est déjà beaucoup.
PCA_TRIGGER_DIM = 30


def _make_pipeline(
    classifier_name: str,
    n_features: int,
    n_train_samples: int,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """Construit le pipeline et retourne aussi (pipeline, param_grid).

    Stratégie pour le PCA :
        - On insère un PCA seulement pour les classifieurs basés DISTANCE
          (SVM, KNN, LDA, LOGREG) en haute dim, car ces méthodes souffrent
          de la malédiction de la dimensionnalité avec n petit.
        - Les méthodes basées ARBRES (RF, GB) gèrent nativement la haute
          dim via leur sélection de features bootstrap, donc PAS de PCA.
    """
    # Classifieurs qui bénéficient du PCA en haute dim
    DISTANCE_BASED = {"SVM_RBF", "SVM_LINEAR", "KNN", "LDA", "LOGREG"}
    # Classifieurs qui gèrent la haute dim nativement (pas de PCA)
    TREE_BASED = {"RF", "GB"}

    use_pca = (classifier_name in DISTANCE_BASED) and (n_features >= PCA_TRIGGER_DIM)
    # Cap PCA components à n_train // 3 pour garder du sens
    max_pca_components = max(2, n_train_samples // 3)
    pca_options = sorted({
        min(5, max_pca_components),
        min(8, max_pca_components),
        min(12, max_pca_components),
    })
    pca_options = [p for p in pca_options if p >= 2 and p <= max_pca_components]

    # Étapes : Scaler -> [PCA?] -> Clf
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(random_state=random_state)))

    # Choix du classifieur
    if classifier_name == "LDA":
        clf = LinearDiscriminantAnalysis()
        clf_grid = {"clf__solver": ["svd", "lsqr"]}
    elif classifier_name == "SVM_RBF":
        clf = SVC(kernel="rbf", probability=True, random_state=random_state,
                  class_weight="balanced")
        clf_grid = {
            "clf__C": [0.1, 1.0, 10.0],          # 100 retiré (overfit en haute dim)
            "clf__gamma": ["scale", 0.01, 0.1],  # 1.0 retiré (collapse)
        }
    elif classifier_name == "SVM_LINEAR":
        clf = SVC(kernel="linear", probability=True, random_state=random_state,
                  class_weight="balanced")
        clf_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier()
        clf_grid = {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
        }
    elif classifier_name == "RF":
        clf = RandomForestClassifier(random_state=random_state, n_jobs=1,
                                     class_weight="balanced")
        clf_grid = {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 5],
            "clf__min_samples_leaf": [1, 3],
        }
    elif classifier_name == "GB":
        clf = GradientBoostingClassifier(random_state=random_state)
        clf_grid = {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        }
    elif classifier_name == "LOGREG":
        clf = LogisticRegression(max_iter=2000, random_state=random_state,
                                 class_weight="balanced")
        clf_grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        }
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    steps.append(("clf", clf))
    pipeline = Pipeline(steps)

    grid = dict(clf_grid)
    if use_pca and pca_options:
        grid["pca__n_components"] = pca_options

    return pipeline, grid


@dataclass
class FittedModel:
    """Wrapper autour d'un pipeline scikit-learn entraîné, avec ses meta.

    NOTE IMPORTANTE :
        predict() utilise predict_proba() avec un seuil de 0.5 plutôt que
        le predict() natif du classifieur. Cela garantit la cohérence
        interne (predict et predict_proba sont toujours d'accord) et règle
        un bug connu de scikit-learn avec SVC(probability=True) sur petit
        n_train, où Platt scaling peut produire des probas inversées par
        rapport à decision_function.

        Pour Random Forest et Gradient Boosting, predict() natif est déjà
        equivalent à proba > 0.5, donc aucune différence.
    """
    name: str
    pipeline: Pipeline
    best_params: dict
    cv_score: float
    cv_score_std: float = 0.0      # écart-type CV pour évaluer la stabilité
    used_pca: bool = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction binaire dérivée des probabilités (seuil 0.5).

        Plus rigoureuse que pipeline.predict() pour SVC + petit n.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            return self.pipeline.predict_proba(X)[:, 1]
        if hasattr(self.pipeline.named_steps["clf"], "decision_function"):
            scores = self.pipeline.decision_function(X)
            return 1.0 / (1.0 + np.exp(-scores))
        return self.pipeline.predict(X).astype(float)


def fit_classifier(
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_repeats: int = 3,
    random_state: int = 42,
    n_jobs: int = 1,
) -> FittedModel:
    """Fit avec grid-search en CV interne (RepeatedStratifiedKFold)."""
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    pipeline, grid = _make_pipeline(
        classifier_name,
        n_features=n_features,
        n_train_samples=n_samples,
        random_state=random_state,
    )

    # Adapte k si trop peu d'échantillons par classe
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    k = min(cv_folds, n_pos, n_neg)
    if k < 2:
        # Pas assez d'exemples pour CV : fit direct
        pipeline.fit(X_train, y_train)
        return FittedModel(
            name=classifier_name,
            pipeline=pipeline,
            best_params={},
            cv_score=float("nan"),
            cv_score_std=float("nan"),
            used_pca="pca" in dict(pipeline.steps),
        )

    # CV interne : RepeatedStratifiedKFold pour estimation stable
    if cv_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=k, n_repeats=cv_repeats, random_state=random_state,
        )
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        pipeline,
        grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    # Recup l'écart-type du best score (utile pour évaluer la stabilité)
    best_idx = search.best_index_
    cv_std = float(search.cv_results_["std_test_score"][best_idx])

    return FittedModel(
        name=classifier_name,
        pipeline=search.best_estimator_,
        best_params={k: v for k, v in search.best_params_.items()},
        cv_score=float(search.best_score_),
        cv_score_std=cv_std,
        used_pca="pca" in dict(search.best_estimator_.steps),
    )
