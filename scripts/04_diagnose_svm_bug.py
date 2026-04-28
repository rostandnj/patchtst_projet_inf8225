"""
scripts/04_diagnose_svm_bug.py

Investigation du bug observé sur SVM_RBF + RICH_STATS / ALL14_MEAN_STD :
    - accuracy ≈ 4% (1/24)
    - AUC ≈ 0.9
    -> le modèle ordonne bien les enfants mais inverse les prédictions

On examine fold par fold :
    1. Quels hyperparams sont sélectionnés par la grid-search interne ?
    2. Comment se distribue la proba prédite vs label vrai ?
    3. La grid-search marche-t-elle correctement (CV interne) ?
    4. Sur le test, on prédit plus souvent une seule classe ?

Usage:
    python scripts/04_diagnose_svm_bug.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.baselines.features import build_child_feature_matrix
from src.evaluation.loso import loso_splits
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def diagnose_one_method(dataset, scheme: str, classifier_name: str, seed: int = 42):
    """Exécute le LOSO en gardant tous les détails utiles au diag."""
    print(f"\n{'=' * 72}")
    print(f"DIAGNOSING: {scheme} / {classifier_name}")
    print(f"{'=' * 72}")

    rows = []
    for fold in loso_splits(dataset.child_ids):
        X_train, y_train, train_names = build_child_feature_matrix(
            dataset, fold.train_child_ids, scheme,
        )
        X_test, y_test, _ = build_child_feature_matrix(
            dataset, [fold.test_child_id], scheme,
        )

        # Pipeline standard
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
        ])
        grid = {
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", 0.01, 0.1, 1.0],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1)
        gs.fit(X_train, y_train)

        best_pipe = gs.best_estimator_
        y_pred_test = int(best_pipe.predict(X_test)[0])
        y_proba_test = float(best_pipe.predict_proba(X_test)[0, 1])
        y_dec_test = float(best_pipe.decision_function(X_test)[0])

        # Diagnostic supplémentaire : quelle est la prédiction sur le train ?
        y_pred_train = best_pipe.predict(X_train)
        train_acc = float((y_pred_train == y_train).mean())
        # Distribution des prédictions sur train
        n_train_pred_1 = int((y_pred_train == 1).sum())
        n_train_true_1 = int((y_train == 1).sum())

        rows.append({
            "fold": fold.fold_index,
            "test_child": fold.test_child_id,
            "y_true": int(y_test[0]),
            "y_pred": y_pred_test,
            "y_proba_1": y_proba_test,
            "y_dec_func": y_dec_test,
            "best_C": best_pipe.named_steps["clf"].C,
            "best_gamma": best_pipe.named_steps["clf"].gamma,
            "cv_score": gs.best_score_,
            "train_acc": train_acc,
            "n_train_pred_1": n_train_pred_1,
            "n_train_true_1": n_train_true_1,
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
        })

    df = pd.DataFrame(rows)

    print(f"\nGlobal stats:")
    print(f"  test accuracy: {(df['y_true'] == df['y_pred']).mean():.4f}")
    print(f"  mean train accuracy (overfit indicator): {df['train_acc'].mean():.4f}")
    print(f"  mean CV-internal score: {df['cv_score'].mean():.4f}")
    print(f"  n_features: {df['n_features'].iloc[0]}, n_train_samples: {df['n_train_samples'].iloc[0]}")

    # Question 1 : que prédit le modèle sur le test ?
    n_pred_0 = (df["y_pred"] == 0).sum()
    n_pred_1 = (df["y_pred"] == 1).sum()
    print(f"\n  predictions distribution on test: {n_pred_0} CTRL, {n_pred_1} ADHD")
    n_true_0 = (df["y_true"] == 0).sum()
    n_true_1 = (df["y_true"] == 1).sum()
    print(f"  ground truth: {n_true_0} CTRL, {n_true_1} ADHD")

    # Question 2 : la proba prédite est-elle correlée au vrai label ?
    proba_for_ctrl = df.loc[df["y_true"] == 0, "y_proba_1"]
    proba_for_adhd = df.loc[df["y_true"] == 1, "y_proba_1"]
    print(f"\n  P(ADHD=1) for CTRL children: mean={proba_for_ctrl.mean():.3f} "
          f"(should be < 0.5)")
    print(f"  P(ADHD=1) for ADHD children: mean={proba_for_adhd.mean():.3f} "
          f"(should be > 0.5)")

    # Question 3 : le modèle est-il dégénéré ? (toujours la même prédiction)
    if n_pred_0 == 24 or n_pred_1 == 24:
        print(f"  ⚠ MODEL DEGENERATE: predicts only one class")

    # Question 4 : quels hyperparams sont sélectionnés ?
    print(f"\n  Selected hyperparams (best_C, best_gamma):")
    hp_counts = df.groupby(["best_C", "best_gamma"]).size().sort_values(ascending=False)
    print(hp_counts.to_string())

    # Question 5 : train accuracy vs test accuracy par fold
    overfit = df["train_acc"] - (df["y_true"] == df["y_pred"]).astype(int)
    print(f"\n  Overfitting indicator (train_acc - test_acc per fold):")
    print(f"    mean: {overfit.mean():.3f}")

    # Question 6 : est-ce que decision_function et proba sont consistents ?
    print(f"\n  Decision function vs proba consistency:")
    pos_dec_pos_proba = ((df["y_dec_func"] > 0) & (df["y_proba_1"] > 0.5)).sum()
    print(f"    cases where decision>0 AND proba>0.5: {pos_dec_pos_proba}/24")

    return df


def main():
    parser = argparse.ArgumentParser(description="Diagnose SVM-RBF bug")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/diagnosis"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_processed_dataset(args.processed_dir)
    print(f"Loaded {len(dataset.child_ids)} children, {dataset.total_strokes} strokes")

    # Cas problématiques
    problematic_cases = [
        ("RICH_STATS", "SVM_RBF"),
        ("ALL14_MEAN_STD", "SVM_RBF"),
        ("ALL14_MEAN", "SVM_RBF"),     # cas intermédiaire (acc=25%)
    ]
    # Cas qui marchent bien pour comparaison
    working_cases = [
        ("ALL14_MEAN_STD", "RF"),
        ("RICH_STATS", "GB"),
    ]

    all_dfs = []
    for scheme, clf in problematic_cases + working_cases:
        df = diagnose_one_method(dataset, scheme, clf, seed=args.seed)
        df["method"] = f"{scheme}/{clf}"
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    out_path = args.out_dir / "svm_diagnosis.csv"
    full.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
