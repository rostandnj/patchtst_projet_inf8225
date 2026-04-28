"""
scripts/03_run_baselines.py

Phase 2 : exécute toutes les baselines classiques en LOSO et exporte les
résultats.

Configuration de l'étude :
    Niveau ENFANT (n=24) :
        - FACI4_MEAN  + {LDA, SVM_RBF, KNN}        # reproduction Faci 2021
        - ALL14_MEAN  + {LDA, SVM_RBF, KNN, RF}    # extension à 14 features
        - ALL14_MEAN_STD + {LDA, SVM_RBF, RF, GB}  # variabilité intra-enfant
        - RICH_STATS  + {SVM_RBF, RF, GB, LOGREG}  # baseline temporelle

    Niveau TRAIT (n=663) avec agrégation enfant par moyenne de probas :
        - STROKE_14   + {LDA, SVM_RBF, RF, GB}     # parallèle à PatchTST

Pour chaque combinaison : 24 folds LOSO, métriques au niveau enfant.
Tests de permutation pour comparer les méthodes deux à deux.

Exports (results/baselines/) :
    summary.csv           : table récapitulative
    detailed.parquet      : prédictions par fold
    pairwise_tests.csv    : p-values des comparaisons
    config.json           : config de l'expérience pour reproductibilité

Usage :
    python scripts/03_run_baselines.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.run_child_level import (
    run_child_level_loso,
    LosoChildResult,
)
from src.baselines.run_stroke_level import (
    run_stroke_level_loso,
    LosoStrokeResult,
)
from src.evaluation.stats import compare_methods, bonferroni_correct
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


# Liste des expériences (scheme, classifier) au niveau enfant
# NOTE: SVM-RBF retiré de RICH_STATS car inadapté à n=23, 140 features (curse
# of dimensionality). Voir résultats v1/v2: il s'effondre à <10% accuracy.
# Les méthodes tree-based (RF, GB) gèrent nativement la haute dim.
CHILD_LEVEL_EXPERIMENTS = [
    # Reproduction Faci et al. 2021
    ("FACI4_MEAN", "LDA"),
    ("FACI4_MEAN", "SVM_RBF"),
    ("FACI4_MEAN", "KNN"),

    # Extension : 14 paramètres (moyenne par enfant)
    ("ALL14_MEAN", "LDA"),
    ("ALL14_MEAN", "SVM_RBF"),
    ("ALL14_MEAN", "KNN"),
    ("ALL14_MEAN", "RF"),

    # 14 paramètres + écart-type intra-enfant (28 features)
    ("ALL14_MEAN_STD", "LDA"),
    ("ALL14_MEAN_STD", "SVM_RBF"),
    ("ALL14_MEAN_STD", "RF"),
    ("ALL14_MEAN_STD", "GB"),

    # Statistiques riches (~140 features) - baseline "temporelle classique"
    # SVM-RBF retiré (n=23 vs 140 features = curse of dim)
    ("RICH_STATS", "RF"),
    ("RICH_STATS", "GB"),
    ("RICH_STATS", "LOGREG"),
]

# Niveau trait
STROKE_LEVEL_EXPERIMENTS = [
    "LDA",
    "SVM_RBF",
    "RF",
    "GB",
]


def main():
    parser = argparse.ArgumentParser(description="Run all classical baselines (LOSO)")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="K pour la grid-search interne")
    parser.add_argument("--cv-repeats", type=int, default=3,
                        help="Nombre de répétitions de la CV interne (child-level)")
    parser.add_argument("--quiet", action="store_true",
                        help="N'affiche pas les détails fold-par-fold")
    args = parser.parse_args()

    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    # 1. Chargement
    print(f"Loading processed dataset from {args.processed_dir}...")
    dataset = load_processed_dataset(args.processed_dir)
    print(f"  {len(dataset.child_ids)} children, "
          f"{dataset.total_strokes} strokes total, "
          f"{len(dataset.param_names)} params per stroke")

    # 2. Niveau enfant
    print("\n" + "=" * 72)
    print("CHILD-LEVEL BASELINES (one prediction per child after aggregation)")
    print("=" * 72)
    child_results: list[LosoChildResult] = []
    for scheme, clf in CHILD_LEVEL_EXPERIMENTS:
        method_name = f"{scheme}/{clf}"
        print(f"\n--- Running {method_name} ---")
        try:
            res = run_child_level_loso(
                dataset=dataset,
                feature_scheme=scheme,
                classifier_name=clf,
                method_label=method_name,
                cv_folds=args.cv_folds,
                cv_repeats=args.cv_repeats,
                random_state=args.seed,
                verbose=verbose,
            )
            child_results.append(res)
            m = res.metrics
            print(f"  -> acc={m.accuracy:.4f} ({m.n_correct}/24) "
                  f"sens={m.sensitivity:.3f} spec={m.specificity:.3f} "
                  f"f1={m.f1:.3f} auc={m.auc:.3f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    # 3. Niveau trait
    print("\n" + "=" * 72)
    print("STROKE-LEVEL BASELINES (per-stroke pred, mean-proba aggregation)")
    print("=" * 72)
    stroke_results: list[LosoStrokeResult] = []
    for clf in STROKE_LEVEL_EXPERIMENTS:
        method_name = f"STROKE_14/{clf}"
        print(f"\n--- Running {method_name} ---")
        try:
            res = run_stroke_level_loso(
                dataset=dataset,
                classifier_name=clf,
                method_label=method_name,
                cv_folds=args.cv_folds,
                random_state=args.seed,
                verbose=verbose,
            )
            stroke_results.append(res)
            sm, cm = res.stroke_metrics, res.child_metrics
            print(f"  -> stroke_acc={sm.accuracy:.4f}  "
                  f"child_acc={cm.accuracy:.4f} ({cm.n_correct}/24) "
                  f"sens={cm.sensitivity:.3f} spec={cm.specificity:.3f} "
                  f"f1={cm.f1:.3f} auc={cm.auc:.3f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    # 4. Tableau de synthèse
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    summary_rows = []
    for r in child_results:
        d = r.to_summary_dict()
        d["level"] = "child"
        summary_rows.append(d)
    for r in stroke_results:
        d = r.to_summary_dict()
        d["level"] = "stroke->child"
        # Aligne les noms de colonnes : on prend la version niveau enfant
        d["accuracy"] = d.pop("child_accuracy")
        d["f1"] = d.pop("child_f1")
        d["sensitivity"] = d.pop("child_sensitivity")
        d["specificity"] = d.pop("child_specificity")
        d["auc"] = d.pop("child_auc")
        d["feature_scheme"] = "STROKE_14"
        summary_rows.append(d)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("accuracy", ascending=False)
    summary_path = args.out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print()
    print(summary_df.to_string(index=False))

    # 5. Détaillé : prédictions fold par fold (utile pour reproductibilité)
    detail_rows = []
    for r in child_results:
        for i, cid in enumerate(r.fold_test_child_ids):
            detail_rows.append({
                "method": r.method_name,
                "level": "child",
                "fold": i,
                "test_child": cid,
                "y_true": int(r.fold_y_true[i]),
                "y_pred": int(r.fold_y_pred[i]),
                "y_proba": float(r.fold_y_proba[i]),
                "best_params": json.dumps(r.fold_best_params[i]),
                "cv_score": r.fold_cv_scores[i],
            })
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_path = args.out_dir / "detailed_child_predictions.parquet"
        detail_df.to_parquet(detail_path, index=False)
        detail_df.to_csv(args.out_dir / "detailed_child_predictions.csv", index=False)
        print(f"Wrote {detail_path}")

    # Pour stroke-level on sauvegarde aussi les prédictions traits
    stroke_detail_rows = []
    for r in stroke_results:
        for i, cid in enumerate(r.all_child_ids):
            stroke_detail_rows.append({
                "method": r.method_name,
                "child_id": cid,
                "y_true": int(r.all_y_true[i]),
                "y_pred": int(r.all_y_pred[i]),
                "y_proba": float(r.all_y_proba[i]),
            })
    if stroke_detail_rows:
        sdf = pd.DataFrame(stroke_detail_rows)
        spath = args.out_dir / "detailed_stroke_predictions.parquet"
        sdf.to_parquet(spath, index=False)
        print(f"Wrote {spath}")

    # 6. Tests appariés : on compare la PROBA prédite par enfant entre méthodes
    print("\n--- Pairwise permutation tests (on child-level predicted probas) ---")
    print("Note: We compare predicted probabilities (continuous) rather than "
          "binary predictions to gain statistical power on n=24 folds.")

    # On regroupe TOUS les résultats child-level (les stroke-level ont leur
    # proba child via mean-aggregation, on les ajoute aussi)
    all_method_probas: dict[str, np.ndarray] = {}
    all_method_truths: dict[str, np.ndarray] = {}
    for r in child_results:
        all_method_probas[r.method_name] = r.fold_y_proba
        all_method_truths[r.method_name] = r.fold_y_true
    for r in stroke_results:
        # On extrait la proba enfant en moyennant les probas trait par enfant
        ids_unique = sorted(set(r.all_child_ids.tolist()))
        proba_per_child = []
        truth_per_child = []
        for cid in ids_unique:
            mask = r.all_child_ids == cid
            proba_per_child.append(float(np.mean(r.all_y_proba[mask])))
            truth_per_child.append(int(r.all_y_true[mask][0]))
        all_method_probas[r.method_name] = np.array(proba_per_child)
        all_method_truths[r.method_name] = np.array(truth_per_child)

    # Pour les tests appariés, on a besoin que toutes les méthodes aient le
    # même ordre d'enfants. Sécurité : on vérifie que les y_true sont
    # identiques pour toutes les méthodes (ils devraient l'être : LOSO
    # déterministe, mêmes 24 enfants, mêmes labels).
    methods = list(all_method_probas.keys())
    if methods:
        ref_truth = all_method_truths[methods[0]]
        for m in methods[1:]:
            if not np.array_equal(all_method_truths[m], ref_truth):
                print(f"  WARNING: method {m} has different label order, skipping pairwise tests")
                methods = [methods[0]]
                break

    pairwise_rows = []
    for i, m_a in enumerate(methods):
        for m_b in methods[i + 1:]:
            # Pour chaque fold : "score de discrimination" = proba(ADHD) si TDAH
            # ou 1-proba(ADHD) si CTRL. Plus c'est haut, mieux la méthode prédit.
            truth = all_method_truths[m_a]
            score_a = np.where(truth == 1, all_method_probas[m_a],
                               1 - all_method_probas[m_a])
            score_b = np.where(truth == 1, all_method_probas[m_b],
                               1 - all_method_probas[m_b])
            res = compare_methods(
                score_a, score_b,
                method_a=m_a, method_b=m_b,
                metric="confidence_in_truth",
                test="permutation",
                n_permutations=10000,
                seed=args.seed,
            )
            pairwise_rows.append(res.to_dict())

    if pairwise_rows:
        pw_df = pd.DataFrame(pairwise_rows)
        # Bonferroni
        sig, alpha_corr = bonferroni_correct(pw_df["p_value"].values, alpha=0.05)
        pw_df["significant_bonferroni_005"] = sig
        pw_df = pw_df.sort_values("p_value")
        pw_path = args.out_dir / "pairwise_tests.csv"
        pw_df.to_csv(pw_path, index=False)
        print(f"\nWrote {pw_path} ({len(pw_df)} comparisons, "
              f"alpha_bonferroni={alpha_corr:.5f})")
        # Affiche top 10 plus significatives
        print("\nTop 10 most significant pairwise differences:")
        print(pw_df.head(10).to_string(index=False))

    # 7. Config pour reproductibilité
    config = {
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "child_level_experiments": CHILD_LEVEL_EXPERIMENTS,
        "stroke_level_experiments": STROKE_LEVEL_EXPERIMENTS,
        "n_children": len(dataset.child_ids),
        "n_strokes_total": dataset.total_strokes,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nWrote {args.out_dir / 'config.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
