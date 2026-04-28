"""
test_phase3_2_sanity_logreg.py

SANITY CHECK : peut-on apprendre TDAH/CTRL sur 18 enfants train avec un
modèle linéaire trivial ?

Si OUI -> les features contiennent du signal, c'est PatchTST qui a un
         problème spécifique (architecture, training).
Si NON -> les features moyennées ne suffisent pas pour discriminer sur
         si peu d'enfants, et PatchTST collapse pour cette raison
         fondamentale (pas un bug).

PROTOCOLE :
    - Sur le même fold que les diagnostics précédents (test=S01,
      train=18, val=5)
    - Calcule les MOYENNES par enfant des 14 paramètres -> matrice (18, 14)
    - Entraîne LogisticRegression (avec C variable) sur ces 18 features
    - Mesure : train accuracy, val accuracy, test prediction

Compare aussi avec ALL14_MEAN_STD (28 features) pour voir si l'écart-type
intra-enfant aide localement.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src.training.inner_cv import inner_cv_splits
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def child_features_mean(ds, child_ids):
    """Moyenne des 14 paramètres par enfant -> matrice (n_children, 14)."""
    X = []
    y = []
    for cid in child_ids:
        params = ds.params_per_child[cid]   # (n_strokes, 14)
        X.append(params.mean(axis=0))
        y.append(ds.labels_per_child[cid])
    return np.stack(X, axis=0), np.array(y, dtype=int)


def child_features_mean_std(ds, child_ids):
    """Moyenne + écart-type des 14 paramètres -> matrice (n_children, 28)."""
    X = []
    y = []
    for cid in child_ids:
        params = ds.params_per_child[cid]
        m = params.mean(axis=0)
        s = params.std(axis=0, ddof=1) if params.shape[0] > 1 else np.zeros(params.shape[1])
        X.append(np.concatenate([m, s]))
        y.append(ds.labels_per_child[cid])
    return np.stack(X, axis=0), np.array(y, dtype=int)


def evaluate(name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Standardise et essaie plusieurs valeurs de C pour la régression logistique."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"\n--- Feature scheme: {name} (n_features={X_train.shape[1]}) ---")
    print(f"{'C':>6s}  {'train_acc':>10s}  {'val_acc':>8s}  {'test_pred':>10s}  {'test_proba':>12s}")
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=2000,
            random_state=42,
        )
        clf.fit(X_train_s, y_train)
        train_acc = clf.score(X_train_s, y_train)
        val_acc = clf.score(X_val_s, y_val)
        test_pred = int(clf.predict(X_test_s)[0])
        test_proba = float(clf.predict_proba(X_test_s)[0, 1])
        flag = "✓" if test_pred == y_test[0] else "✗"
        print(f"  {C:>6.2f}  {train_acc:>10.3f}  {val_acc:>8.3f}  "
              f"{test_pred:>10d} {flag}  {test_proba:>12.4f}")


def main():
    set_global_seed(42)
    ds = load_processed_dataset(Path("data/processed"))
    print(f"Loaded {len(ds.child_ids)} children")

    # Même fold que les diagnostics précédents
    test_child = "S01"
    train_pool = [c for c in ds.child_ids if c != test_child]
    splits = list(inner_cv_splits(
        train_pool, ds.labels_per_child, n_splits=5, n_repeats=1, seed=42,
    ))
    s0 = splits[0]

    train_ids = s0.train_inner_ids
    val_ids = s0.val_inner_ids
    test_ids = [test_child]

    print(f"Test fold: {test_child} (true label: "
          f"{ds.labels_per_child[test_child]})")
    train_labels = [ds.labels_per_child[c] for c in train_ids]
    val_labels = [ds.labels_per_child[c] for c in val_ids]
    print(f"Train (n={len(train_ids)}): "
          f"CTRL={train_labels.count(0)}, ADHD={train_labels.count(1)}")
    print(f"Val   (n={len(val_ids)}): "
          f"CTRL={val_labels.count(0)}, ADHD={val_labels.count(1)}")

    # Test 1 : moyennes uniquement (14 features)
    X_tr, y_tr = child_features_mean(ds, train_ids)
    X_va, y_va = child_features_mean(ds, val_ids)
    X_te, y_te = child_features_mean(ds, test_ids)
    evaluate("ALL14_MEAN", X_tr, y_tr, X_va, y_va, X_te, y_te)

    # Test 2 : moyennes + écart-types (28 features)
    X_tr, y_tr = child_features_mean_std(ds, train_ids)
    X_va, y_va = child_features_mean_std(ds, val_ids)
    X_te, y_te = child_features_mean_std(ds, test_ids)
    evaluate("ALL14_MEAN_STD", X_tr, y_tr, X_va, y_va, X_te, y_te)

    # Recap
    print(f"\n=== SUMMARY ===")
    print(f"Si train_acc = 100% pour C élevé (faible régularisation) :")
    print(f"  -> les features contiennent un signal, mais peut-être uniquement")
    print(f"     parce que la regression linéaire mémorise (overfit).")
    print(f"  -> Le PatchTST ne devrait pas être pire que ça.")
    print(f"")
    print(f"Si train_acc << 100% même pour C grand :")
    print(f"  -> les features ne sont PAS séparables linéairement.")
    print(f"  -> Le mode collapse de PatchTST est cohérent avec cette limite.")
    print(f"  -> Décision : passer à Exp B (traces brutes) ou réviser le")
    print(f"     setup d'Exp A (peut-être normaliser, fenêtres plus grandes,")
    print(f"     ou abandonner l'idée séquence inter-traits avec L=20).")


if __name__ == "__main__":
    main()
