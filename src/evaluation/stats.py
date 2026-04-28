"""
src/evaluation/stats.py

Tests statistiques pour comparer les performances de plusieurs méthodes
sous le même protocole LOSO.

PRINCIPE : pour chaque fold k, on a une métrique m_A^k (méthode A) et
m_B^k (méthode B). On veut savoir si les méthodes diffèrent
significativement. Comme les folds sont APPARIÉS (mêmes enfants test),
on utilise des tests appariés.

Tests fournis :
    - paired_permutation_test : test exact non-paramétrique, robuste pour
      petit n. Recommandé ici (n_folds=24).
    - wilcoxon_signed_rank : alternative paramétrique-libre.
    - Bonferroni : correction pour comparaisons multiples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class TestResult:
    """Résultat d'un test de comparaison entre deux méthodes."""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float            # mean_a - mean_b
    p_value: float
    test_name: str
    n_folds: int

    def to_dict(self) -> dict:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "p_value": self.p_value,
            "test_name": self.test_name,
            "n_folds": self.n_folds,
        }


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: int = 0,
) -> float:
    """Test de permutation apparié exact.

    H0 : la différence moyenne entre A et B est 0.
    Test : pour chaque permutation, on échange aléatoirement les labels
    A/B fold par fold, et on calcule la nouvelle différence moyenne.
    La p-value = fraction de permutations dont la différence est au moins
    aussi extrême que la différence observée.

    Args:
        scores_a, scores_b: vecteurs de métriques fold-par-fold, même shape.
        n_permutations: nombre de permutations (10000 = précision ~0.0001).
        alternative: 'two-sided', 'greater' (A > B), 'less' (A < B).
        seed: pour reproductibilité.

    Returns:
        p-value.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    if scores_a.shape != scores_b.shape:
        raise ValueError(f"Shape mismatch: {scores_a.shape} vs {scores_b.shape}")

    n = len(scores_a)
    diffs = scores_a - scores_b
    observed = diffs.mean()

    rng = np.random.default_rng(seed)
    # Pour chaque permutation, on flip indépendamment le signe de chaque diff
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    perm_means = (signs * diffs).mean(axis=1)

    if alternative == "two-sided":
        p = float(np.mean(np.abs(perm_means) >= np.abs(observed)))
    elif alternative == "greater":
        p = float(np.mean(perm_means >= observed))
    elif alternative == "less":
        p = float(np.mean(perm_means <= observed))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Évite p=0 exactement (artefact d'échantillonnage), retourne 1/n_perm minimum
    return max(p, 1.0 / n_permutations)


def compare_methods(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    method_a: str = "A",
    method_b: str = "B",
    metric: str = "accuracy",
    test: str = "permutation",
    n_permutations: int = 10000,
    seed: int = 0,
) -> TestResult:
    """Compare deux méthodes sur un ensemble de folds appariés.

    Args:
        scores_a, scores_b: vecteurs (n_folds,) de la métrique sur chaque fold.
        method_a, method_b: noms des méthodes pour le rapport.
        metric: nom de la métrique pour le rapport.
        test: 'permutation' ou 'wilcoxon'.

    Returns:
        TestResult avec p-value et stats descriptives.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if test == "permutation":
        p = paired_permutation_test(
            scores_a, scores_b,
            n_permutations=n_permutations,
            alternative="two-sided",
            seed=seed,
        )
        test_name = f"paired_permutation(n={n_permutations})"
    elif test == "wilcoxon":
        try:
            stat, p = wilcoxon(scores_a, scores_b, alternative="two-sided",
                               zero_method="wilcox")
            p = float(p)
        except ValueError:
            # Cas où tous les diffs sont nuls
            p = 1.0
        test_name = "wilcoxon_signed_rank"
    else:
        raise ValueError(f"Unknown test: {test}")

    return TestResult(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        mean_a=float(scores_a.mean()),
        mean_b=float(scores_b.mean()),
        mean_diff=float(scores_a.mean() - scores_b.mean()),
        p_value=p,
        test_name=test_name,
        n_folds=int(len(scores_a)),
    )


def bonferroni_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    """Correction de Bonferroni pour comparaisons multiples.

    Returns:
        (significant_mask, alpha_corrected)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    alpha_corrected = alpha / n if n > 0 else alpha
    return p_values < alpha_corrected, float(alpha_corrected)
