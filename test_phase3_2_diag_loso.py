"""
test_phase3_2_diag_loso.py

Diagnostic rapide : mini-LOSO sur 5 enfants pour vérifier que PatchTST
ne souffre pas d'un overfitting/data leakage majeur avant de lancer
la Phase 3.3 complète (~5h).

Pour chaque enfant test ∈ [S01, S02, S03, S04, S05]:
    train_pool = 23 autres enfants
    val_pool   = 4 enfants stratifiés (1 split)
    train      = 19 enfants
    Entraîne PatchTSTClassifier supervisé
    Prédit sur l'enfant test
    Compare au vrai label

Affiche un résumé : accuracy 5/5, distribution des probas vs labels,
warning si signe d'overfitting.

Durée estimée : ~5-10 min sur M5 Pro.
"""

from __future__ import annotations

# Active fallback MPS avant tout import torch
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_a_datasets import SequenceChildDataset
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.config import FullExpConfig, OptimConfig, TrainingConfig
from src.training.device import get_device
from src.training.inner_cv import inner_cv_splits
from src.training.supervised import train_supervised, predict_proba
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def run_one_fold(ds, test_child_id, device, verbose=False):
    """Un fold mini-LOSO complet (1 inner split, pas 15)."""
    train_pool = [c for c in ds.child_ids if c != test_child_id]

    # On prend juste le PREMIER split de l'inner CV (sur 15 possibles)
    splits = list(inner_cv_splits(
        train_pool, ds.labels_per_child, n_splits=5, n_repeats=1, seed=42,
    ))
    s0 = splits[0]

    cfg = FullExpConfig(
        exp_name=f"diag_test={test_child_id}",
        optim=OptimConfig(lr=1e-3, weight_decay=0.01, warmup_epochs=5),
        training=TrainingConfig(
            max_epochs=50,
            batch_size=8,
            early_stopping_patience=10,
            use_class_weight=True,
            use_trial_subsampling=True,
            subsample_strategy="stratified",
            seed=42,
        ),
        seq_len=20, n_channels=14,
        patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.2,
    )

    train_ds = SequenceChildDataset(
        ds, s0.train_inner_ids, seq_len=20,
        is_train=True, subsample_strategy="stratified", seed=42,
    )
    val_ds = SequenceChildDataset(
        ds, s0.val_inner_ids, seq_len=20,
        is_train=False, subsample_strategy="stratified", seed=43,
    )
    test_ds = SequenceChildDataset(
        ds, [test_child_id], seq_len=20,
        is_train=False, subsample_strategy="stratified",
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    patchtst_cfg = PatchTSTConfig(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len,
        patch_len=cfg.patch_len, stride=cfg.stride,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        d_ff=cfg.d_ff, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
    )
    model = PatchTSTClassifier(patchtst_cfg, n_classes=2)

    history, best_state = train_supervised(
        model, train_loader, val_loader, device, cfg,
        verbose=verbose, log_every=10,
    )

    # Inférence sur enfant test
    model.load_state_dict(best_state)
    test_proba = float(predict_proba(model, test_loader, device)[0])
    test_pred = int(test_proba >= 0.5)
    test_true = ds.labels_per_child[test_child_id]

    return {
        "test_child": test_child_id,
        "true": test_true,
        "pred": test_pred,
        "proba_adhd": test_proba,
        "correct": test_pred == test_true,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "best_val_acc": history.best_val_acc,
        "n_epochs_run": history.n_epochs_run,
        "stopped_early": history.stopped_early,
    }


def main():
    set_global_seed(42)
    device = get_device()
    print(f"=== Phase 3.2 mini-LOSO diagnostic ===")
    print(f"Device: {device}")

    ds = load_processed_dataset(Path("data/processed"))
    print(f"Loaded {len(ds.child_ids)} children")

    # On prend les 5 premiers enfants comme tests
    test_subset = ds.child_ids[:5]
    print(f"Testing folds: {test_subset}")
    print(f"True labels:   {[ds.labels_per_child[c] for c in test_subset]}")
    print(f"  (0=CTRL, 1=ADHD)\n")

    results = []
    t_start = time.time()
    for cid in test_subset:
        print(f"--- Fold test={cid} ---")
        t0 = time.time()
        r = run_one_fold(ds, cid, device, verbose=False)
        dt = time.time() - t0
        ok = "✓" if r["correct"] else "✗"
        print(f"  true={r['true']} pred={r['pred']} "
              f"proba_adhd={r['proba_adhd']:.4f}  {ok}  "
              f"(best_ep={r['best_epoch']}, val_loss={r['best_val_loss']:.4f}, "
              f"val_acc={r['best_val_acc']:.3f}, "
              f"epochs_run={r['n_epochs_run']}, took={dt:.1f}s)")
        results.append(r)

    total_dt = time.time() - t_start

    # Synthèse
    print(f"\n=== Summary ===")
    n_correct = sum(r["correct"] for r in results)
    print(f"Mini-LOSO accuracy: {n_correct}/5 = {n_correct/5:.1%}")
    print(f"Total time: {total_dt:.1f}s ({total_dt/5:.1f}s per fold)")
    print(f"Estimated full LOSO time (24 folds × 15 inner splits): "
          f"~{(total_dt / 5) * 24 * 15 / 3600:.1f}h "
          f"(this approximation only — actual run uses different optims)")

    # Diagnostic
    print(f"\n--- Diagnostic ---")
    probas = [r["proba_adhd"] for r in results]
    extreme_count = sum(1 for p in probas if p < 0.05 or p > 0.95)
    if extreme_count >= 4:
        print(f"  ⚠ Model is making EXTREME predictions ({extreme_count}/5 "
              f"in [0,0.05] or [0.95,1])")
        print(f"     This suggests overconfidence/overfitting.")

    val_acc_perfect = sum(1 for r in results if r["best_val_acc"] >= 0.99)
    if val_acc_perfect >= 4:
        print(f"  ⚠ Val accuracy is perfect (≥0.99) in {val_acc_perfect}/5 folds.")
        print(f"     With only 4-5 val samples, this is suspicious.")

    early_convergence = sum(1 for r in results if r["best_epoch"] < 5)
    if early_convergence >= 4:
        print(f"  ⚠ Best epoch < 5 in {early_convergence}/5 folds — too fast.")

    if n_correct >= 4:
        print(f"  ✓ Model generalizes correctly ({n_correct}/5). Safe to launch full LOSO.")
    elif n_correct == 3:
        print(f"  ~ Marginal (3/5 = 60%). Need full LOSO to conclude.")
    elif n_correct <= 2:
        print(f"  ⚠ Model FAILS to generalize ({n_correct}/5). DO NOT launch full LOSO.")
        print(f"     Possible fixes: increase dropout, decrease model size, "
              f"add regularization.")

    print(f"\n--- Per-fold details ---")
    for r in results:
        ok = "✓" if r["correct"] else "✗"
        tag = "ADHD" if r["true"] == 1 else "CTRL"
        print(f"  {r['test_child']} [{tag}]: P(ADHD)={r['proba_adhd']:.4f} -> "
              f"pred={'ADHD' if r['pred']==1 else 'CTRL'} {ok}")


if __name__ == "__main__":
    main()
