"""
test_phase3_2_diag_grid.py

Diagnostic comparatif : teste plusieurs configurations PatchTST sur le
même mini-LOSO 5-folds pour identifier ce qui résout le mode collapse.

CONFIGURATIONS TESTÉES :
    baseline       : config actuelle (n_layers=2, d_model=64, dropout=0.2)
    smaller        : n_layers=1, d_model=32 (~30k params au lieu de 105k)
    regularized    : dropout=0.4, weight_decay=0.05
    label_smooth   : label_smoothing=0.1 dans CrossEntropyLoss
    slow_lr        : lr=3e-4 + warmup=10 + max_epochs=80
    all_fixes      : combine smaller + regularized + label_smooth + slow_lr

POUR CHAQUE CONFIG :
    - Mini-LOSO sur S01..S05 (1 inner split par fold)
    - Tracking : accuracy, distribution des probas, mode collapse indicator

CRITÈRE DE SUCCÈS :
    - Probas variées (pas toutes à 0.0 ou 1.0)
    - Accuracy >= 3/5 sur le sous-set test (signe de discrimination réelle,
      pas de prédiction constante)

Durée estimée : ~1 minute total sur M5 Pro.
"""

from __future__ import annotations

# Active fallback MPS avant tout import torch
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_a_datasets import SequenceChildDataset
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.config import FullExpConfig, OptimConfig, TrainingConfig
from src.training.device import get_device
from src.training.early_stopping import EarlyStopping
from src.training.inner_cv import inner_cv_splits
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.training.supervised import _epoch_loop, _compute_class_weights, predict_proba
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


# -------------------------------------------------------------------
# Variant of train_supervised that supports label smoothing
# -------------------------------------------------------------------
def train_supervised_with_smoothing(
    model, train_loader, val_loader, device, config, label_smoothing=0.0,
):
    """Comme train_supervised mais avec label_smoothing en option."""
    model = model.to(device)
    optimizer = build_optimizer(model, config.optim)
    scheduler = build_scheduler(optimizer, config.optim, config.training.max_epochs)

    if config.training.use_class_weight:
        class_weights = _compute_class_weights(train_loader, device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    early = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta,
        mode="min",
    )

    best_state = {}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    n_epochs_run = 0

    for epoch in range(config.training.max_epochs):
        train_loss, train_acc = _epoch_loop(
            model, train_loader, device, criterion,
            optimizer=optimizer, grad_clip=config.optim.grad_clip,
        )
        val_loss, val_acc = _epoch_loop(
            model, val_loader, device, criterion, optimizer=None,
        )
        n_epochs_run = epoch + 1

        if early.is_best(val_loss):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_acc = val_acc

        if early.step(val_loss, epoch):
            break
        scheduler.step()

    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "n_epochs_run": n_epochs_run,
    }


# -------------------------------------------------------------------
# Configurations à tester
# -------------------------------------------------------------------
CONFIGS = {
    "baseline": {
        "label_smoothing": 0.0,
        "patchtst": dict(d_model=64, n_heads=4, n_layers=2, d_ff=128,
                         dropout=0.2, attn_dropout=0.1),
        "optim": dict(lr=1e-3, weight_decay=0.01, warmup_epochs=5),
        "max_epochs": 50, "patience": 10,
    },
    "smaller": {
        "label_smoothing": 0.0,
        "patchtst": dict(d_model=32, n_heads=4, n_layers=1, d_ff=64,
                         dropout=0.2, attn_dropout=0.1),
        "optim": dict(lr=1e-3, weight_decay=0.01, warmup_epochs=5),
        "max_epochs": 50, "patience": 10,
    },
    "regularized": {
        "label_smoothing": 0.0,
        "patchtst": dict(d_model=64, n_heads=4, n_layers=2, d_ff=128,
                         dropout=0.4, attn_dropout=0.2),
        "optim": dict(lr=1e-3, weight_decay=0.05, warmup_epochs=5),
        "max_epochs": 50, "patience": 10,
    },
    "label_smooth": {
        "label_smoothing": 0.1,
        "patchtst": dict(d_model=64, n_heads=4, n_layers=2, d_ff=128,
                         dropout=0.2, attn_dropout=0.1),
        "optim": dict(lr=1e-3, weight_decay=0.01, warmup_epochs=5),
        "max_epochs": 50, "patience": 10,
    },
    "slow_lr": {
        "label_smoothing": 0.0,
        "patchtst": dict(d_model=64, n_heads=4, n_layers=2, d_ff=128,
                         dropout=0.2, attn_dropout=0.1),
        "optim": dict(lr=3e-4, weight_decay=0.01, warmup_epochs=10),
        "max_epochs": 80, "patience": 15,
    },
    "all_fixes": {
        "label_smoothing": 0.1,
        "patchtst": dict(d_model=32, n_heads=4, n_layers=1, d_ff=64,
                         dropout=0.4, attn_dropout=0.2),
        "optim": dict(lr=3e-4, weight_decay=0.05, warmup_epochs=10),
        "max_epochs": 80, "patience": 15,
    },
}


def build_config(name, hp):
    """Construit FullExpConfig à partir des hyperparams nommés."""
    return FullExpConfig(
        exp_name=f"diag_{name}",
        optim=OptimConfig(**hp["optim"]),
        training=TrainingConfig(
            max_epochs=hp["max_epochs"],
            batch_size=8,
            early_stopping_patience=hp["patience"],
            use_class_weight=True,
            use_trial_subsampling=True,
            subsample_strategy="stratified",
            seed=42,
        ),
        seq_len=20, n_channels=14,
        patch_len=4, stride=2,
        **hp["patchtst"],
    )


def run_one_fold(ds, test_child_id, hp_name, hp, device):
    """Un fold mini-LOSO complet (1 inner split, pas 15)."""
    train_pool = [c for c in ds.child_ids if c != test_child_id]
    splits = list(inner_cv_splits(
        train_pool, ds.labels_per_child, n_splits=5, n_repeats=1, seed=42,
    ))
    s0 = splits[0]

    cfg = build_config(hp_name, hp)

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
    n_params = model.n_parameters()

    out = train_supervised_with_smoothing(
        model, train_loader, val_loader, device, cfg,
        label_smoothing=hp["label_smoothing"],
    )

    model.load_state_dict(out["best_state"])
    test_proba = float(predict_proba(model, test_loader, device)[0])
    test_pred = int(test_proba >= 0.5)
    test_true = ds.labels_per_child[test_child_id]

    return {
        "test_child": test_child_id,
        "true": test_true,
        "pred": test_pred,
        "proba_adhd": test_proba,
        "correct": test_pred == test_true,
        "best_epoch": out["best_epoch"],
        "best_val_loss": out["best_val_loss"],
        "best_val_acc": out["best_val_acc"],
        "n_epochs_run": out["n_epochs_run"],
        "n_params": n_params,
    }


def evaluate_config(ds, hp_name, hp, device, test_subset):
    """Évalue une config sur le mini-LOSO."""
    results = []
    t0 = time.time()
    for cid in test_subset:
        r = run_one_fold(ds, cid, hp_name, hp, device)
        results.append(r)
    dt = time.time() - t0

    n_correct = sum(r["correct"] for r in results)
    probas = [r["proba_adhd"] for r in results]
    extreme = sum(1 for p in probas if p < 0.05 or p > 0.95)
    proba_std = float(np.std(probas))
    proba_min = float(min(probas))
    proba_max = float(max(probas))

    n_params = results[0]["n_params"]
    return {
        "config": hp_name,
        "n_params": n_params,
        "accuracy": n_correct / len(test_subset),
        "n_correct": n_correct,
        "n_total": len(test_subset),
        "extreme_count": extreme,
        "proba_min": proba_min,
        "proba_max": proba_max,
        "proba_std": proba_std,
        "probas": probas,
        "preds": [r["pred"] for r in results],
        "trues": [r["true"] for r in results],
        "duration_s": dt,
        "avg_best_epoch": float(np.mean([r["best_epoch"] for r in results])),
    }


def main():
    set_global_seed(42)
    device = get_device()
    print(f"=== Phase 3.2 grid diagnostic ===")
    print(f"Device: {device}")

    ds = load_processed_dataset(Path("data/processed"))
    print(f"Loaded {len(ds.child_ids)} children")

    test_subset = ds.child_ids[:5]
    true_labels = [ds.labels_per_child[c] for c in test_subset]
    print(f"Test folds:    {test_subset}")
    print(f"True labels:   {true_labels} (0=CTRL, 1=ADHD)")

    all_results = []
    for hp_name, hp in CONFIGS.items():
        print(f"\n--- Config: {hp_name} ---")
        result = evaluate_config(ds, hp_name, hp, device, test_subset)
        all_results.append(result)
        # Affichage compact
        proba_str = ", ".join(f"{p:.3f}" for p in result["probas"])
        pred_str = "".join(str(p) for p in result["preds"])
        true_str = "".join(str(t) for t in result["trues"])
        print(f"  n_params={result['n_params']}, took={result['duration_s']:.1f}s, "
              f"avg_best_ep={result['avg_best_epoch']:.1f}")
        print(f"  probas:    [{proba_str}]")
        print(f"  preds:     {pred_str}")
        print(f"  trues:     {true_str}")
        print(f"  accuracy:  {result['n_correct']}/{result['n_total']} "
              f"({result['accuracy']:.1%})")
        print(f"  proba_std: {result['proba_std']:.3f} (>0.1 = good variability)")
        print(f"  extreme:   {result['extreme_count']}/5 "
              f"({'⚠ MODE COLLAPSE' if result['extreme_count'] >= 4 else 'OK'})")

    # Synthèse comparative
    print(f"\n{'=' * 72}")
    print(f"SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'config':<15s} {'params':>7s} {'acc':>6s} {'std':>6s} "
          f"{'extreme':>8s} {'time':>6s}")
    for r in all_results:
        flag = "  ⚠" if r["extreme_count"] >= 4 else "   "
        print(f"{r['config']:<15s} {r['n_params']:>7d} "
              f"{r['accuracy']:>6.1%} {r['proba_std']:>6.3f} "
              f"{r['extreme_count']:>3d}/5 {r['duration_s']:>5.1f}s{flag}")

    # Recommandation
    print(f"\n--- Recommendation ---")
    # Trouver la meilleure config : accuracy + diversité de probas
    sorted_results = sorted(
        all_results,
        key=lambda r: (r["accuracy"], r["proba_std"]),
        reverse=True,
    )
    best = sorted_results[0]
    if best["accuracy"] >= 0.6 and best["extreme_count"] < 4:
        print(f"  ✓ Best config: '{best['config']}' "
              f"(accuracy={best['accuracy']:.1%}, proba_std={best['proba_std']:.3f})")
        print(f"  Safe to launch full LOSO with this config.")
    elif best["accuracy"] >= 0.6:
        print(f"  ~ Best by accuracy: '{best['config']}' but probas still extreme.")
        print(f"    Mode collapse remains a concern. Try larger inner CV (k=3).")
    else:
        print(f"  ⚠ No config achieves both >60% accuracy AND non-extreme probas.")
        print(f"    Need deeper investigation: check if labels are correlated with")
        print(f"    individual paramètres, try k=3 inner CV, or augment data.")


if __name__ == "__main__":
    main()
