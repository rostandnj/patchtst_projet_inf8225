"""
test_phase3_2_diag_deep.py

Diagnostic PROFOND du mode collapse observé dans test_phase3_2_diag_grid.py.

Vérifie trois hypothèses dans l'ordre :

    1. LABELS & CLASS WEIGHTS
       - Distribution réelle des labels dans le train
       - Class weights calculés
       - Distribution dans les batches (sanity check du DataLoader)

    2. REVIN SUR DONNÉES RÉELLES
       - Statistiques (mean, std) des paramètres d'entrée pour quelques enfants
       - Sortie de RevIN : valeurs extrêmes, NaN, inf
       - Variance unitaire bien obtenue

    3. LOGITS DU MODÈLE
       - Forward d'un PatchTST entraîné sur train et test
       - Distribution des logits par classe (avant softmax)
       - Identifie si le modèle prédit constamment un même logit
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
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
from src.training.inner_cv import inner_cv_splits
from src.training.supervised import (
    train_supervised, _compute_class_weights, _epoch_loop,
)
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def main():
    set_global_seed(42)
    device = get_device()
    print(f"Device: {device}")

    ds = load_processed_dataset(Path("data/processed"))
    print(f"Loaded {len(ds.child_ids)} children, {ds.total_strokes} strokes")

    # Construit un fold de référence : test=S01, inner_split=0
    test_child = "S01"
    train_pool = [c for c in ds.child_ids if c != test_child]
    splits = list(inner_cv_splits(
        train_pool, ds.labels_per_child, n_splits=5, n_repeats=1, seed=42,
    ))
    s0 = splits[0]

    # =====================================================================
    # 1. LABELS & CLASS WEIGHTS
    # =====================================================================
    section("1. LABELS & CLASS WEIGHTS")

    train_labels = [ds.labels_per_child[c] for c in s0.train_inner_ids]
    val_labels = [ds.labels_per_child[c] for c in s0.val_inner_ids]
    print(f"Train (n={len(train_labels)}): "
          f"CTRL={train_labels.count(0)}, ADHD={train_labels.count(1)}")
    print(f"Val   (n={len(val_labels)}): "
          f"CTRL={val_labels.count(0)}, ADHD={val_labels.count(1)}")
    print(f"Test (n=1): {ds.labels_per_child[test_child]} "
          f"({'ADHD' if ds.labels_per_child[test_child]==1 else 'CTRL'})")

    train_ds = SequenceChildDataset(
        ds, s0.train_inner_ids, seq_len=20,
        is_train=True, subsample_strategy="stratified", seed=42,
    )
    val_ds = SequenceChildDataset(
        ds, s0.val_inner_ids, seq_len=20,
        is_train=False, subsample_strategy="stratified", seed=43,
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Sanity check : itère sur le train_loader pour 1 epoch et compte les labels
    label_counts = {0: 0, 1: 0}
    for x_batch, y_batch in train_loader:
        for y in y_batch.tolist():
            label_counts[y] += 1
    print(f"Labels seen in 1 epoch through DataLoader: "
          f"CTRL={label_counts[0]}, ADHD={label_counts[1]}")

    # Class weights réellement calculés
    cw = _compute_class_weights(train_loader, device)
    print(f"Computed class weights: CTRL={cw[0].item():.4f}, ADHD={cw[1].item():.4f}")
    print(f"  (1.0 each = balanced ; <1.0 means ignored ; >1.0 means amplified)")

    # =====================================================================
    # 2. REVIN SUR DONNÉES RÉELLES
    # =====================================================================
    section("2. REVIN INPUT/OUTPUT SUR DONNÉES RÉELLES")

    # Prend un batch type
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    print(f"Input x: shape={tuple(x_batch.shape)}")
    print(f"  raw stats:  min={x_batch.min().item():.3f}, "
          f"max={x_batch.max().item():.3f}, "
          f"mean={x_batch.mean().item():.3f}, std={x_batch.std().item():.3f}")
    print(f"  any NaN/inf: {torch.isnan(x_batch).any().item()} / "
          f"{torch.isinf(x_batch).any().item()}")

    # Stats par canal (les 14 paramètres)
    print(f"\n  Per-channel stats (over batch and time):")
    print(f"  {'channel':>10s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    for c in range(min(14, x_batch.shape[1])):
        ch = x_batch[:, c, :]
        print(f"  {ds.param_names[c]:>10s} "
              f"{ch.mean().item():>10.3f} {ch.std().item():>10.3f} "
              f"{ch.min().item():>10.3f} {ch.max().item():>10.3f}")

    # Trouver canaux avec variance ~0 (problématique pour RevIN)
    per_channel_per_sample_std = x_batch.std(dim=-1)  # (B, M)
    near_zero = (per_channel_per_sample_std < 1e-6).sum().item()
    if near_zero > 0:
        print(f"\n  ⚠ {near_zero} (sample, channel) pairs have std ≈ 0 "
              f"(out of {x_batch.shape[0] * x_batch.shape[1]})")
        print(f"     This causes RevIN to amplify noise via division by eps.")

    # Fait passer dans RevIN seul
    cfg_test = PatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model_test = PatchTSTClassifier(cfg_test, n_classes=2).to(device)
    with torch.no_grad():
        x_norm = model_test.backbone.revin(x_batch, mode="norm")
    print(f"\n  After RevIN:")
    print(f"    min={x_norm.min().item():.3f}, max={x_norm.max().item():.3f}, "
          f"mean={x_norm.mean().item():.4f}, std={x_norm.std().item():.3f}")
    print(f"    any NaN/inf: {torch.isnan(x_norm).any().item()} / "
          f"{torch.isinf(x_norm).any().item()}")
    extreme_count = ((x_norm.abs() > 10).sum().item())
    if extreme_count > 0:
        print(f"    ⚠ {extreme_count} values have |x| > 10 after RevIN "
              f"(should be near zero)")

    # =====================================================================
    # 3. LOGITS APRÈS UN MINI ENTRAÎNEMENT
    # =====================================================================
    section("3. LOGITS DU MODÈLE APRÈS ENTRAÎNEMENT")

    cfg = FullExpConfig(
        exp_name="diag_deep",
        optim=OptimConfig(lr=1e-3, weight_decay=0.01, warmup_epochs=5),
        training=TrainingConfig(
            max_epochs=20, batch_size=8,
            early_stopping_patience=15,
            use_class_weight=True,
            use_trial_subsampling=True,
            subsample_strategy="stratified",
            seed=42,
        ),
        seq_len=20, n_channels=14,
        patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg_test, n_classes=2)
    print(f"Training for max {cfg.training.max_epochs} epochs (silent)...")
    history, best_state = train_supervised(
        model, train_loader, val_loader, device, cfg,
        verbose=False, log_every=100,
    )
    model.load_state_dict(best_state)
    model.eval()
    print(f"Done. Best epoch={history.best_epoch}, "
          f"val_loss={history.best_val_loss:.4f}, "
          f"val_acc={history.best_val_acc:.3f}")

    # Logits sur tous les samples train, val, test
    print(f"\n  Logits distribution (per dataset):")
    print(f"  {'dataset':>8s} {'true':>5s} {'logit_CTRL':>12s} {'logit_ADHD':>12s} "
          f"{'softmax_ADHD':>14s} {'pred':>5s}")

    def predict_logits(dataset, name):
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        rows = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)                    # (B, 2)
                proba = torch.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1)
                for i in range(x.size(0)):
                    rows.append((
                        name,
                        int(y[i].item()),
                        float(logits[i, 0].item()),
                        float(logits[i, 1].item()),
                        float(proba[i, 1].item()),
                        int(pred[i].item()),
                    ))
        return rows

    # Train (2 premiers samples), val (tous), test
    train_eval_ds = SequenceChildDataset(
        ds, s0.train_inner_ids, seq_len=20,
        is_train=False, subsample_strategy="stratified",  # eval mode
    )
    test_ds = SequenceChildDataset(
        ds, [test_child], seq_len=20,
        is_train=False, subsample_strategy="stratified",
    )

    rows = []
    rows.extend(predict_logits(train_eval_ds, "train"))
    rows.extend(predict_logits(val_ds, "val"))
    rows.extend(predict_logits(test_ds, "test"))

    for name, true, logit_c, logit_a, sm_a, pred in rows:
        flag = "★" if (name == "test") else ("•" if name == "val" else " ")
        print(f"  {flag} {name:>6s} {true:>5d} {logit_c:>12.3f} {logit_a:>12.3f} "
              f"{sm_a:>14.4f} {pred:>5d}")

    # Synthèse de la distribution des logits
    print(f"\n  Synthèse :")
    logits_ctrl = [r[2] for r in rows]
    logits_adhd = [r[3] for r in rows]
    print(f"    logit_CTRL : min={min(logits_ctrl):.3f}, max={max(logits_ctrl):.3f}, "
          f"mean={np.mean(logits_ctrl):.3f}, std={np.std(logits_ctrl):.3f}")
    print(f"    logit_ADHD : min={min(logits_adhd):.3f}, max={max(logits_adhd):.3f}, "
          f"mean={np.mean(logits_adhd):.3f}, std={np.std(logits_adhd):.3f}")
    if np.std(logits_adhd) < 0.1:
        print(f"    ⚠ logit_ADHD a une variance quasi-nulle -> le modèle prédit un "
              f"score quasi-constant pour ADHD")
    if np.mean(logits_adhd) < np.mean(logits_ctrl) - 5:
        print(f"    ⚠ logit_ADHD est massivement inférieur à logit_CTRL -> "
              f"biais systématique vers CTRL")

    # =====================================================================
    # 4. CONCLUSION
    # =====================================================================
    section("CONCLUSION")
    if cw[1].item() < 0.5:
        print(f"  ⚠ Class weight ADHD ({cw[1].item():.3f}) trop faible -> "
              f"le modèle ignore ADHD")
    elif np.std(logits_adhd) < 0.1:
        print(f"  ⚠ Mode collapse confirmé : le modèle prédit un logit constant.")
        print(f"     Possible cause : insuffisance de signal au niveau séquence,")
        print(f"     ou batch size / class imbalance dans le val.")
    elif np.mean(logits_adhd) < np.mean(logits_ctrl) - 5:
        print(f"  ⚠ Biais vers CTRL malgré class_weight balanced.")
    else:
        print(f"  Pas de bug évident détecté. Le modèle apprend mais converge")
        print(f"  vers une frontière de décision peu informative.")


if __name__ == "__main__":
    main()
