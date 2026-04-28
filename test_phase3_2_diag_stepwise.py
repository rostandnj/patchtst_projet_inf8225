"""
test_phase3_2_diag_stepwise.py

DIAGNOSTIC PAS-À-PAS : observer la dynamique d'entraînement de PatchTST
gradient par gradient pendant les 5 premières epochs, pour identifier
le moment EXACT où le mode collapse s'enclenche.

À CHAQUE BATCH on logge :
    - Logits avant/après le pas de gradient
    - Loss totale et loss séparée par classe vraie (CTRL vs ADHD)
    - Norme du gradient global et sur la dernière couche
    - Distribution des prédictions (combien CTRL/ADHD)
    - Statistiques softmax (combien de probas saturées)

À LA FIN de chaque epoch on logge :
    - Logits sur train (eval mode) et val
    - Snapshot du modèle pour analyse

Sortie : tableau dense avec toutes les métriques pour analyse précise.

Durée estimée : ~30 secondes (5 epochs × ~3 batches).
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_a_datasets import SequenceChildDataset
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.config import OptimConfig
from src.training.device import get_device
from src.training.inner_cv import inner_cv_splits
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.training.supervised import _compute_class_weights
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def section(title):
    print(f"\n{'=' * 76}")
    print(f"  {title}")
    print(f"{'=' * 76}")


@torch.no_grad()
def snapshot_logits(model, loader, device, name):
    """Évalue le modèle (eval mode) et retourne les logits + labels."""
    model.eval()
    logits_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        logits_list.append(logits.cpu().numpy())
        labels_list.append(y.numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)


def summarize_logits(logits, labels, name):
    """Imprime un résumé des logits (par classe vraie)."""
    if len(logits) == 0:
        return
    probs_adhd = F.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = logits.argmax(axis=-1)
    n_correct = int((preds == labels).sum())
    print(f"  [{name}] n={len(labels)} acc={n_correct}/{len(labels)} = {n_correct/len(labels):.2f}")

    for cls in (0, 1):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        cls_name = "CTRL" if cls == 0 else "ADHD"
        l_ctrl = logits[mask, 0].mean()
        l_adhd = logits[mask, 1].mean()
        p_adhd = probs_adhd[mask].mean()
        sat_high = float((probs_adhd[mask] > 0.99).mean())
        sat_low = float((probs_adhd[mask] < 0.01).mean())
        print(f"    True={cls_name} (n={int(mask.sum())}): "
              f"logit_C={l_ctrl:+.3f}  logit_A={l_adhd:+.3f}  "
              f"P(ADHD)_avg={p_adhd:.4f}  "
              f"sat>0.99: {sat_high:.0%}  sat<0.01: {sat_low:.0%}")


def main():
    set_global_seed(42)
    device = get_device()
    print(f"Device: {device}")

    ds = load_processed_dataset(Path("data/processed"))

    # Fold de référence
    test_child = "S01"
    train_pool = [c for c in ds.child_ids if c != test_child]
    splits = list(inner_cv_splits(
        train_pool, ds.labels_per_child, n_splits=5, n_repeats=1, seed=42,
    ))
    s0 = splits[0]

    train_ds = SequenceChildDataset(
        ds, s0.train_inner_ids, seq_len=20,
        is_train=True, subsample_strategy="stratified", seed=42,
    )
    val_ds = SequenceChildDataset(
        ds, s0.val_inner_ids, seq_len=20,
        is_train=False, subsample_strategy="stratified",
    )
    train_eval_ds = SequenceChildDataset(   # train en mode eval pour snapshots
        ds, s0.train_inner_ids, seq_len=20,
        is_train=False, subsample_strategy="stratified",
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=8, shuffle=False)

    # Modèle baseline
    cfg = PatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2).to(device)

    optim_cfg = OptimConfig(lr=1e-3, weight_decay=0.01, warmup_epochs=5, grad_clip=1.0)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, total_epochs=5)

    class_weights = _compute_class_weights(train_loader, device)
    print(f"\nClass weights: CTRL={class_weights[0]:.3f}, ADHD={class_weights[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Snapshot initial avant tout entraînement
    section("INITIAL STATE (epoch 0, before any training)")
    train_logits, train_labels = snapshot_logits(model, train_eval_loader, device, "TRAIN")
    summarize_logits(train_logits, train_labels, "TRAIN")
    val_logits, val_labels = snapshot_logits(model, val_loader, device, "VAL")
    summarize_logits(val_logits, val_labels, "VAL")

    # Boucle d'entraînement par PAS, pas par EPOCH
    section("STEP-BY-STEP TRAINING (5 epochs × ~3 batches)")
    print(f"Format per step:")
    print(f"  step E.B  lr=...  loss=...  loss_CTRL=...  loss_ADHD=...")
    print(f"           grad_norm=...  fc_grad_norm=...")
    print(f"           preds: CTRL=X ADHD=Y  satur_high=N satur_low=M")
    print()

    # Récupère la dernière couche linéaire pour suivre ses gradients
    fc_layer = model.head.fc

    for epoch in range(5):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            # Forward
            logits = model(x)                                      # (B, 2)
            losses_per_sample = F.cross_entropy(
                logits, y, weight=class_weights, reduction='none'
            )                                                       # (B,)
            loss = losses_per_sample.mean()

            # Loss séparée par classe vraie
            mask_ctrl = (y == 0)
            mask_adhd = (y == 1)
            loss_ctrl = losses_per_sample[mask_ctrl].mean().item() if mask_ctrl.any() else 0.0
            loss_adhd = losses_per_sample[mask_adhd].mean().item() if mask_adhd.any() else 0.0

            # Stats softmax saturation
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                sat_high = int((probs[:, 1] > 0.99).sum().item())
                sat_low = int((probs[:, 1] < 0.01).sum().item())
                preds = logits.argmax(dim=-1)
                n_pred_ctrl = int((preds == 0).sum().item())
                n_pred_adhd = int((preds == 1).sum().item())

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Norme des gradients
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            fc_grad_norm = fc_layer.weight.grad.norm().item() if fc_layer.weight.grad is not None else 0.0

            clip_gradients(model, optim_cfg.grad_clip)
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            true_ctrl = int(mask_ctrl.sum().item())
            true_adhd = int(mask_adhd.sum().item())

            print(f"step {epoch}.{batch_idx}  lr={current_lr:.5f}  "
                  f"loss={loss.item():.4f}  loss_CTRL={loss_ctrl:.4f}  "
                  f"loss_ADHD={loss_adhd:.4f}")
            print(f"               grad_norm={total_grad_norm:.4f}  "
                  f"fc_grad_norm={fc_grad_norm:.4f}")
            print(f"               batch: y_true CTRL={true_ctrl}/ADHD={true_adhd}  "
                  f"preds CTRL={n_pred_ctrl}/ADHD={n_pred_adhd}  "
                  f"satur_high={sat_high} satur_low={sat_low}")

        scheduler.step()

        # Snapshot fin d'epoch
        print(f"\n  -- End of epoch {epoch} --")
        train_logits, train_labels = snapshot_logits(model, train_eval_loader, device, "TRAIN")
        summarize_logits(train_logits, train_labels, "TRAIN")
        val_logits, val_labels = snapshot_logits(model, val_loader, device, "VAL")
        summarize_logits(val_logits, val_labels, "VAL")
        print()

    section("INTERPRETATION GUIDE")
    print("CHERCHE DANS LA TRACE :")
    print()
    print("1. À quel step satur_high+satur_low = batch_size (toutes les probas")
    print("   saturent à 0 ou 1) ? -> moment du mode collapse")
    print()
    print("2. Que vaut fc_grad_norm à ce moment ? Si ~0 -> gradients morts,")
    print("   le modèle ne peut plus s'extraire de cette région.")
    print()
    print("3. Comment évolue loss_ADHD vs loss_CTRL ? Si loss_ADHD reste élevée")
    print("   alors que loss_CTRL chute à 0 -> biais asymétrique : le modèle")
    print("   prédit bien CTRL et ignore les ADHD.")
    print()
    print("4. Sur le snapshot fin d'epoch en TRAIN : est-ce que la moyenne des")
    print("   logit_ADHD pour les vrais ADHD remonte epoch après epoch, ou")
    print("   reste-t-elle bloquée ? -> diagnostic de la rigidité du collapse.")


if __name__ == "__main__":
    main()
