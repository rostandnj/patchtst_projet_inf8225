"""
test_phase3_2_simple_loso.py

LOSO simple :
    - Pas de validation interne (pas de split val)
    - Train sur 22 enfants strictement équilibrés (11 CTRL + 11 ADHD)
      en retirant 1 enfant de la classe majoritaire
    - 30 epochs fixes, pas d'early stopping
    - Test sur l'enfant LOSO exclu

OBJECTIF : tester l'hypothèse "le déséquilibre 8 vs 10 dans le train interne
était la cause du biais vers la classe majoritaire".

Si ça marche -> on a notre setup pour le LOSO complet.
Si ça ne marche pas -> on saura que le problème est plus profond.

Pour démarrer, on fait sur les 5 enfants S01..S05 (rapide, ~30s).
Si encourageant, on lance sur les 24 enfants (~2 min).
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_a_datasets import SequenceChildDataset
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.config import OptimConfig
from src.training.device import get_device
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.training.supervised import predict_proba
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def balance_train_ids(train_ids, labels_per_child, seed=42):
    """Retire 1 enfant de la classe majoritaire pour équilibrer 1:1.

    Returns:
        (balanced_ids, removed_id, removed_label)
    """
    labels = [labels_per_child[c] for c in train_ids]
    counts = Counter(labels)
    if counts[0] == counts[1]:
        return list(train_ids), None, None
    majority = 0 if counts[0] > counts[1] else 1
    rng = np.random.default_rng(seed)
    candidates = [c for c in train_ids if labels_per_child[c] == majority]
    removed = str(rng.choice(candidates))
    balanced = [c for c in train_ids if c != removed]
    return balanced, removed, majority


def train_simple(model, train_loader, device, n_epochs=30, lr=1e-3, weight_decay=0.01,
                  warmup_epochs=5, grad_clip=1.0, verbose=False):
    """Entraînement simple sans validation, n_epochs fixes."""
    model = model.to(device)
    optim_cfg = OptimConfig(lr=lr, weight_decay=weight_decay,
                             warmup_epochs=warmup_epochs, grad_clip=grad_clip)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, n_epochs)

    # Pas de class weight : on a déjà équilibré les données
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": []}
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_n, total_correct = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(model, grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)
            total_correct += int((logits.argmax(-1) == y).sum().item())
        avg_loss = total_loss / max(1, total_n)
        acc = total_correct / max(1, total_n)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(acc)
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"    ep{epoch:3d}  lr={optimizer.param_groups[0]['lr']:.5f}  "
                  f"loss={avg_loss:.4f}  acc={acc:.3f}")
        scheduler.step()
    return history


def run_one_fold(ds, test_child_id, device, n_epochs=30, seed=42, verbose=False):
    """Un fold LOSO simple sans validation interne."""
    train_pool = [c for c in ds.child_ids if c != test_child_id]
    balanced_ids, removed_id, removed_label = balance_train_ids(
        train_pool, ds.labels_per_child, seed=seed,
    )
    train_labels = [ds.labels_per_child[c] for c in balanced_ids]
    n_ctrl = train_labels.count(0)
    n_adhd = train_labels.count(1)

    set_global_seed(seed)
    train_ds = SequenceChildDataset(
        ds, balanced_ids, seq_len=20,
        is_train=True, subsample_strategy="stratified", seed=seed,
    )
    test_ds = SequenceChildDataset(
        ds, [test_child_id], seq_len=20,
        is_train=False, subsample_strategy="stratified",
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    cfg = PatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2)

    history = train_simple(model, train_loader, device, n_epochs=n_epochs,
                            verbose=verbose)

    test_proba = float(predict_proba(model, test_loader, device)[0])
    test_pred = int(test_proba >= 0.5)
    test_true = ds.labels_per_child[test_child_id]

    return {
        "test_child": test_child_id,
        "true": test_true,
        "pred": test_pred,
        "proba_adhd": test_proba,
        "correct": test_pred == test_true,
        "removed_id": removed_id,
        "removed_label": removed_label,
        "n_train_ctrl": n_ctrl,
        "n_train_adhd": n_adhd,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Run on all 24 children (default: only S01..S05)")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = get_device()
    print(f"=== Simple LOSO (balanced 1:1, no val, {args.n_epochs} epochs) ===")
    print(f"Device: {device}")

    ds = load_processed_dataset(Path("data/processed"))
    print(f"Loaded {len(ds.child_ids)} children")

    test_subset = ds.child_ids if args.all else ds.child_ids[:5]
    n = len(test_subset)
    print(f"Testing folds: {n} children {'(ALL)' if args.all else '(S01..S05)'}")

    results = []
    t_start = time.time()
    for cid in test_subset:
        t0 = time.time()
        r = run_one_fold(ds, cid, device, n_epochs=args.n_epochs, seed=args.seed,
                          verbose=args.verbose)
        dt = time.time() - t0
        ok = "✓" if r["correct"] else "✗"
        tag = "ADHD" if r["true"] == 1 else "CTRL"
        rmv = (f" (removed {r['removed_id']}, label {r['removed_label']})"
               if r["removed_id"] else " (no removal)")
        print(f"  {cid} [{tag}]: train={r['n_train_ctrl']}+{r['n_train_adhd']}{rmv}  "
              f"P(ADHD)={r['proba_adhd']:.4f}  pred={'ADHD' if r['pred']==1 else 'CTRL'} {ok}  "
              f"({dt:.1f}s)")
        results.append(r)

    total_dt = time.time() - t_start

    # Synthèse
    print(f"\n=== Summary ===")
    n_correct = sum(r["correct"] for r in results)
    print(f"Accuracy: {n_correct}/{n} = {n_correct/n:.1%}")
    print(f"Total time: {total_dt:.1f}s ({total_dt/n:.1f}s per fold)")

    probas = [r["proba_adhd"] for r in results]
    extreme = sum(1 for p in probas if p < 0.05 or p > 0.95)
    print(f"Probas: min={min(probas):.3f}, max={max(probas):.3f}, "
          f"mean={np.mean(probas):.3f}, std={np.std(probas):.3f}")
    print(f"Extreme (>0.95 or <0.05): {extreme}/{n}")

    # Métrique de "détection" : proba prédite vs label vrai
    probas_for_ctrl = [r["proba_adhd"] for r in results if r["true"] == 0]
    probas_for_adhd = [r["proba_adhd"] for r in results if r["true"] == 1]
    if probas_for_ctrl and probas_for_adhd:
        print(f"\nDistribution P(ADHD) per true class:")
        print(f"  True=CTRL (n={len(probas_for_ctrl)}): "
              f"mean={np.mean(probas_for_ctrl):.3f}, max={max(probas_for_ctrl):.3f}")
        print(f"  True=ADHD (n={len(probas_for_adhd)}): "
              f"mean={np.mean(probas_for_adhd):.3f}, min={min(probas_for_adhd):.3f}")
        if np.mean(probas_for_adhd) > np.mean(probas_for_ctrl) + 0.1:
            print(f"  ✓ Le modèle discrimine : ADHD a plus de proba que CTRL")
        else:
            print(f"  ⚠ Pas de discrimination claire entre classes")

    # Synthèse des prédictions
    pred_counts = Counter(r["pred"] for r in results)
    print(f"\nPredictions: CTRL={pred_counts.get(0, 0)}, ADHD={pred_counts.get(1, 0)}")
    if pred_counts.get(0, 0) == n:
        print(f"  ⚠ Toujours en mode collapse (tout CTRL)")
    elif pred_counts.get(1, 0) == n:
        print(f"  ⚠ Toujours en mode collapse (tout ADHD)")
    else:
        print(f"  ✓ Le modèle prédit les deux classes")


if __name__ == "__main__":
    main()
