"""
test_phase3_2_init_bias.py

INVESTIGATION : pourquoi le PatchTSTClassifier prédit-il systématiquement
P(ADHD) ≈ 0.45 à l'initialisation, biaisant tout l'entraînement vers CTRL ?

Étapes :
    1. Créer un PatchTSTClassifier neuf, inspecter les poids du fc final
    2. Mesurer la distribution des probas à l'init sur les 24 enfants
    3. Reset le bias à zéro et re-mesurer
    4. Lancer le simple LOSO avec un FIX de bias init

Hypothèse : avec init.zeros_(fc.bias), le modèle démarre avec P(ADHD)≈0.5
exactement, ce qui devrait permettre l'apprentissage du signal réel.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
from collections import Counter
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
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.training.supervised import predict_proba
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def inspect_init(model):
    """Inspecte les poids/biais de la couche fc finale."""
    fc = model.head.fc
    print(f"  fc.weight: shape={tuple(fc.weight.shape)}")
    print(f"    [class 0] mean={fc.weight[0].mean().item():+.4f}, "
          f"std={fc.weight[0].std().item():.4f}, "
          f"L2={fc.weight[0].norm().item():.4f}")
    print(f"    [class 1] mean={fc.weight[1].mean().item():+.4f}, "
          f"std={fc.weight[1].std().item():.4f}, "
          f"L2={fc.weight[1].norm().item():.4f}")
    print(f"  fc.bias: {fc.bias.detach().cpu().numpy()}")


def compute_init_probas(ds, device, child_ids):
    """Calcule P(ADHD) à l'initialisation pour chaque enfant."""
    cfg = PatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2).to(device)
    model.eval()
    probas = []
    with torch.no_grad():
        for cid in child_ids:
            test_ds = SequenceChildDataset(
                ds, [cid], seq_len=20, is_train=False,
                subsample_strategy="stratified",
            )
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
            for x, _ in test_loader:
                x = x.to(device)
                logits = model(x)
                p = F.softmax(logits, dim=-1)[0, 1].item()
                probas.append(p)
    return model, np.array(probas)


def balance_train_ids(train_ids, labels_per_child, seed=42):
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


def train_simple(model, train_loader, device, n_epochs=30, lr=1e-3,
                  weight_decay=0.01, warmup_epochs=5, grad_clip=1.0):
    model = model.to(device)
    optim_cfg = OptimConfig(lr=lr, weight_decay=weight_decay,
                             warmup_epochs=warmup_epochs, grad_clip=grad_clip)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, n_epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(model, grad_clip)
            optimizer.step()
        scheduler.step()


def run_one_fold(ds, test_child_id, device, fix_bias=False, n_epochs=30, seed=42):
    train_pool = [c for c in ds.child_ids if c != test_child_id]
    balanced_ids, _, _ = balance_train_ids(train_pool, ds.labels_per_child, seed=seed)

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

    if fix_bias:
        # Reset bias et weight de la classification head pour démarrer neutre
        with torch.no_grad():
            nn.init.zeros_(model.head.fc.bias)
            # Plus petite norme pour le weight final aussi
            model.head.fc.weight.mul_(0.01)

    train_simple(model, train_loader, device, n_epochs=n_epochs)
    test_proba = float(predict_proba(model, test_loader, device)[0])
    test_pred = int(test_proba >= 0.5)
    test_true = ds.labels_per_child[test_child_id]
    return {
        "test_child": test_child_id,
        "true": test_true, "pred": test_pred,
        "proba_adhd": test_proba,
        "correct": test_pred == test_true,
    }


def main():
    set_global_seed(42)
    device = get_device()
    print(f"=== PatchTST init bias diagnostic ===")
    print(f"Device: {device}")
    ds = load_processed_dataset(Path("data/processed"))

    # ----------------------------------------------------------------
    # 1. Inspect fresh model
    # ----------------------------------------------------------------
    print(f"\n--- 1. Inspect freshly initialized PatchTSTClassifier ---")
    cfg = PatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model_fresh = PatchTSTClassifier(cfg, n_classes=2).to(device)
    inspect_init(model_fresh)

    # ----------------------------------------------------------------
    # 2. P(ADHD) à l'init pour tous les enfants
    # ----------------------------------------------------------------
    print(f"\n--- 2. P(ADHD) at init for all 24 children ---")
    set_global_seed(42)
    model_fresh, probas_init = compute_init_probas(ds, device, ds.child_ids)
    print(f"  P(ADHD) init: min={probas_init.min():.4f}, max={probas_init.max():.4f}, "
          f"mean={probas_init.mean():.4f}, std={probas_init.std():.4f}")
    above_half = int((probas_init >= 0.5).sum())
    print(f"  Probas ≥ 0.5: {above_half}/24 (would predict ADHD)")
    print(f"  Probas < 0.5: {24-above_half}/24 (would predict CTRL)")

    # ----------------------------------------------------------------
    # 3. Reset bias et re-mesurer
    # ----------------------------------------------------------------
    print(f"\n--- 3. After fix_bias (zero bias + 0.01 weight scale) ---")
    set_global_seed(42)
    cfg2 = PatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model_fixed = PatchTSTClassifier(cfg2, n_classes=2).to(device)
    with torch.no_grad():
        nn.init.zeros_(model_fixed.head.fc.bias)
        model_fixed.head.fc.weight.mul_(0.01)
    print(f"  After fix:")
    inspect_init(model_fixed)

    # Recalcule probas avec ce modèle fixé
    model_fixed.eval()
    probas_fixed = []
    with torch.no_grad():
        for cid in ds.child_ids:
            test_ds = SequenceChildDataset(
                ds, [cid], seq_len=20, is_train=False,
                subsample_strategy="stratified",
            )
            for x, _ in DataLoader(test_ds, batch_size=1, shuffle=False):
                x = x.to(device)
                logits = model_fixed(x)
                probas_fixed.append(F.softmax(logits, dim=-1)[0, 1].item())
    probas_fixed = np.array(probas_fixed)
    print(f"  P(ADHD) after fix: min={probas_fixed.min():.4f}, "
          f"max={probas_fixed.max():.4f}, mean={probas_fixed.mean():.4f}")

    # ----------------------------------------------------------------
    # 4. Mini-LOSO avec fix_bias=True
    # ----------------------------------------------------------------
    print(f"\n--- 4. Simple LOSO on S01..S05 with fix_bias=True ---")
    test_subset = ds.child_ids[:5]
    results_fix = []
    for cid in test_subset:
        t0 = time.time()
        r = run_one_fold(ds, cid, device, fix_bias=True, n_epochs=30, seed=42)
        dt = time.time() - t0
        ok = "✓" if r["correct"] else "✗"
        tag = "ADHD" if r["true"] == 1 else "CTRL"
        print(f"  {cid} [{tag}]: P(ADHD)={r['proba_adhd']:.4f} "
              f"pred={'ADHD' if r['pred']==1 else 'CTRL'} {ok}  ({dt:.1f}s)")
        results_fix.append(r)

    n_ok = sum(r["correct"] for r in results_fix)
    probas = [r["proba_adhd"] for r in results_fix]
    print(f"\n  Accuracy: {n_ok}/5 = {n_ok/5:.0%}")
    print(f"  Probas: min={min(probas):.3f}, max={max(probas):.3f}, "
          f"mean={np.mean(probas):.3f}, std={np.std(probas):.3f}")
    extreme = sum(1 for p in probas if p < 0.05 or p > 0.95)
    print(f"  Extreme: {extreme}/5")
    if np.std(probas) > 0.05:
        print(f"  ✓ Probas show some variability — discrimination is happening")
    else:
        print(f"  ⚠ Still no variability in predictions")


if __name__ == "__main__":
    main()
