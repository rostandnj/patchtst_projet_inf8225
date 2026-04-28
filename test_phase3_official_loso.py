"""
test_phase3_official_loso.py

LOSO simple sur les 24 enfants en Exp A, avec PatchTST OFFICIEL
(yuqinie98/PatchTST) au lieu de notre implémentation maison.

OBJECTIF : valider que notre conclusion (PatchTST AUC ~0.45 sur Exp A)
n'est pas due à un bug d'implémentation, en utilisant le backbone certifié.

PRÉ-REQUIS :
    Avant de lancer ce script, télécharger les fichiers du repo officiel :

        cd /tmp
        git clone https://github.com/yuqinie98/PatchTST.git
        cp PatchTST/PatchTST_supervised/layers/RevIN.py             <project>/src/models/patchtst_official/
        cp PatchTST/PatchTST_supervised/layers/PatchTST_layers.py   <project>/src/models/patchtst_official/
        cp PatchTST/PatchTST_supervised/layers/PatchTST_backbone.py <project>/src/models/patchtst_official/

PROTOCOLE (identique à test_phase3_2_simple_loso_v2.py pour comparaison) :
    - Train équilibré 1:1 (11 CTRL + 11 ADHD = 22 enfants)
    - Pas de validation interne, 30 epochs fixes
    - Test sur l'enfant LOSO exclu
    - 5 seeds pour estimer la variance

Durée estimée : ~3 min sur M5 Pro (24 folds × 5 seeds × ~3s).
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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_a_datasets import SequenceChildDataset
from src.models.patchtst_official_wrapper import (
    OfficialPatchTSTConfig, OfficialPatchTSTClassifier,
)
from src.training.config import OptimConfig
from src.training.device import get_device
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.training.supervised import predict_proba
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed


def balance_train_ids(train_ids, labels_per_child, seed=42):
    labels = [labels_per_child[c] for c in train_ids]
    counts = Counter(labels)
    if counts[0] == counts[1]:
        return list(train_ids), None
    majority = 0 if counts[0] > counts[1] else 1
    rng = np.random.default_rng(seed)
    candidates = [c for c in train_ids if labels_per_child[c] == majority]
    removed = str(rng.choice(candidates))
    balanced = [c for c in train_ids if c != removed]
    return balanced, removed


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


def run_one_fold(ds, test_child_id, device, n_epochs=30, seed=42):
    train_pool = [c for c in ds.child_ids if c != test_child_id]
    balanced_ids, _ = balance_train_ids(train_pool, ds.labels_per_child, seed=seed)

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

    cfg = OfficialPatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
        revin=True, affine=True, subtract_last=False,
    )
    model = OfficialPatchTSTClassifier(cfg, n_classes=2, init_head_zero=True)

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


def run_full_loso(ds, device, n_epochs=30, seed=42, verbose=True):
    results = []
    t_start = time.time()
    for cid in ds.child_ids:
        t0 = time.time()
        r = run_one_fold(ds, cid, device, n_epochs=n_epochs, seed=seed)
        dt = time.time() - t0
        if verbose:
            ok = "✓" if r["correct"] else "✗"
            tag = "ADHD" if r["true"] == 1 else "CTRL"
            print(f"  {cid} [{tag}]: P(ADHD)={r['proba_adhd']:.4f} "
                  f"pred={'ADHD' if r['pred']==1 else 'CTRL'} {ok}  ({dt:.1f}s)")
        results.append(r)
    return results, time.time() - t_start


def summarize(results):
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    probas = [r["proba_adhd"] for r in results]
    y_true = np.array([r["true"] for r in results])
    y_pred = np.array([r["pred"] for r in results])
    y_proba = np.array(probas)
    accuracy = n_correct / n
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")
    return {
        "accuracy": accuracy, "n_correct": n_correct, "n_total": n,
        "f1": f1, "sensitivity": sens, "specificity": spec,
        "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "proba_min": float(min(probas)), "proba_max": float(max(probas)),
        "proba_mean": float(np.mean(probas)), "proba_std": float(np.std(probas)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp_a_official_loso"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"=== OFFICIAL PatchTST simple LOSO (Exp A) ===")
    print(f"Device: {device}")
    print(f"n_epochs={args.n_epochs}, multi_seed={args.multi_seed}")

    # Vérifie que les fichiers du repo officiel sont présents
    official_path = Path("src/models/patchtst_official")
    required_files = ["RevIN.py", "PatchTST_layers.py", "PatchTST_backbone.py"]
    missing = [f for f in required_files if not (official_path / f).exists()]
    if missing:
        print(f"\n❌ MISSING FILES in {official_path}/")
        for f in missing:
            print(f"    - {f}")
        print(f"\nPlease run:")
        print(f"  git clone https://github.com/yuqinie98/PatchTST.git /tmp/PatchTST_official")
        print(f"  cp /tmp/PatchTST_official/PatchTST_supervised/layers/*.py {official_path}/")
        return

    print(f"  ✓ All official files present in {official_path}/")

    ds = load_processed_dataset(Path("data/processed"))
    print(f"Loaded {len(ds.child_ids)} children")

    # Affiche taille du modèle
    cfg = OfficialPatchTSTConfig(
        n_channels=14, seq_len=20, patch_len=4, stride=2,
        n_layers=2, d_model=64, n_heads=4, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model_test = OfficialPatchTSTClassifier(cfg, n_classes=2)
    print(f"Official PatchTST model: {model_test.n_parameters():,} parameters")

    all_runs = []
    for seed_idx in range(args.multi_seed):
        seed = args.seed + seed_idx * 1000
        print(f"\n--- Run with seed={seed} ---")
        results, total_dt = run_full_loso(
            ds, device, n_epochs=args.n_epochs, seed=seed,
            verbose=not args.quiet,
        )
        summary = summarize(results)
        summary["seed"] = seed
        summary["total_time_s"] = total_dt
        all_runs.append(summary)

        print(f"\n  acc={summary['accuracy']:.3f} ({summary['n_correct']}/{summary['n_total']})  "
              f"f1={summary['f1']:.3f}  sens={summary['sensitivity']:.3f}  "
              f"spec={summary['specificity']:.3f}  auc={summary['auc']:.3f}")
        print(f"  probas: min={summary['proba_min']:.3f}, max={summary['proba_max']:.3f}, "
              f"mean={summary['proba_mean']:.3f}, std={summary['proba_std']:.3f}")
        print(f"  conf: TN={summary['tn']} FP={summary['fp']} "
              f"FN={summary['fn']} TP={summary['tp']}")
        print(f"  total time: {total_dt:.1f}s")

        df = pd.DataFrame(results)
        df["seed"] = seed
        df.to_csv(args.out_dir / f"predictions_seed{seed}.csv", index=False)

    if args.multi_seed > 1:
        print(f"\n=== Multi-seed summary ===")
        accs = [r["accuracy"] for r in all_runs]
        aucs = [r["auc"] for r in all_runs]
        print(f"Accuracy:  mean={np.mean(accs):.3f}  std={np.std(accs):.3f}  "
              f"min={min(accs):.3f}  max={max(accs):.3f}")
        print(f"AUC:       mean={np.mean(aucs):.3f}  std={np.std(aucs):.3f}  "
              f"min={min(aucs):.3f}  max={max(aucs):.3f}")

    pd.DataFrame(all_runs).to_csv(args.out_dir / "summary.csv", index=False)
    print(f"\nResults saved to {args.out_dir}/")

    print(f"\n=== Comparison ===")
    print(f"  Custom PatchTST (5 seeds):   acc=0.400 ± 0.077, AUC=0.450 ± 0.085")
    print(f"  Official PatchTST ({args.multi_seed} seed{'s' if args.multi_seed > 1 else ''}): "
          f"acc={np.mean([r['accuracy'] for r in all_runs]):.3f}"
          f"{' ± ' + str(round(np.std([r['accuracy'] for r in all_runs]), 3)) if args.multi_seed > 1 else ''}, "
          f"AUC={np.mean([r['auc'] for r in all_runs]):.3f}"
          f"{' ± ' + str(round(np.std([r['auc'] for r in all_runs]), 3)) if args.multi_seed > 1 else ''}")
    print(f"  Baseline RICH_STATS/GB:      acc=0.792, AUC=0.806")


if __name__ == "__main__":
    main()
