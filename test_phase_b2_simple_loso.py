"""
test_phase_b2_simple_loso.py

Phase B.2 — LOSO simple sur traces brutes (Exp B).

Architecture :
    - Un échantillon = un trait brut (x, y) shape (2, 200)
    - PatchTSTClassifier custom avec init_head_zero=True
    - Train équilibré 1:1 (retire 1 enfant de la classe majoritaire)
    - 30 epochs fixes, pas de validation interne
    - Test sur l'enfant LOSO exclu : prédiction trait par trait,
      agrégation au niveau enfant via mean_proba

Différences clés vs Phase 3.2 (Exp A) :
    - 663 échantillons train (vs 22 séquences) → beaucoup plus de signal
    - n_channels=2 (vs 14)
    - seq_len=200 (vs 20)
    - patch_len=16, stride=8 → 25 patches par trait

OBJECTIF : valider que PatchTST sur traces brutes peut atteindre ou dépasser
le baseline STROKE_14/GB (~42% acc, qui était le baseline trait-par-trait
sur paramètres sigma-lognormaux). Idéalement, se rapprocher de
RICH_STATS/GB (79.2%, qui est le SOTA absolu sur ce dataset).

Durée estimée : ~5-10 min par seed sur M5 Pro (24 folds × ~15-30s).
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

from src.data.exp_b_datasets import (
    RawTraceStrokeDataset, aggregate_stroke_predictions,
)
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.config import OptimConfig
from src.training.device import get_device
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed
from src.utils.traces_io import load_traces


def balance_train_ids(train_ids, labels_per_child, seed=42):
    """Retire 1 enfant de la classe majoritaire pour équilibrer 1:1."""
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
    """Entraînement simple sans validation, n_epochs fixes."""
    model = model.to(device)
    optim_cfg = OptimConfig(lr=lr, weight_decay=weight_decay,
                             warmup_epochs=warmup_epochs, grad_clip=grad_clip)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, n_epochs)
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
        scheduler.step()
    return history


@torch.no_grad()
def predict_strokes(model, loader, device):
    """Inférence trait par trait : retourne un array P(ADHD) par échantillon."""
    model.eval()
    all_probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def run_one_fold(traces, labels_per_child, child_ids, test_child_id,
                  device, n_epochs=30, seed=42, batch_size=32,
                  channels=("x", "y"), target_len=200,
                  min_motion_std=1.0, scale_xy=100.0, pad_strategy="edge"):
    """Un fold LOSO simple sans validation interne."""
    train_pool = [c for c in child_ids if c != test_child_id]
    balanced_ids, removed = balance_train_ids(train_pool, labels_per_child, seed=seed)

    set_global_seed(seed)

    # Datasets
    train_ds = RawTraceStrokeDataset(
        traces=traces, labels_per_child=labels_per_child,
        child_ids=balanced_ids,
        target_len=target_len, channels=channels, center_xy_flag=True,
        min_motion_std=min_motion_std, scale_xy=scale_xy, pad_strategy=pad_strategy,
    )
    test_ds = RawTraceStrokeDataset(
        traces=traces, labels_per_child=labels_per_child,
        child_ids=[test_child_id],
        target_len=target_len, channels=channels, center_xy_flag=True,
        min_motion_std=min_motion_std, scale_xy=scale_xy, pad_strategy=pad_strategy,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0)

    # Modèle
    cfg = PatchTSTConfig(
        n_channels=len(channels), seq_len=target_len,
        patch_len=16, stride=8,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2, init_head_zero=True)

    # Train
    train_simple(model, train_loader, device, n_epochs=n_epochs)

    # Prédiction trait par trait
    stroke_probs = predict_strokes(model, test_loader, device)
    cids_per_stroke = test_ds.child_id_per_item()

    # Agrégation enfant
    _, child_proba = aggregate_stroke_predictions(
        cids_per_stroke, stroke_probs, aggregation="mean_proba",
    )
    test_proba = float(child_proba[0])
    test_pred = int(test_proba >= 0.5)
    test_true = labels_per_child[test_child_id]

    return {
        "test_child": test_child_id,
        "true": test_true, "pred": test_pred,
        "proba_adhd": test_proba,
        "correct": test_pred == test_true,
        "n_test_strokes": len(stroke_probs),
        "stroke_probs_mean": float(stroke_probs.mean()),
        "stroke_probs_std": float(stroke_probs.std()),
        "stroke_probs_min": float(stroke_probs.min()),
        "stroke_probs_max": float(stroke_probs.max()),
        "n_train_balanced": len(balanced_ids),
        "removed": removed or "",
    }


def run_full_loso(traces, labels_per_child, child_ids, device,
                   n_epochs=30, seed=42, batch_size=32,
                   channels=("x", "y"), verbose=True,
                   min_motion_std=1.0, scale_xy=100.0, pad_strategy="edge"):
    """Boucle LOSO complète sur tous les enfants."""
    results = []
    t_start = time.time()
    for cid in child_ids:
        t0 = time.time()
        r = run_one_fold(
            traces, labels_per_child, child_ids, cid,
            device, n_epochs=n_epochs, seed=seed,
            batch_size=batch_size, channels=channels,
            min_motion_std=min_motion_std,
            scale_xy=scale_xy,
            pad_strategy=pad_strategy,
        )
        dt = time.time() - t0
        if verbose:
            ok = "✓" if r["correct"] else "✗"
            tag = "ADHD" if r["true"] == 1 else "CTRL"
            print(f"  {cid} [{tag}]: P(ADHD)={r['proba_adhd']:.4f} "
                  f"pred={'ADHD' if r['pred']==1 else 'CTRL'} {ok}  "
                  f"(n_strokes={r['n_test_strokes']}, stroke_std={r['stroke_probs_std']:.3f}, "
                  f"{dt:.1f}s)")
        results.append(r)
    return results, time.time() - t_start


def summarize(results):
    """Calcule les métriques globales sur tous les folds LOSO."""
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
    parser.add_argument("--multi-seed", type=int, default=1,
                        help="Nombre de seeds (défaut 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--channels", type=str, default="x,y",
                        help="Comma-separated list: 'x,y' or 'x,y,v'")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/exp_b_simple_loso"))
    parser.add_argument("--quiet", action="store_true")
    # ---- Phase B fixes ----
    parser.add_argument("--min-motion-std", type=float, default=1.0,
                        help="Filtre les traits dont std(x) ou std(y) < this (mm). 0 = pas de filtrage.")
    parser.add_argument("--scale-xy", type=float, default=100.0,
                        help="Divise x, y par ce facteur. 100 = coordonnées en cm. 1 = brut en mm.")
    parser.add_argument("--pad-strategy", type=str, default="edge",
                        choices=["zero", "edge"],
                        help="'zero' (zero pad) ou 'edge' (répète dernière valeur, recommandé).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    channels = tuple(args.channels.split(","))

    print(f"=== Phase B.2 — Simple LOSO on raw traces ===")
    print(f"Device: {device}")
    print(f"n_epochs={args.n_epochs}, multi_seed={args.multi_seed}, "
          f"batch_size={args.batch_size}, channels={channels}")
    print(f"Fixes: min_motion_std={args.min_motion_std} mm, "
          f"scale_xy={args.scale_xy}, pad_strategy={args.pad_strategy}")

    # Chargement
    print(f"\nLoading data...")
    traces = load_traces(Path("data/processed"))
    ds_params = load_processed_dataset(Path("data/processed"))
    labels = ds_params.labels_per_child
    print(f"  Loaded {traces.total_strokes} strokes from {len(traces.child_ids)} children")

    # Probe rapide pour rapporter combien de traits seront filtrés
    probe_ds = RawTraceStrokeDataset(
        traces=traces, labels_per_child=labels,
        child_ids=traces.child_ids,
        target_len=200, channels=channels, center_xy_flag=True,
        min_motion_std=args.min_motion_std, scale_xy=args.scale_xy,
        pad_strategy=args.pad_strategy,
    )
    print(f"  After filtering (motion_std < {args.min_motion_std} mm): "
          f"{len(probe_ds)} strokes kept "
          f"({probe_ds.n_filtered_motion} filtered out)")

    # Affiche taille du modèle
    cfg_test = PatchTSTConfig(
        n_channels=len(channels), seq_len=200,
        patch_len=16, stride=8,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model_test = PatchTSTClassifier(cfg_test, n_classes=2, init_head_zero=True)
    print(f"  Model: {model_test.n_parameters():,} parameters")
    del model_test

    all_runs = []
    for seed_idx in range(args.multi_seed):
        seed = args.seed + seed_idx * 1000
        print(f"\n--- Run with seed={seed} ---")
        results, total_dt = run_full_loso(
            traces, labels, traces.child_ids, device,
            n_epochs=args.n_epochs, seed=seed,
            batch_size=args.batch_size, channels=channels,
            verbose=not args.quiet,
            min_motion_std=args.min_motion_std,
            scale_xy=args.scale_xy,
            pad_strategy=args.pad_strategy,
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

        # Sauvegarde des prédictions
        df = pd.DataFrame(results)
        df["seed"] = seed
        df.to_csv(args.out_dir / f"predictions_seed{seed}.csv", index=False)

    # Synthèse multi-seed
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

    # Comparaison avec Exp A et baselines
    print(f"\n=== Comparison ===")
    print(f"  Baseline STROKE_14/GB (per-stroke):     ~42%")
    print(f"  Exp A PatchTST custom (params, 5 seeds): acc=40.0%, AUC=0.450")
    print(f"  Exp A PatchTST official (params):        acc=40.0%, AUC=0.394")
    avg_acc = np.mean([r["accuracy"] for r in all_runs])
    avg_auc = np.mean([r["auc"] for r in all_runs])
    n_seeds_str = f"{args.multi_seed} seed{'s' if args.multi_seed > 1 else ''}"
    print(f"  Exp B PatchTST (raw traces, {n_seeds_str}):  acc={avg_acc:.1%}, AUC={avg_auc:.3f}")
    print(f"  Baseline RICH_STATS/GB (params):         acc=79.2%, AUC=0.806")

    if avg_acc > 0.79:
        print(f"\n  🎉 PatchTST on raw traces BEATS the best baseline!")
    elif avg_acc > 0.65:
        print(f"\n  ✓ PatchTST on raw traces shows clear improvement vs Exp A")
        print(f"     ({avg_acc:.1%} vs 40% on parameters)")
    elif avg_acc > 0.50:
        print(f"\n  ~ PatchTST on raw traces marginally better than Exp A")
    else:
        print(f"\n  ⚠ PatchTST on raw traces did NOT improve vs Exp A")


if __name__ == "__main__":
    main()
