"""
test_phase_c1_simple_loso.py

Phase C.1 — LOSO simple sur traces brutes + paramètres sigma-lognormaux
combinés en 16 canaux.

DESIGN : pour chaque trait, l'input est (16, 200) où :
    - Canaux 0-1 : x(t), y(t) avec scale_xy=100, mean centering, pad edge
    - Canaux 2-15 : 14 paramètres sigma-lognormaux diffusés constants

PROTOCOLE : identique à test_phase_b2_simple_loso.py
    - Train équilibré 1:1 (11 CTRL + 11 ADHD = 22 enfants)
    - 30 epochs fixes, pas de validation interne
    - Backend CPU forcé (KNM_DEVICE=cpu) pour éviter le bug MPS
    - Multi-seed pour estimer la variance

OBJECTIFS DE COMPARAISON :
    - Exp A PatchTST (params, 5 seeds): acc=40.0%, AUC=0.450 (simple LOSO sur séquences)
    - Exp B PatchTST (traces, 5 seeds): acc=77.5%, AUC=0.842 (référence)
    - Exp C PatchTST (combiné, ?? seeds): acc=??, AUC=??

Si Exp C > Exp B : la combinaison apporte un signal additionnel.
Si Exp C ≈ Exp B : les paramètres sont redondants avec ce que PatchTST
                   apprend déjà des traces brutes.
Si Exp C < Exp B : les paramètres ajoutent du bruit, le modèle est
                   distrait par les canaux constants.

Durée estimée : ~3 min single-seed, ~15 min multi-seed 5 sur CPU.
"""

from __future__ import annotations

# Force CPU AVANT tout import torch
import os
os.environ["KNM_DEVICE"] = "cpu"

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

from src.data.exp_b_datasets import aggregate_stroke_predictions
from src.data.exp_c_datasets import RawTraceWithParamsDataset
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.config import OptimConfig
from src.training.device import get_device
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed
from src.utils.traces_io import load_traces


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
        history["train_loss"].append(total_loss / max(1, total_n))
        history["train_acc"].append(total_correct / max(1, total_n))
        scheduler.step()
    return history


@torch.no_grad()
def predict_strokes(model, loader, device):
    model.eval()
    all_probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def run_one_fold(traces, params_ds, labels_per_child, child_ids, test_child_id,
                  device, n_epochs=30, seed=42, batch_size=32, target_len=200):
    train_pool = [c for c in child_ids if c != test_child_id]
    balanced_ids, removed = balance_train_ids(train_pool, labels_per_child, seed=seed)

    set_global_seed(seed)

    train_ds = RawTraceWithParamsDataset(
        traces=traces, params_dataset=params_ds, labels_per_child=labels_per_child,
        child_ids=balanced_ids,
        target_len=target_len, trace_channels=("x", "y"),
        center_xy_flag=True, scale_xy=100.0, pad_strategy="edge",
        min_motion_std=1.0,
    )
    test_ds = RawTraceWithParamsDataset(
        traces=traces, params_dataset=params_ds, labels_per_child=labels_per_child,
        child_ids=[test_child_id],
        target_len=target_len, trace_channels=("x", "y"),
        center_xy_flag=True, scale_xy=100.0, pad_strategy="edge",
        min_motion_std=1.0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    cfg = PatchTSTConfig(
        n_channels=16, seq_len=target_len,  # 2 traces + 14 params
        patch_len=16, stride=8,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2, init_head_zero=True)

    train_simple(model, train_loader, device, n_epochs=n_epochs)

    stroke_probs = predict_strokes(model, test_loader, device)
    cids_per_stroke = test_ds.child_id_per_item()
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
    }


def run_full_loso(traces, params_ds, labels_per_child, child_ids, device,
                   n_epochs=30, seed=42, batch_size=32, verbose=True):
    results = []
    t_start = time.time()
    for cid in child_ids:
        t0 = time.time()
        r = run_one_fold(
            traces, params_ds, labels_per_child, child_ids, cid,
            device, n_epochs=n_epochs, seed=seed, batch_size=batch_size,
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/exp_c_simple_loso"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"=== Phase C.1 — Simple LOSO on traces + sigma-lognormal params ===")
    print(f"Device: {device}")
    print(f"n_epochs={args.n_epochs}, multi_seed={args.multi_seed}, "
          f"batch_size={args.batch_size}")
    print(f"Channels: 2 (traces x,y) + 14 (sigma-lognormal params) = 16")

    print(f"\nLoading data...")
    traces = load_traces(Path("data/processed"))
    params_ds = load_processed_dataset(Path("data/processed"))
    labels = params_ds.labels_per_child
    print(f"  Loaded {traces.total_strokes} strokes from {len(traces.child_ids)} children")

    # Probe rapide
    probe_ds = RawTraceWithParamsDataset(
        traces=traces, params_dataset=params_ds, labels_per_child=labels,
        child_ids=traces.child_ids,
        target_len=200, trace_channels=("x", "y"),
        center_xy_flag=True, scale_xy=100.0, pad_strategy="edge",
        min_motion_std=1.0,
    )
    print(f"  After filtering: {len(probe_ds)} strokes "
          f"(motion: {probe_ds.n_filtered_motion}, missing params: {probe_ds.n_missing_params})")
    print(f"  n_total_channels: {probe_ds.n_total_channels}")

    cfg_test = PatchTSTConfig(
        n_channels=16, seq_len=200,
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
            traces, params_ds, labels, traces.child_ids, device,
            n_epochs=args.n_epochs, seed=seed,
            batch_size=args.batch_size,
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
        senss = [r["sensitivity"] for r in all_runs]
        specs = [r["specificity"] for r in all_runs]
        print(f"Accuracy:    mean={np.mean(accs):.3f}  std={np.std(accs):.3f}  "
              f"min={min(accs):.3f}  max={max(accs):.3f}")
        print(f"AUC:         mean={np.mean(aucs):.3f}  std={np.std(aucs):.3f}  "
              f"min={min(aucs):.3f}  max={max(aucs):.3f}")
        print(f"Sensitivity: mean={np.mean(senss):.3f}  std={np.std(senss):.3f}")
        print(f"Specificity: mean={np.mean(specs):.3f}  std={np.std(specs):.3f}")

    pd.DataFrame(all_runs).to_csv(args.out_dir / "summary.csv", index=False)
    print(f"\nResults saved to {args.out_dir}/")

    print(f"\n=== Comparison ===")
    print(f"  Baseline RICH_STATS/GB:                 acc=79.2%, AUC=0.806")
    print(f"  Exp A PatchTST (params, 5 seeds):       acc=40.0%, AUC=0.450")
    print(f"  Exp B PatchTST (traces, 5 seeds):       acc=77.5%, AUC=0.842")
    avg_acc = np.mean([r["accuracy"] for r in all_runs])
    avg_auc = np.mean([r["auc"] for r in all_runs])
    n_str = f"{args.multi_seed} seed{'s' if args.multi_seed > 1 else ''}"
    print(f"  Exp C PatchTST (traces+params, {n_str}): acc={avg_acc:.1%}, AUC={avg_auc:.3f}")

    diff_b = avg_acc - 0.775
    if avg_acc > 0.80:
        print(f"\n  🎉 Exp C BEATS Exp B and approaches/beats best baseline!")
    elif avg_acc > 0.78:
        print(f"\n  ✓ Exp C slightly improves over Exp B (+{diff_b*100:.1f} pts acc)")
    elif abs(avg_acc - 0.775) < 0.02:
        print(f"\n  ~ Exp C is statistically equivalent to Exp B")
        print(f"     → params are redundant with what PatchTST learns from traces")
    else:
        print(f"\n  ⚠ Exp C is WORSE than Exp B ({diff_b*100:+.1f} pts acc)")
        print(f"     → params are introducing noise")


if __name__ == "__main__":
    main()
