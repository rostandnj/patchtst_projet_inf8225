"""
test_attention_analysis_multiseed.py

Analyse des poids d'attention de PatchTST sur Exp B et Exp C, AGRÉGÉE sur
plusieurs entraînements indépendants pour évaluer la stabilité de l'attention.

DIFFÉRENCE vs version single-seed :
    - Au lieu d'entraîner 1 modèle par expérience, on en entraîne N (default 3)
    - On capture les profils d'attention de CHAQUE modèle
    - On agrège : moyenne + écart-type entre les N runs
    - Les visualisations affichent les bandes de confiance ±1σ

OBJECTIF SCIENTIFIQUE :
    Si les N profils d'attention sont similaires (std faible), le modèle
    apprend de manière reproductible à regarder les mêmes positions
    temporelles → interprétation robuste.

    Si les profils varient beaucoup (std élevée), l'attention dépend de
    l'initialisation aléatoire et ne peut pas être interprétée fiablement.

DURÉE : ~20-25 min pour 3 seeds (Exp B + Exp C × 3 entraînements).
"""

from __future__ import annotations

import os
os.environ["KNM_DEVICE"] = "cpu"

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_b_datasets import RawTraceStrokeDataset
from src.data.exp_c_datasets import RawTraceWithParamsDataset
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.config import OptimConfig
from src.training.device import get_device
from src.training.optimizer import build_optimizer, build_scheduler, clip_gradients
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed
from src.utils.traces_io import load_traces


# =====================================================================
# Capture des poids d'attention via monkey-patching
# =====================================================================

class AttentionCapture:
    """Capture les poids d'attention pendant le forward."""

    def __init__(self, model):
        self.model = model
        self.attentions = []

    def __enter__(self):
        from src.models.transformer import TransformerEncoderLayer
        self.layers = [m for m in self.model.modules()
                        if isinstance(m, TransformerEncoderLayer)]
        self._original_forwards = {}
        for layer in self.layers:
            self._original_forwards[layer] = layer.forward

            def make_patched(layer_ref):
                def patched_forward(x, key_padding_mask=None):
                    x_norm = layer_ref.norm1(x)
                    attn_out, attn_weights = layer_ref.self_attn(
                        x_norm, x_norm, x_norm,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        average_attn_weights=True,
                    )
                    self.attentions.append(attn_weights.detach().cpu())
                    x = x + layer_ref.dropout1(attn_out)
                    x = x + layer_ref.dropout2(layer_ref.ffn(layer_ref.norm2(x)))
                    return x
                return patched_forward

            layer.forward = make_patched(layer)
        return self

    def __exit__(self, *args):
        for layer, original in self._original_forwards.items():
            layer.forward = original


# =====================================================================
# Helpers
# =====================================================================

def balance_train_ids(train_ids, labels_per_child, seed=42):
    labels = [labels_per_child[c] for c in train_ids]
    counts = Counter(labels)
    if counts[0] == counts[1]:
        return list(train_ids), None
    majority = 0 if counts[0] > counts[1] else 1
    rng = np.random.default_rng(seed)
    candidates = [c for c in train_ids if labels_per_child[c] == majority]
    removed = str(rng.choice(candidates))
    return [c for c in train_ids if c != removed], removed


def train_quickly(model, train_loader, device, n_epochs=30):
    model = model.to(device)
    cfg = OptimConfig(lr=1e-3, weight_decay=0.01, warmup_epochs=5, grad_clip=1.0)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, n_epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(model, cfg.grad_clip)
            optimizer.step()
        scheduler.step()


def capture_attentions(model, ds, device, n_layers, n_channels, n_patches):
    """Capture les attentions sur tout le dataset, retourne par classe."""
    model.eval()
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    per_layer_collected = [[] for _ in range(n_layers)]
    labels_per_stroke = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with AttentionCapture(model) as cap:
                _ = model(x)
            for layer_idx in range(n_layers):
                a = cap.attentions[layer_idx]
                B = x.size(0)
                a = a.reshape(B, n_channels, n_patches, n_patches)
                per_layer_collected[layer_idx].append(a)
            labels_per_stroke.extend(y.tolist())

    per_layer = [torch.cat(per_layer_collected[i], dim=0)
                  for i in range(n_layers)]
    labels_arr = np.array(labels_per_stroke)
    ctrl_mask = labels_arr == 0
    adhd_mask = labels_arr == 1

    attns_ctrl = [a[ctrl_mask] for a in per_layer]
    attns_adhd = [a[adhd_mask] for a in per_layer]
    meta = {
        "n_ctrl_strokes": int(ctrl_mask.sum()),
        "n_adhd_strokes": int(adhd_mask.sum()),
        "n_layers": n_layers,
        "n_channels": n_channels,
        "n_patches": n_patches,
    }
    return attns_ctrl, attns_adhd, meta


# =====================================================================
# Pipeline pour un modèle, multi-seed
# =====================================================================

def train_and_capture_one_seed(traces, params_ds, labels, child_ids,
                                  test_child_id, model_type, device,
                                  n_epochs=30, seed=42):
    """Entraîne 1 fold avec une seed donnée et capture les attentions."""
    train_pool = [c for c in child_ids if c != test_child_id]
    balanced_ids, _ = balance_train_ids(train_pool, labels, seed=seed)
    set_global_seed(seed)

    if model_type == "ExpB":
        train_ds = RawTraceStrokeDataset(
            traces=traces, labels_per_child=labels, child_ids=balanced_ids,
            target_len=200, channels=("x", "y"), center_xy_flag=True,
            min_motion_std=1.0, scale_xy=100.0, pad_strategy="edge",
        )
        eval_ds = RawTraceStrokeDataset(
            traces=traces, labels_per_child=labels, child_ids=child_ids,
            target_len=200, channels=("x", "y"), center_xy_flag=True,
            min_motion_std=1.0, scale_xy=100.0, pad_strategy="edge",
        )
        n_channels = 2
    else:  # ExpC
        train_ds = RawTraceWithParamsDataset(
            traces=traces, params_dataset=params_ds, labels_per_child=labels,
            child_ids=balanced_ids,
            target_len=200, trace_channels=("x", "y"), center_xy_flag=True,
            scale_xy=100.0, pad_strategy="edge", min_motion_std=1.0,
        )
        eval_ds = RawTraceWithParamsDataset(
            traces=traces, params_dataset=params_ds, labels_per_child=labels,
            child_ids=child_ids,
            target_len=200, trace_channels=("x", "y"), center_xy_flag=True,
            scale_xy=100.0, pad_strategy="edge", min_motion_std=1.0,
        )
        n_channels = 16

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    cfg = PatchTSTConfig(
        n_channels=n_channels, seq_len=200,
        patch_len=16, stride=8,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        dropout=0.2, attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2, init_head_zero=True)
    train_quickly(model, train_loader, device, n_epochs=n_epochs)

    from src.models.patching import compute_n_patches
    n_patches = compute_n_patches(200, 16, 8)

    attns_ctrl, attns_adhd, meta = capture_attentions(
        model, eval_ds, device, n_layers=2,
        n_channels=n_channels, n_patches=n_patches,
    )
    return attns_ctrl, attns_adhd, meta


def aggregate_multi_seed(traces, params_ds, labels, child_ids,
                           test_child_id, model_type, device,
                           n_epochs, seeds):
    """Lance N entraînements et agrège les profils d'attention.

    Returns:
        results : dict avec
            - 'temporal_ctrl': (n_seeds, n_layers, n_patches) profil temporel CTRL
            - 'temporal_adhd': (n_seeds, n_layers, n_patches) profil temporel ADHD
            - 'temporal_ctrl_traces': (n_seeds, n_layers, n_patches) traces only (Exp C)
            - 'temporal_adhd_traces': idem ADHD
            - 'channel_concentration': (n_seeds, n_layers, n_channels) pour Exp C
            - 'meta': dict de méta-données
    """
    n_seeds = len(seeds)
    print(f"\n{'='*70}")
    print(f"  {model_type} multi-seed analysis ({n_seeds} seeds)")
    print(f"{'='*70}")

    all_temporal_ctrl = []  # liste de (n_layers, n_patches) — moyenne sur tous canaux
    all_temporal_adhd = []
    all_temporal_ctrl_traces = []  # uniquement canaux 0,1 (traces)
    all_temporal_adhd_traces = []
    all_channel_concentration = []  # (n_layers, n_channels)

    meta_ref = None
    for seed_idx, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n  Seed {seed} ({seed_idx+1}/{n_seeds})...")
        attns_ctrl, attns_adhd, meta = train_and_capture_one_seed(
            traces, params_ds, labels, child_ids, test_child_id,
            model_type, device, n_epochs=n_epochs, seed=seed,
        )
        meta_ref = meta
        n_layers = meta["n_layers"]
        n_channels = meta["n_channels"]

        # Profil temporel : attention reçue par chaque patch (moyenne sur axe query)
        # Tensor shape: (n_strokes, M, N, N) -> (n_layers, n_patches)
        temporal_ctrl = []
        temporal_adhd = []
        temporal_ctrl_traces = []
        temporal_adhd_traces = []
        for layer_idx in range(n_layers):
            # Reçue par patch i = mean over query axis
            ac_recv = attns_ctrl[layer_idx].mean(dim=2)  # (n_strokes, M, N)
            aa_recv = attns_adhd[layer_idx].mean(dim=2)
            # Moyenne sur strokes et tous canaux
            temporal_ctrl.append(ac_recv.mean(dim=(0, 1)).numpy())  # (N,)
            temporal_adhd.append(aa_recv.mean(dim=(0, 1)).numpy())
            # Traces only (canaux 0, 1)
            temporal_ctrl_traces.append(ac_recv[:, :2, :].mean(dim=(0, 1)).numpy())
            temporal_adhd_traces.append(aa_recv[:, :2, :].mean(dim=(0, 1)).numpy())

        all_temporal_ctrl.append(np.stack(temporal_ctrl))  # (n_layers, n_patches)
        all_temporal_adhd.append(np.stack(temporal_adhd))
        all_temporal_ctrl_traces.append(np.stack(temporal_ctrl_traces))
        all_temporal_adhd_traces.append(np.stack(temporal_adhd_traces))

        # Channel concentration (uniquement pour Exp C, mais on calcule toujours)
        channel_conc = []
        for layer_idx in range(n_layers):
            all_attn = torch.cat([attns_ctrl[layer_idx], attns_adhd[layer_idx]], dim=0)
            eps = 1e-9
            entropy = -(all_attn * torch.log(all_attn + eps)).sum(dim=-1)  # (n_strokes, M, N)
            entropy_per_channel = entropy.mean(dim=(0, 2))  # (M,)
            max_entropy = float(np.log(meta["n_patches"]))
            concentration = 1 - (entropy_per_channel / max_entropy)
            channel_conc.append(concentration.numpy())
        all_channel_concentration.append(np.stack(channel_conc))  # (n_layers, n_channels)

        dt = time.time() - t0
        print(f"    ✓ Done in {dt:.1f}s")

    return {
        "temporal_ctrl": np.stack(all_temporal_ctrl),  # (n_seeds, n_layers, n_patches)
        "temporal_adhd": np.stack(all_temporal_adhd),
        "temporal_ctrl_traces": np.stack(all_temporal_ctrl_traces),
        "temporal_adhd_traces": np.stack(all_temporal_adhd_traces),
        "channel_concentration": np.stack(all_channel_concentration),
        "meta": meta_ref,
        "n_seeds": n_seeds,
        "seeds": seeds,
    }


# =====================================================================
# Visualisations
# =====================================================================

def plot_temporal_profile_multiseed(results, model_name, out_dir):
    """Profil temporel avec bandes de confiance ±1σ entre seeds."""
    n_seeds = results["n_seeds"]
    meta = results["meta"]
    n_layers = meta["n_layers"]
    n_patches = meta["n_patches"]

    fig, axes = plt.subplots(n_layers, 1, figsize=(11, 4 * n_layers), squeeze=False)

    for layer_idx in range(n_layers):
        # CTRL : (n_seeds, n_patches)
        ctrl_runs = results["temporal_ctrl"][:, layer_idx, :]
        adhd_runs = results["temporal_adhd"][:, layer_idx, :]

        ctrl_mean = ctrl_runs.mean(axis=0)
        ctrl_std = ctrl_runs.std(axis=0)
        adhd_mean = adhd_runs.mean(axis=0)
        adhd_std = adhd_runs.std(axis=0)

        ax = axes[layer_idx, 0]
        x = np.arange(n_patches)

        # CTRL
        ax.plot(x, ctrl_mean, "o-", label=f"CTRL (mean over {n_seeds} seeds)",
                color="steelblue", linewidth=2)
        ax.fill_between(x, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                         alpha=0.25, color="steelblue", label="CTRL ±1σ")
        # Lignes individuelles fines
        for s_idx in range(n_seeds):
            ax.plot(x, ctrl_runs[s_idx], "-", color="steelblue", alpha=0.3, linewidth=0.8)

        # ADHD
        ax.plot(x, adhd_mean, "s-", label=f"ADHD (mean over {n_seeds} seeds)",
                color="crimson", linewidth=2)
        ax.fill_between(x, adhd_mean - adhd_std, adhd_mean + adhd_std,
                         alpha=0.25, color="crimson", label="ADHD ±1σ")
        for s_idx in range(n_seeds):
            ax.plot(x, adhd_runs[s_idx], "-", color="crimson", alpha=0.3, linewidth=0.8)

        # Référence uniforme
        ax.axhline(1.0 / n_patches, color="gray", linestyle="--", alpha=0.5,
                    label=f"Uniform (1/{n_patches})")

        # Titre avec stabilité
        ctrl_inter_seed_std = ctrl_runs.std(axis=0).mean()
        adhd_inter_seed_std = adhd_runs.std(axis=0).mean()
        ax.set_title(f"{model_name} Layer {layer_idx} — Temporal attention profile "
                      f"(over {n_seeds} seeds)\n"
                      f"Inter-seed std: CTRL={ctrl_inter_seed_std:.4f}, "
                      f"ADHD={adhd_inter_seed_std:.4f}")
        ax.set_xlabel("Patch index (temporal position)")
        ax.set_ylabel("Mean attention received")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"temporal_profile_{model_name}_multiseed.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


def plot_comparison_b_vs_c_multiseed(results_b, results_c, out_dir):
    """Comparaison directe B vs C avec bandes de confiance."""
    n_patches = results_b["meta"]["n_patches"]
    last = results_b["meta"]["n_layers"] - 1

    # Profils traces only (pour comparer pommes avec pommes)
    b_ctrl = results_b["temporal_ctrl_traces"][:, last, :]  # (n_seeds, n_patches)
    b_adhd = results_b["temporal_adhd_traces"][:, last, :]
    c_ctrl = results_c["temporal_ctrl_traces"][:, last, :]
    c_adhd = results_c["temporal_adhd_traces"][:, last, :]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(n_patches)

    # CTRL
    ax = axes[0]
    b_mean, b_std = b_ctrl.mean(axis=0), b_ctrl.std(axis=0)
    c_mean, c_std = c_ctrl.mean(axis=0), c_ctrl.std(axis=0)
    ax.plot(x, b_mean, "o-", label="Exp B (traces only)", color="steelblue", linewidth=2)
    ax.fill_between(x, b_mean - b_std, b_mean + b_std, alpha=0.25, color="steelblue")
    ax.plot(x, c_mean, "s--", label="Exp C (traces+params, traces channels only)",
            color="darkorange", linewidth=2)
    ax.fill_between(x, c_mean - c_std, c_mean + c_std, alpha=0.25, color="darkorange")
    ax.axhline(1.0 / n_patches, color="gray", linestyle=":", alpha=0.5, label="Uniform")
    ax.set_title(f"CTRL — Last layer attention profile\n"
                  f"(traces channels x, y only, mean over seeds)")
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Mean attention received")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ADHD
    ax = axes[1]
    b_mean, b_std = b_adhd.mean(axis=0), b_adhd.std(axis=0)
    c_mean, c_std = c_adhd.mean(axis=0), c_adhd.std(axis=0)
    ax.plot(x, b_mean, "o-", label="Exp B (traces only)", color="steelblue", linewidth=2)
    ax.fill_between(x, b_mean - b_std, b_mean + b_std, alpha=0.25, color="steelblue")
    ax.plot(x, c_mean, "s--", label="Exp C (traces+params, traces channels only)",
            color="darkorange", linewidth=2)
    ax.fill_between(x, c_mean - c_std, c_mean + c_std, alpha=0.25, color="darkorange")
    ax.axhline(1.0 / n_patches, color="gray", linestyle=":", alpha=0.5, label="Uniform")
    ax.set_title(f"ADHD — Last layer attention profile\n"
                  f"(traces channels x, y only, mean over seeds)")
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Mean attention received")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "comparison_B_vs_C_multiseed.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


def plot_channel_importance_multiseed(results, out_dir):
    """Importance par canal pour Exp C, avec barres d'erreur entre seeds."""
    meta = results["meta"]
    n_layers = meta["n_layers"]
    n_channels = meta["n_channels"]
    if n_channels < 16:
        # Pas pertinent pour Exp B
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)
    channel_labels = ["x", "y"] + [f"p{i}" for i in range(14)]
    colors = ["steelblue", "steelblue"] + ["seagreen"] * 14

    for layer_idx in range(n_layers):
        # (n_seeds, n_channels)
        conc = results["channel_concentration"][:, layer_idx, :]
        conc_mean = conc.mean(axis=0)
        conc_std = conc.std(axis=0)

        ax = axes[0, layer_idx]
        bars = ax.bar(range(n_channels), conc_mean, yerr=conc_std, capsize=4,
                       color=colors[:n_channels], alpha=0.8,
                       error_kw={"elinewidth": 1.5, "ecolor": "black"})
        ax.set_xticks(range(n_channels))
        ax.set_xticklabels(channel_labels[:n_channels], rotation=45, fontsize=9)
        ax.set_title(f"Exp C Layer {layer_idx} — Attention concentration by channel\n"
                      f"(mean ±std over {results['n_seeds']} seeds)")
        ax.set_ylabel("Concentration (1 - normalized entropy)")
        ax.grid(alpha=0.3, axis="y")

        # Annotation : top-3 canaux
        top3_idx = np.argsort(-conc_mean)[:3]
        for i, idx in enumerate(top3_idx):
            ax.annotate(f"#{i+1}", xy=(idx, conc_mean[idx]),
                          xytext=(0, 5), textcoords="offset points",
                          ha="center", fontsize=9, fontweight="bold", color="red")

    plt.tight_layout()
    out_path = out_dir / "channel_importance_C_multiseed.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path}")


# =====================================================================
# Stats & Summary
# =====================================================================

def print_stability_summary(results, model_name):
    """Imprime les stats de stabilité inter-seed."""
    meta = results["meta"]
    n_layers = meta["n_layers"]

    print(f"\n=== Stability summary: {model_name} ===")
    for layer_idx in range(n_layers):
        ctrl_runs = results["temporal_ctrl"][:, layer_idx, :]
        adhd_runs = results["temporal_adhd"][:, layer_idx, :]

        ctrl_mean = ctrl_runs.mean(axis=0)
        ctrl_std_inter = ctrl_runs.std(axis=0).mean()
        adhd_mean = adhd_runs.mean(axis=0)
        adhd_std_inter = adhd_runs.std(axis=0).mean()

        print(f"  Layer {layer_idx}:")
        print(f"    CTRL profile : argmax_patch={int(np.argmax(ctrl_mean))}, "
              f"max={ctrl_mean.max():.4f}, "
              f"inter_seed_std={ctrl_std_inter:.4f}")
        print(f"    ADHD profile : argmax_patch={int(np.argmax(adhd_mean))}, "
              f"max={adhd_mean.max():.4f}, "
              f"inter_seed_std={adhd_std_inter:.4f}")

        # Ratio max/uniform pour évaluer la concentration
        uniform = 1.0 / meta["n_patches"]
        concentration_ctrl = ctrl_mean.max() / uniform
        concentration_adhd = adhd_mean.max() / uniform
        print(f"    Concentration ratio (max/uniform): "
              f"CTRL={concentration_ctrl:.2f}x, ADHD={concentration_adhd:.2f}x")


def save_numerical_results(results_b, results_c, out_dir):
    """Sauvegarde les profils numériques en CSV pour réutilisation."""
    import pandas as pd
    rows = []
    for model_name, results in [("ExpB", results_b), ("ExpC", results_c)]:
        meta = results["meta"]
        for layer_idx in range(meta["n_layers"]):
            for patch_idx in range(meta["n_patches"]):
                ctrl_runs = results["temporal_ctrl"][:, layer_idx, patch_idx]
                adhd_runs = results["temporal_adhd"][:, layer_idx, patch_idx]
                rows.append({
                    "model": model_name,
                    "layer": layer_idx,
                    "patch": patch_idx,
                    "ctrl_mean": float(ctrl_runs.mean()),
                    "ctrl_std": float(ctrl_runs.std()),
                    "adhd_mean": float(adhd_runs.mean()),
                    "adhd_std": float(adhd_runs.std()),
                    "n_seeds": results["n_seeds"],
                })
    df = pd.DataFrame(rows)
    out_path = out_dir / "attention_profiles_multiseed.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved numerical results to {out_path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-child", type=str, default="S01")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of independent training runs per model")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/attention_analysis_multiseed_one_child"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    seeds = [args.seed_base + 1000 * i for i in range(args.n_seeds)]

    print(f"=== Attention analysis: Exp B vs Exp C (multi-seed) ===")
    print(f"Device: {device}")
    print(f"Test child: {args.test_child}")
    print(f"N seeds: {args.n_seeds} → seeds={seeds}")
    print(f"N epochs per training: {args.n_epochs}")
    print(f"Output dir: {args.out_dir}")
    print(f"Total trainings: {2 * args.n_seeds} (Exp B × {args.n_seeds} + Exp C × {args.n_seeds})")
    print(f"Estimated duration: ~{2 * args.n_seeds * 4} min")

    print(f"\nLoading data...")
    traces = load_traces(Path("data/processed"))
    params_ds = load_processed_dataset(Path("data/processed"))
    labels = params_ds.labels_per_child

    if args.test_child not in traces.child_ids:
        args.test_child = traces.child_ids[0]
        print(f"⚠ Falling back to {args.test_child}")

    t_global = time.time()

    # ========== Exp B multi-seed ==========
    results_b = aggregate_multi_seed(
        traces, params_ds, labels, traces.child_ids,
        args.test_child, "ExpB", device,
        n_epochs=args.n_epochs, seeds=seeds,
    )

    # ========== Exp C multi-seed ==========
    results_c = aggregate_multi_seed(
        traces, params_ds, labels, traces.child_ids,
        args.test_child, "ExpC", device,
        n_epochs=args.n_epochs, seeds=seeds,
    )

    print(f"\n{'='*70}")
    print(f"  Generating visualizations")
    print(f"{'='*70}")

    plot_temporal_profile_multiseed(results_b, "ExpB", args.out_dir)
    plot_temporal_profile_multiseed(results_c, "ExpC", args.out_dir)
    plot_comparison_b_vs_c_multiseed(results_b, results_c, args.out_dir)
    plot_channel_importance_multiseed(results_c, args.out_dir)
    save_numerical_results(results_b, results_c, args.out_dir)

    print_stability_summary(results_b, "ExpB")
    print_stability_summary(results_c, "ExpC")

    total_time = time.time() - t_global
    print(f"\n✓ Total execution time: {total_time / 60:.1f} min")
    print(f"✓ All results saved in {args.out_dir}/")
    print(f"  Files generated:")
    for f in sorted(args.out_dir.glob("*")):
        print(f"    - {f.name}")


if __name__ == "__main__":
    main()
