"""
test_attention_analysis_loso.py

Analyse des poids d'attention de PatchTST en protocole LOSO STRICT × multi-seed.

DIFFÉRENCE vs version multiseed simple :
    - Version précédente : 1 fold (S01 fixe) × N seeds → mesure variance d'init
    - Cette version : 24 folds LOSO × N seeds → rigueur LOSO + variance d'init

PROTOCOLE :
    Pour chaque seed :
        Pour chaque enfant test_child in [S01, ..., S26] :
            1. Train sur 22 enfants équilibrés (test_child exclu)
            2. Capture l'attention sur les traits de test_child UNIQUEMENT
               (= rigueur LOSO : aucune fuite de données)
        Agrège les profils d'attention par classe (CTRL/ADHD)

MÉTHODOLOGIE D'AGRÉGATION :
    - Niveau 1 : pour chaque (seed, fold), un profil d'attention par patch
                 calculé sur les traits de test_child
    - Niveau 2 : agrégation par classe sur les 24 folds (12 CTRL + 12 ADHD)
    - Niveau 3 : agrégation finale sur les N seeds

VISUALISATIONS :
    Identiques à la version single-fold mais avec :
        - Bandes ±1σ inter-seed
        - Lignes individuelles par seed
        - Stats numériques détaillées (variance fold + variance seed)

DURÉE : ~30 min pour 3 seeds × 24 folds × 2 modèles = 144 entraînements.

USAGE :
    python test_attention_analysis_loso.py                    # 3 seeds, full LOSO
    python test_attention_analysis_loso.py --n-seeds 5        # 5 seeds (~50 min)
    python test_attention_analysis_loso.py --n-seeds 1        # 1 seed (~10 min)
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


def capture_attentions_on_test(model, test_ds, device, n_layers, n_channels, n_patches):
    """Capture les attentions UNIQUEMENT sur l'enfant test (rigueur LOSO).

    Returns:
        per_layer : list[Tensor] de shape (n_test_strokes, n_channels, n_patches, n_patches)
    """
    model.eval()
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    per_layer_collected = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            with AttentionCapture(model) as cap:
                _ = model(x)
            for layer_idx in range(n_layers):
                a = cap.attentions[layer_idx]
                B = x.size(0)
                a = a.reshape(B, n_channels, n_patches, n_patches)
                per_layer_collected[layer_idx].append(a)

    return [torch.cat(per_layer_collected[i], dim=0) for i in range(n_layers)]


# =====================================================================
# Pipeline pour un fold
# =====================================================================

def run_one_fold_with_capture(traces, params_ds, labels, child_ids,
                                test_child_id, model_type, device,
                                n_epochs=30, seed=42):
    """Entraîne 1 fold LOSO + capture les attentions sur l'enfant test."""
    train_pool = [c for c in child_ids if c != test_child_id]
    balanced_ids, _ = balance_train_ids(train_pool, labels, seed=seed)
    set_global_seed(seed)

    if model_type == "ExpB":
        train_ds = RawTraceStrokeDataset(
            traces=traces, labels_per_child=labels, child_ids=balanced_ids,
            target_len=200, channels=("x", "y"), center_xy_flag=True,
            min_motion_std=1.0, scale_xy=100.0, pad_strategy="edge",
        )
        test_ds = RawTraceStrokeDataset(
            traces=traces, labels_per_child=labels, child_ids=[test_child_id],
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
        test_ds = RawTraceWithParamsDataset(
            traces=traces, params_dataset=params_ds, labels_per_child=labels,
            child_ids=[test_child_id],
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

    attns_per_layer = capture_attentions_on_test(
        model, test_ds, device,
        n_layers=2, n_channels=n_channels, n_patches=n_patches,
    )
    return attns_per_layer, n_channels, n_patches


# =====================================================================
# Agrégation LOSO complet × multi-seed
# =====================================================================

def aggregate_loso_multiseed(traces, params_ds, labels, child_ids,
                                model_type, device, n_epochs, seeds):
    """Pour chaque seed, lance les 24 folds LOSO et agrège.

    Returns dict avec :
        'temporal_per_seed' : (n_seeds, n_layers, n_patches) profil temporel
                              CTRL agrégé sur les 12 folds CTRL, par seed
        'temporal_per_seed_adhd' : idem pour ADHD
        'temporal_per_seed_traces' : (n_seeds, n_layers, n_patches)
                                      uniquement canaux 0,1
        'temporal_per_seed_traces_adhd' : idem
        'channel_concentration_per_seed' : (n_seeds, n_layers, n_channels)
        'meta' : dict
    """
    n_seeds = len(seeds)
    n_children = len(child_ids)
    print(f"\n{'='*72}")
    print(f"  {model_type} LOSO × multi-seed ({n_seeds} seeds × {n_children} folds)")
    print(f"{'='*72}")

    # Stockage final : par seed
    seed_temporal_ctrl = []  # liste de (n_layers, n_patches)
    seed_temporal_adhd = []
    seed_temporal_ctrl_traces = []  # canaux 0,1 uniquement
    seed_temporal_adhd_traces = []
    seed_channel_conc = []

    meta_ref = None

    for seed_idx, seed in enumerate(seeds):
        t_seed = time.time()
        print(f"\n  ── Seed {seed} ({seed_idx+1}/{n_seeds}) ──")

        # Stockage par fold pour cette seed
        # Pour chaque couche : liste de (n_test_strokes, M, N, N)
        # On va agréger par classe à la fin
        fold_attns_per_layer = [[], []]  # [layer0, layer1] -> list of tensors per fold
        fold_labels = []  # 0 = CTRL, 1 = ADHD pour chaque fold

        for cid_idx, cid in enumerate(child_ids):
            t_fold = time.time()
            attns_per_layer, n_channels, n_patches = run_one_fold_with_capture(
                traces, params_ds, labels, child_ids, cid,
                model_type, device, n_epochs=n_epochs, seed=seed,
            )
            for layer_idx in range(2):
                fold_attns_per_layer[layer_idx].append(attns_per_layer[layer_idx])
            fold_labels.append(labels[cid])
            dt = time.time() - t_fold
            if cid_idx == 0 or (cid_idx + 1) % 6 == 0 or cid_idx == n_children - 1:
                print(f"    fold {cid_idx+1}/{n_children} ({cid}): {dt:.1f}s")

        # Agrégation par classe pour cette seed
        # Pour chaque couche, on a 24 tensors (un par fold)
        ctrl_indices = [i for i, lab in enumerate(fold_labels) if lab == 0]
        adhd_indices = [i for i, lab in enumerate(fold_labels) if lab == 1]

        # Profil temporel par couche : moyenne sur folds (pondéré par n_strokes par fold)
        layer_temporal_ctrl = []
        layer_temporal_adhd = []
        layer_temporal_ctrl_traces = []
        layer_temporal_adhd_traces = []
        layer_channel_conc = []

        for layer_idx in range(2):
            # Concat tous les traits CTRL (puis ADHD) sur tous les folds
            ctrl_tensors = [fold_attns_per_layer[layer_idx][i] for i in ctrl_indices]
            adhd_tensors = [fold_attns_per_layer[layer_idx][i] for i in adhd_indices]
            all_ctrl = torch.cat(ctrl_tensors, dim=0)  # (n_total_ctrl_strokes, M, N, N)
            all_adhd = torch.cat(adhd_tensors, dim=0)

            # Profil temporel : attention reçue par patch
            ac_recv = all_ctrl.mean(dim=2)  # (n_strokes, M, N)
            aa_recv = all_adhd.mean(dim=2)

            # Tous canaux confondus
            layer_temporal_ctrl.append(ac_recv.mean(dim=(0, 1)).numpy())  # (N,)
            layer_temporal_adhd.append(aa_recv.mean(dim=(0, 1)).numpy())

            # Traces uniquement (canaux 0, 1)
            layer_temporal_ctrl_traces.append(ac_recv[:, :2, :].mean(dim=(0, 1)).numpy())
            layer_temporal_adhd_traces.append(aa_recv[:, :2, :].mean(dim=(0, 1)).numpy())

            # Concentration par canal
            all_attn = torch.cat([all_ctrl, all_adhd], dim=0)
            eps = 1e-9
            entropy = -(all_attn * torch.log(all_attn + eps)).sum(dim=-1)
            entropy_per_channel = entropy.mean(dim=(0, 2))  # (M,)
            max_entropy = float(np.log(n_patches))
            concentration = 1 - (entropy_per_channel / max_entropy)
            layer_channel_conc.append(concentration.numpy())

        seed_temporal_ctrl.append(np.stack(layer_temporal_ctrl))
        seed_temporal_adhd.append(np.stack(layer_temporal_adhd))
        seed_temporal_ctrl_traces.append(np.stack(layer_temporal_ctrl_traces))
        seed_temporal_adhd_traces.append(np.stack(layer_temporal_adhd_traces))
        seed_channel_conc.append(np.stack(layer_channel_conc))

        meta_ref = {
            "n_layers": 2,
            "n_channels": n_channels,
            "n_patches": n_patches,
            "n_ctrl_folds": len(ctrl_indices),
            "n_adhd_folds": len(adhd_indices),
            "n_total_ctrl_strokes": int(all_ctrl.shape[0]),
            "n_total_adhd_strokes": int(all_adhd.shape[0]),
        }

        dt_seed = time.time() - t_seed
        print(f"  ── Seed {seed} done in {dt_seed/60:.1f} min ──")

    return {
        "temporal_ctrl": np.stack(seed_temporal_ctrl),  # (n_seeds, n_layers, n_patches)
        "temporal_adhd": np.stack(seed_temporal_adhd),
        "temporal_ctrl_traces": np.stack(seed_temporal_ctrl_traces),
        "temporal_adhd_traces": np.stack(seed_temporal_adhd_traces),
        "channel_concentration": np.stack(seed_channel_conc),
        "meta": meta_ref,
        "n_seeds": n_seeds,
        "seeds": seeds,
    }


# =====================================================================
# Visualisations
# =====================================================================

def plot_temporal_profile(results, model_name, out_dir):
    """Profil temporel avec bandes de confiance ±1σ inter-seed."""
    n_seeds = results["n_seeds"]
    meta = results["meta"]
    n_layers = meta["n_layers"]
    n_patches = meta["n_patches"]

    fig, axes = plt.subplots(n_layers, 1, figsize=(11, 4 * n_layers), squeeze=False)

    for layer_idx in range(n_layers):
        ctrl_runs = results["temporal_ctrl"][:, layer_idx, :]
        adhd_runs = results["temporal_adhd"][:, layer_idx, :]

        ctrl_mean = ctrl_runs.mean(axis=0)
        ctrl_std = ctrl_runs.std(axis=0)
        adhd_mean = adhd_runs.mean(axis=0)
        adhd_std = adhd_runs.std(axis=0)

        ax = axes[layer_idx, 0]
        x = np.arange(n_patches)

        ax.plot(x, ctrl_mean, "o-", label=f"CTRL (mean over {n_seeds} seeds × {meta['n_ctrl_folds']} folds)",
                color="steelblue", linewidth=2)
        ax.fill_between(x, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                         alpha=0.25, color="steelblue", label="CTRL ±1σ inter-seed")
        for s_idx in range(n_seeds):
            ax.plot(x, ctrl_runs[s_idx], "-", color="steelblue", alpha=0.3, linewidth=0.8)

        ax.plot(x, adhd_mean, "s-", label=f"ADHD (mean over {n_seeds} seeds × {meta['n_adhd_folds']} folds)",
                color="crimson", linewidth=2)
        ax.fill_between(x, adhd_mean - adhd_std, adhd_mean + adhd_std,
                         alpha=0.25, color="crimson", label="ADHD ±1σ inter-seed")
        for s_idx in range(n_seeds):
            ax.plot(x, adhd_runs[s_idx], "-", color="crimson", alpha=0.3, linewidth=0.8)

        ax.axhline(1.0 / n_patches, color="gray", linestyle="--", alpha=0.5,
                    label=f"Uniform (1/{n_patches})")

        ctrl_inter_seed_std = ctrl_runs.std(axis=0).mean()
        adhd_inter_seed_std = adhd_runs.std(axis=0).mean()
        ax.set_title(f"{model_name} Layer {layer_idx} — LOSO temporal profile "
                      f"({n_seeds} seeds × 24 folds)\n"
                      f"Inter-seed std: CTRL={ctrl_inter_seed_std:.4f}, "
                      f"ADHD={adhd_inter_seed_std:.4f}")
        ax.set_xlabel("Patch index (temporal position, 0=start, 23=end)")
        ax.set_ylabel("Mean attention received")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"loso_temporal_profile_{model_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path.name}")


def plot_comparison_b_vs_c(results_b, results_c, out_dir):
    """Comparaison directe B vs C avec bandes de confiance."""
    n_patches = results_b["meta"]["n_patches"]
    last = results_b["meta"]["n_layers"] - 1

    b_ctrl = results_b["temporal_ctrl_traces"][:, last, :]
    b_adhd = results_b["temporal_adhd_traces"][:, last, :]
    c_ctrl = results_c["temporal_ctrl_traces"][:, last, :]
    c_adhd = results_c["temporal_adhd_traces"][:, last, :]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(n_patches)

    for ax, (b_runs, c_runs, label) in zip(
        axes, [(b_ctrl, c_ctrl, "CTRL"), (b_adhd, c_adhd, "ADHD")]
    ):
        b_mean, b_std = b_runs.mean(axis=0), b_runs.std(axis=0)
        c_mean, c_std = c_runs.mean(axis=0), c_runs.std(axis=0)
        ax.plot(x, b_mean, "o-", label="Exp B (traces only)", color="steelblue", linewidth=2)
        ax.fill_between(x, b_mean - b_std, b_mean + b_std, alpha=0.25, color="steelblue")
        ax.plot(x, c_mean, "s--", label="Exp C (traces+params, traces channels only)",
                color="darkorange", linewidth=2)
        ax.fill_between(x, c_mean - c_std, c_mean + c_std, alpha=0.25, color="darkorange")
        ax.axhline(1.0 / n_patches, color="gray", linestyle=":", alpha=0.5, label="Uniform")
        ax.set_title(f"{label} — Last layer LOSO temporal profile")
        ax.set_xlabel("Patch index")
        ax.set_ylabel("Mean attention received")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "loso_comparison_B_vs_C.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path.name}")


def plot_channel_importance(results, out_dir):
    """Importance par canal pour Exp C."""
    meta = results["meta"]
    n_layers = meta["n_layers"]
    n_channels = meta["n_channels"]
    if n_channels < 16:
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)
    channel_labels = ["x", "y"] + [f"p{i}" for i in range(14)]
    colors = ["steelblue", "steelblue"] + ["seagreen"] * 14

    for layer_idx in range(n_layers):
        conc = results["channel_concentration"][:, layer_idx, :]
        conc_mean = conc.mean(axis=0)
        conc_std = conc.std(axis=0)

        ax = axes[0, layer_idx]
        ax.bar(range(n_channels), conc_mean, yerr=conc_std, capsize=4,
                color=colors[:n_channels], alpha=0.8,
                error_kw={"elinewidth": 1.5, "ecolor": "black"})
        ax.set_xticks(range(n_channels))
        ax.set_xticklabels(channel_labels[:n_channels], rotation=45, fontsize=9)
        ax.set_title(f"Exp C Layer {layer_idx} — LOSO Attention concentration\n"
                      f"(mean ±std over {results['n_seeds']} seeds × 24 folds)")
        ax.set_ylabel("Concentration (1 - normalized entropy)")
        ax.grid(alpha=0.3, axis="y")

        top3_idx = np.argsort(-conc_mean)[:3]
        for i, idx in enumerate(top3_idx):
            ax.annotate(f"#{i+1}", xy=(idx, conc_mean[idx]),
                          xytext=(0, 5), textcoords="offset points",
                          ha="center", fontsize=10, fontweight="bold", color="red")

    plt.tight_layout()
    out_path = out_dir / "loso_channel_importance_C.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out_path.name}")


# =====================================================================
# Stats & Summary
# =====================================================================

def print_stability_summary(results, model_name):
    meta = results["meta"]
    n_layers = meta["n_layers"]

    print(f"\n=== Stability summary: {model_name} (LOSO {results['n_seeds']} seeds × 24 folds) ===")
    print(f"  Total CTRL strokes: {meta['n_total_ctrl_strokes']}")
    print(f"  Total ADHD strokes: {meta['n_total_adhd_strokes']}")
    for layer_idx in range(n_layers):
        ctrl_runs = results["temporal_ctrl"][:, layer_idx, :]
        adhd_runs = results["temporal_adhd"][:, layer_idx, :]

        ctrl_mean = ctrl_runs.mean(axis=0)
        ctrl_std_inter = ctrl_runs.std(axis=0).mean()
        adhd_mean = adhd_runs.mean(axis=0)
        adhd_std_inter = adhd_runs.std(axis=0).mean()

        print(f"  Layer {layer_idx}:")
        print(f"    CTRL : argmax_patch={int(np.argmax(ctrl_mean))}, "
              f"max={ctrl_mean.max():.4f}, "
              f"inter_seed_std={ctrl_std_inter:.4f}")
        print(f"    ADHD : argmax_patch={int(np.argmax(adhd_mean))}, "
              f"max={adhd_mean.max():.4f}, "
              f"inter_seed_std={adhd_std_inter:.4f}")

        uniform = 1.0 / meta["n_patches"]
        print(f"    Concentration ratio (max/uniform): "
              f"CTRL={ctrl_mean.max()/uniform:.2f}x, "
              f"ADHD={adhd_mean.max()/uniform:.2f}x")


def save_numerical_results(results_b, results_c, out_dir):
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
                    "protocol": "LOSO_full",
                })
    df = pd.DataFrame(rows)
    out_path = out_dir / "loso_attention_profiles.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved numerical results to {out_path.name}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/attention_analysis_loso"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    seeds = [args.seed_base + 1000 * i for i in range(args.n_seeds)]

    print(f"=== LOSO Attention analysis: Exp B vs Exp C ===")
    print(f"Device: {device}")
    print(f"N seeds: {args.n_seeds} → {seeds}")
    print(f"N epochs per training: {args.n_epochs}")
    print(f"Output dir: {args.out_dir}")

    print(f"\nLoading data...")
    traces = load_traces(Path("data/processed"))
    params_ds = load_processed_dataset(Path("data/processed"))
    labels = params_ds.labels_per_child
    n_children = len(traces.child_ids)
    total_trainings = 2 * args.n_seeds * n_children
    print(f"  {n_children} children × {args.n_seeds} seeds × 2 models = "
          f"{total_trainings} trainings")
    est_min = (args.n_seeds * n_children * 6) / 60 + (args.n_seeds * n_children * 18) / 60
    print(f"  Estimated total duration: ~{est_min:.0f} min")

    t_global = time.time()

    # Exp B
    results_b = aggregate_loso_multiseed(
        traces, params_ds, labels, traces.child_ids,
        "ExpB", device, n_epochs=args.n_epochs, seeds=seeds,
    )

    # Exp C
    results_c = aggregate_loso_multiseed(
        traces, params_ds, labels, traces.child_ids,
        "ExpC", device, n_epochs=args.n_epochs, seeds=seeds,
    )

    print(f"\n{'='*72}")
    print(f"  Generating visualizations")
    print(f"{'='*72}")

    plot_temporal_profile(results_b, "ExpB", args.out_dir)
    plot_temporal_profile(results_c, "ExpC", args.out_dir)
    plot_comparison_b_vs_c(results_b, results_c, args.out_dir)
    plot_channel_importance(results_c, args.out_dir)
    save_numerical_results(results_b, results_c, args.out_dir)

    print_stability_summary(results_b, "ExpB")
    print_stability_summary(results_c, "ExpC")

    total_time = time.time() - t_global
    print(f"\n✓ Total execution time: {total_time/60:.1f} min")
    print(f"✓ All results saved in {args.out_dir}/")


if __name__ == "__main__":
    main()
