"""
scripts/02_qc_visualize.py

Contrôle qualité visuel après preprocessing.

Figures générées (results/figures/qc/) :
    qc_sample_traces.png          : 12 traces nettoyées aléatoires (x,y + v(t))
    qc_length_distributions.png   : distributions longueur/durée par groupe
    qc_parameter_boxplots.png     : boxplots des 14 paramètres par groupe
    qc_strokes_per_child.png      : nombre de traits valides par enfant
    qc_nblog_focus.png            : focus sur nbLog (feature discriminante)

Usage :
    python scripts/02_qc_visualize.py \
        --raw-dir data/raw \
        --out-dir results/figures/qc \
        --duration-max 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_all_children
from src.data.raw_traces import compute_derived_channels
from src.data.sigma_lognormal import PARAMETER_NAMES


def plot_sample_traces(children, out_dir: Path, n_samples: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    ctrl = [(c, s) for c in children if c.label == 0 for s in c.strokes]
    adhd = [(c, s) for c in children if c.label == 1 for s in c.strokes]
    n_each = n_samples // 2
    if len(ctrl) < n_each or len(adhd) < n_each:
        print("  skipping sample traces (not enough strokes)")
        return
    sel_ctrl = rng.choice(len(ctrl), n_each, replace=False)
    sel_adhd = rng.choice(len(adhd), n_each, replace=False)
    picks = [ctrl[i] for i in sel_ctrl] + [adhd[i] for i in sel_adhd]

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2 * n_samples))
    for i, (child, stroke) in enumerate(picks):
        tag = "CTRL" if child.label == 0 else "ADHD"
        ax_xy, ax_v = axes[i, 0], axes[i, 1]
        ax_xy.plot(stroke.raw_trace.x, stroke.raw_trace.y, 'b-', lw=1)
        ax_xy.set_title(
            f"{child.child_id} t{stroke.trial_id} [{tag}] "
            f"N={stroke.raw_trace.n_samples} dur={stroke.raw_trace.duration_s:.2f}s",
            fontsize=8,
        )
        ax_xy.set_xlabel("x (mm)", fontsize=7)
        ax_xy.set_ylabel("y (mm)", fontsize=7)
        ax_xy.tick_params(labelsize=6)
        ax_xy.set_aspect('equal', adjustable='datalim')

        ch = compute_derived_channels(stroke.raw_trace)
        ax_v.plot(stroke.raw_trace.t, ch['v'], 'r-', lw=1)
        ax_v.set_title(
            f"SNR={stroke.sigma_params.SNR:.1f}dB nbLog={stroke.sigma_params.nbLog}",
            fontsize=8,
        )
        ax_v.set_xlabel("t (s)", fontsize=7)
        ax_v.set_ylabel("v (mm/s)", fontsize=7)
        ax_v.tick_params(labelsize=6)

    plt.tight_layout()
    out_path = out_dir / "qc_sample_traces.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out_path}")


def plot_length_distributions(children, out_dir: Path):
    rows = []
    for c in children:
        for s in c.strokes:
            rows.append({
                "group": "CTRL" if c.label == 0 else "ADHD",
                "n_samples": s.raw_trace.n_samples,
                "duration_s": s.raw_trace.duration_s,
            })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, sub in df.groupby("group"):
        axes[0].hist(sub["n_samples"], bins=30, alpha=0.5, label=label)
        axes[1].hist(sub["duration_s"], bins=30, alpha=0.5, label=label)
    axes[0].set_xlabel("n_samples (cleaned)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Trace length distribution")
    axes[0].legend()
    axes[1].set_xlabel("duration (s)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Trace duration distribution")
    axes[1].legend()
    plt.tight_layout()
    out_path = out_dir / "qc_length_distributions.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out_path}")


def plot_parameter_boxplots(children, out_dir: Path):
    rows = []
    for c in children:
        for s in c.strokes:
            r = {"group": "CTRL" if c.label == 0 else "ADHD"}
            for name in PARAMETER_NAMES:
                r[name] = getattr(s.sigma_params, name)
            rows.append(r)
    df = pd.DataFrame(rows)

    n = len(PARAMETER_NAMES)   # 14
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()
    for i, name in enumerate(PARAMETER_NAMES):
        data_ctrl = df[df["group"] == "CTRL"][name].dropna()
        data_adhd = df[df["group"] == "ADHD"][name].dropna()
        axes[i].boxplot([data_ctrl, data_adhd], labels=["CTRL", "ADHD"],
                        showfliers=False)
        axes[i].set_title(name, fontsize=10)
        axes[i].tick_params(labelsize=8)
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    out_path = out_dir / "qc_parameter_boxplots.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out_path}")


def plot_nblog_focus(children, out_dir: Path):
    """Focus sur nbLog — devrait être discriminant avec SSVn."""
    data_ctrl, data_adhd = [], []
    for c in children:
        for s in c.strokes:
            if c.label == 0:
                data_ctrl.append(s.sigma_params.nbLog)
            else:
                data_adhd.append(s.sigma_params.nbLog)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    max_val = max(max(data_ctrl, default=1), max(data_adhd, default=1))
    bins = np.arange(0.5, max_val + 1.5, 1)
    axes[0].hist([data_ctrl, data_adhd], bins=bins, alpha=0.7,
                 label=["CTRL", "ADHD"], density=True)
    axes[0].set_xlabel("nbLog (SSVn)")
    axes[0].set_ylabel("density")
    axes[0].set_title("nbLog distribution (normalized)")
    axes[0].legend()
    axes[0].set_xticks(range(1, int(max_val) + 1))

    axes[1].boxplot([data_ctrl, data_adhd], labels=["CTRL", "ADHD"])
    axes[1].set_ylabel("nbLog (SSVn)")
    axes[1].set_title(
        f"nbLog boxplot\n"
        f"CTRL mean={np.mean(data_ctrl):.2f}  ADHD mean={np.mean(data_adhd):.2f}"
    )
    plt.tight_layout()
    out_path = out_dir / "qc_nblog_focus.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out_path}")


def plot_strokes_per_child(children, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ids = [c.child_id for c in children]
    counts = [c.n_strokes for c in children]
    colors = ['tab:blue' if c.label == 0 else 'tab:red' for c in children]
    ax.bar(ids, counts, color=colors)
    ax.set_ylabel("valid strokes")
    ax.set_title("Valid strokes per child after filtering (blue=CTRL, red=ADHD)")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    # Lignes repères utiles pour Exp A
    ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    out_path = out_dir / "qc_strokes_per_child.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="QC visualization")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/qc"))
    parser.add_argument("--snr-min", type=float, default=15.0)
    parser.add_argument("--d-max", type=float, default=500.0)
    parser.add_argument("--duration-max", type=float, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading children from {args.raw_dir}...")
    children, _ = load_all_children(
        args.raw_dir,
        snr_min=args.snr_min,
        d_max_mm=args.d_max,
        duration_max_s=args.duration_max,
    )
    print(f"Loaded {len(children)} children with "
          f"{sum(c.n_strokes for c in children)} valid strokes")

    print("Generating QC figures...")
    plot_sample_traces(children, args.out_dir)
    plot_length_distributions(children, args.out_dir)
    plot_parameter_boxplots(children, args.out_dir)
    plot_nblog_focus(children, args.out_dir)
    plot_strokes_per_child(children, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
