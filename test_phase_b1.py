"""
test_phase_b1.py

Test d'intégration Phase B.1 : chargement des traces brutes + dataset.

Vérifications :
    1. Chargement de traces.npz : 663 traits, 24 enfants
    2. Stats sur les longueurs de traits
    3. RawTraceStrokeDataset : un trait par échantillon, shape (2, 200)
    4. Test avec channels=('x','y','v') pour vérifier le calcul de vélocité
    5. Vérification que pas de NaN/Inf
    6. Test du PatchTST custom sur ces données (forward pass uniquement)
    7. Aggregation au niveau enfant
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_b_datasets import (
    RawTraceStrokeDataset,
    aggregate_stroke_predictions,
)
from src.models.patchtst import PatchTSTConfig, PatchTSTClassifier
from src.training.device import get_device
from src.utils.data_io import load_processed_dataset
from src.utils.seeds import set_global_seed
from src.utils.traces_io import load_traces


def section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def main():
    set_global_seed(42)
    device = get_device()
    print(f"=== Phase B.1 integration test ===")
    print(f"Device: {device}")

    # =====================================================================
    # 1. Chargement des traces brutes
    # =====================================================================
    section("1. Load traces.npz")
    traces = load_traces(Path("data/processed"))
    print(f"  Loaded {len(traces.child_ids)} children, "
          f"{traces.total_strokes} valid strokes")
    print(f"  Children: {traces.child_ids[:5]}... (showing first 5)")

    stats = traces.trace_lengths_stats()
    print(f"  Trace lengths (samples per stroke):")
    print(f"    min={stats['min']}, max={stats['max']}")
    print(f"    median={stats['median']:.1f}, p25={stats['p25']:.1f}, "
          f"p75={stats['p75']:.1f}, p95={stats['p95']:.1f}")
    print(f"    mean={stats['mean']:.1f}")

    assert traces.total_strokes >= 600, \
        f"Expected ~663 strokes, got {traces.total_strokes}"
    print(f"  ✓ Total strokes >= 600 as expected")

    n_strokes_per_child = {cid: traces.n_strokes(cid) for cid in traces.child_ids}
    print(f"  Strokes per child: min={min(n_strokes_per_child.values())}, "
          f"max={max(n_strokes_per_child.values())}, "
          f"mean={np.mean(list(n_strokes_per_child.values())):.1f}")

    # =====================================================================
    # 2. Charge les labels depuis params.npz
    # =====================================================================
    section("2. Load labels from params.npz")
    ds_params = load_processed_dataset(Path("data/processed"))
    labels = ds_params.labels_per_child
    n_ctrl = sum(1 for v in labels.values() if v == 0)
    n_adhd = sum(1 for v in labels.values() if v == 1)
    print(f"  Labels: CTRL={n_ctrl}, ADHD={n_adhd}")
    assert n_ctrl == 12 and n_adhd == 12, \
        f"Expected 12+12, got {n_ctrl}+{n_adhd}"
    print(f"  ✓ Balanced 12+12 as expected")

    # =====================================================================
    # 3. RawTraceStrokeDataset basique (x, y), T=200
    # =====================================================================
    section("3. RawTraceStrokeDataset basic (x, y), T=200")
    ds_xy = RawTraceStrokeDataset(
        traces=traces,
        labels_per_child=labels,
        child_ids=traces.child_ids,
        target_len=200,
        channels=("x", "y"),
        center_xy_flag=True,
    )
    print(f"  Total samples: {len(ds_xy)}")
    print(f"  n_channels: {ds_xy.n_channels}")

    x_sample, y_sample = ds_xy[0]
    print(f"  Sample 0: x.shape={tuple(x_sample.shape)}, y={y_sample.item()}")
    assert x_sample.shape == (2, 200), \
        f"Expected (2, 200), got {tuple(x_sample.shape)}"
    print(f"  ✓ Sample shape is (2, 200) as expected")

    # Stats sur 100 samples random pour vérifier la santé des données
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(ds_xy), size=min(100, len(ds_xy)), replace=False)
    all_x = []
    n_padded_samples = 0
    n_truncated_samples = 0
    actual_lengths = []
    for idx in sample_idx:
        item = ds_xy._build_item(*ds_xy._items[idx])
        all_x.append(item.x.numpy())
        actual_lengths.append(item.n_actual)
        if item.n_actual < 200:
            n_padded_samples += 1
        # On considère qu'un sample est tronqué si son n_actual originel > 200
        # mais comme n_actual est cap à target_len, on déduit autrement :
        # on regarde si la trace originelle dépassait 200
        cid, tid = ds_xy._items[idx]
        original_len = len(traces.traces[(cid, tid)]["x"])
        if original_len > 200:
            n_truncated_samples += 1
    all_x = np.stack(all_x, axis=0)
    print(f"  Stats on 100 random samples:")
    print(f"    x_channel: mean={all_x[:, 0, :].mean():.4f}, "
          f"std={all_x[:, 0, :].std():.4f}, "
          f"min={all_x[:, 0, :].min():.2f}, max={all_x[:, 0, :].max():.2f}")
    print(f"    y_channel: mean={all_x[:, 1, :].mean():.4f}, "
          f"std={all_x[:, 1, :].std():.4f}, "
          f"min={all_x[:, 1, :].min():.2f}, max={all_x[:, 1, :].max():.2f}")
    print(f"    n_padded (< 200 samples): {n_padded_samples}/100")
    print(f"    n_truncated (originally > 200): {n_truncated_samples}/100")
    print(f"    n_actual: mean={np.mean(actual_lengths):.1f}, "
          f"min={min(actual_lengths)}, max={max(actual_lengths)}")

    # Vérif pas de NaN/Inf
    has_nan = bool(np.isnan(all_x).any())
    has_inf = bool(np.isinf(all_x).any())
    print(f"    NaN present: {has_nan}, Inf present: {has_inf}")
    assert not has_nan, "Found NaN in samples"
    assert not has_inf, "Found Inf in samples"
    print(f"  ✓ No NaN or Inf detected")

    # =====================================================================
    # 4. RawTraceStrokeDataset avec vélocité (3 canaux)
    # =====================================================================
    section("4. RawTraceStrokeDataset with velocity (x, y, v), T=200")
    ds_xyv = RawTraceStrokeDataset(
        traces=traces,
        labels_per_child=labels,
        child_ids=traces.child_ids,
        target_len=200,
        channels=("x", "y", "v"),
        center_xy_flag=True,
    )
    x_v_sample, _ = ds_xyv[0]
    print(f"  Sample 0 with velocity: x.shape={tuple(x_v_sample.shape)}")
    assert x_v_sample.shape == (3, 200)
    v_channel = x_v_sample[2].numpy()
    print(f"    v channel: mean={v_channel.mean():.4f}, "
          f"std={v_channel.std():.4f}, "
          f"min={v_channel.min():.2f}, max={v_channel.max():.2f}")
    print(f"    (units: mm/s, expected to be in 0-2000 range typically)")
    assert v_channel.min() >= 0, "Velocity magnitude should be non-negative"
    print(f"  ✓ Velocity computed correctly (non-negative)")

    # =====================================================================
    # 5. DataLoader avec batches
    # =====================================================================
    section("5. DataLoader batches")
    loader = DataLoader(ds_xy, batch_size=16, shuffle=True)
    print(f"  Num batches: {len(loader)}")
    x_batch, y_batch = next(iter(loader))
    print(f"  Batch shape: x={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
    print(f"  Batch labels distribution: CTRL={int((y_batch == 0).sum())}, "
          f"ADHD={int((y_batch == 1).sum())}")
    assert x_batch.shape == (16, 2, 200)
    assert y_batch.shape == (16,)
    print(f"  ✓ Batches are correctly shaped")

    # =====================================================================
    # 6. Forward PatchTST custom sur ces données
    # =====================================================================
    section("6. Forward PatchTST custom on raw traces")
    cfg = PatchTSTConfig(
        n_channels=2,             # x, y
        seq_len=200,              # T=200
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.2,
        attn_dropout=0.1,
    )
    model = PatchTSTClassifier(cfg, n_classes=2, init_head_zero=True).to(device)
    print(f"  Model: {model.n_parameters():,} parameters "
          f"(n_channels={cfg.n_channels}, seq_len={cfg.seq_len})")

    # Forward
    model.eval()
    with torch.no_grad():
        x_dev = x_batch.to(device)
        logits = model(x_dev)
    print(f"  Forward output: shape={tuple(logits.shape)}, "
          f"dtype={logits.dtype}")
    assert logits.shape == (16, 2)

    # Vérif probas neutres à l'init
    probs = torch.softmax(logits, dim=-1)
    p_adhd_mean = probs[:, 1].mean().item()
    p_adhd_std = probs[:, 1].std().item()
    print(f"  P(ADHD) at init: mean={p_adhd_mean:.4f}, std={p_adhd_std:.4f}")
    assert 0.45 < p_adhd_mean < 0.55, \
        f"P(ADHD) at init should be ~0.5 with init_head_zero, got {p_adhd_mean}"
    print(f"  ✓ P(ADHD) ~ 0.5 at init (init_head_zero working correctly)")

    # =====================================================================
    # 7. Aggregation niveau enfant
    # =====================================================================
    section("7. Aggregate stroke predictions to child level")
    # Construit un dataset d'évaluation pour 5 enfants
    test_cids = traces.child_ids[:5]
    ds_eval = RawTraceStrokeDataset(
        traces=traces,
        labels_per_child=labels,
        child_ids=test_cids,
        target_len=200,
        channels=("x", "y"),
    )
    loader_eval = DataLoader(ds_eval, batch_size=32, shuffle=False)
    print(f"  Evaluation dataset: {len(ds_eval)} strokes from {len(test_cids)} children")

    # Inférence trait par trait
    all_probs = []
    model.eval()
    with torch.no_grad():
        for x_b, _ in loader_eval:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    cids_per_stroke = ds_eval.child_id_per_item()
    unique_cids, proba_per_child = aggregate_stroke_predictions(
        cids_per_stroke, all_probs, aggregation="mean_proba",
    )
    print(f"  Aggregated to {len(unique_cids)} children:")
    for cid, p in zip(unique_cids, proba_per_child):
        true_label = "ADHD" if labels[cid] == 1 else "CTRL"
        print(f"    {cid} [{true_label}]: P(ADHD)_mean={p:.4f}")

    # =====================================================================
    # FINAL
    # =====================================================================
    section("✓ All Phase B.1 tests passed!")
    print(f"")
    print(f"Phase B.1 ready. Next step: train PatchTST on raw traces (Phase B.2).")
    print(f"")
    print(f"Summary of decisions:")
    print(f"  - Granularity: one stroke = one sample")
    print(f"  - Total samples: {len(ds_xy)}")
    print(f"  - n_channels: 2 (x, y) by default; 'v' available")
    print(f"  - target_len T: 200 (median trace ~{stats['median']:.0f})")
    print(f"  - Pre-processing: mean centering of (x, y) per stroke")
    print(f"  - Padding: zero-pad right; Truncation: centered")


if __name__ == "__main__":
    main()
