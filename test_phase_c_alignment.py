"""
test_phase_c_alignment.py

Test rapide (5 secondes) : vérifie que le mapping (cid, tid) -> params
fonctionne correctement avant de lancer le LOSO complet.

ATTENTION : si ce test échoue, le code Exp C ne pourra pas marcher.
Il faut alors regarder la structure exacte de ProcessedDataset et
adapter src/data/exp_c_datasets.py:_get_params_for_stroke().
"""

from __future__ import annotations

import os
os.environ.setdefault("KNM_DEVICE", "cpu")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.exp_c_datasets import RawTraceWithParamsDataset
from src.utils.data_io import load_processed_dataset
from src.utils.traces_io import load_traces


def main():
    print("=== Test alignment Exp C ===")

    # 1. Charge les deux datasets
    print("\n--- 1. Loading data ---")
    traces = load_traces(Path("data/processed"))
    params_ds = load_processed_dataset(Path("data/processed"))
    labels = params_ds.labels_per_child

    print(f"Traces: {len(traces.child_ids)} children, {traces.total_strokes} strokes")
    print(f"ProcessedDataset attributes:")
    attrs = [a for a in dir(params_ds) if not a.startswith('_')]
    print(f"  {attrs}")

    # 2. Inspecte la structure
    print("\n--- 2. Inspecting ProcessedDataset structure ---")
    cid = traces.child_ids[0]  # S01
    print(f"Sample child: {cid}")

    if hasattr(params_ds, 'params_per_child'):
        params_arr = params_ds.params_per_child[cid]
        print(f"  params_per_child[{cid}].shape = {params_arr.shape}")
    else:
        print(f"  ⚠ ProcessedDataset has NO params_per_child attribute")

    if hasattr(params_ds, 'stroke_indices_per_child'):
        tids = params_ds.stroke_indices_per_child[cid]
        print(f"  stroke_indices_per_child[{cid}] = {list(tids)[:10]}... (showing first 10)")
        print(f"  total = {len(tids)}")
    else:
        print(f"  ⚠ ProcessedDataset has NO stroke_indices_per_child attribute")

    # Comparaison avec traces
    tids_in_traces = traces.trial_ids_per_child[cid]
    print(f"  traces.trial_ids_per_child[{cid}] = {tids_in_traces[:10]}... (first 10)")
    print(f"  total in traces = {len(tids_in_traces)}")

    # 3. Test création du dataset
    print("\n--- 3. Trying to create RawTraceWithParamsDataset ---")
    try:
        ds = RawTraceWithParamsDataset(
            traces=traces, params_dataset=params_ds, labels_per_child=labels,
            child_ids=traces.child_ids, target_len=200,
        )
        print(f"  ✓ Dataset created: {len(ds)} samples")
        print(f"    n_filtered_motion = {ds.n_filtered_motion}")
        print(f"    n_missing_params = {ds.n_missing_params}")
        print(f"    n_total_channels = {ds.n_total_channels}")

        if len(ds) == 0:
            print(f"  ⚠ Dataset is EMPTY — alignment failed completely")
            return

        # Test 1 sample
        x_sample, y_sample = ds[0]
        print(f"  Sample 0: x.shape={tuple(x_sample.shape)}, y={y_sample.item()}")
        assert x_sample.shape == (16, 200), \
            f"Expected (16, 200), got {tuple(x_sample.shape)}"
        print(f"  ✓ Shape (16, 200) as expected")

        # Vérif que les canaux 2-15 (params) sont bien constants
        print(f"\n  Per-channel std on sample 0 (canaux 2-15 doivent être à 0):")
        for c in range(16):
            std = float(x_sample[c].std())
            label = "trace" if c < 2 else f"param_{c-2}"
            flag = "" if c < 2 else (" ✓" if std < 1e-6 else " ⚠ NOT CONSTANT")
            print(f"    ch {c:2d} ({label:>10s}): mean={float(x_sample[c].mean()):+.4f}, "
                  f"std={std:.4f}{flag}")

        # Sanity check : les valeurs des canaux 2-15 doivent égaler les
        # paramètres sigma-lognormaux du trait
        cid_sample, tid_sample = ds._items[0]
        params_expected = ds._get_params_for_stroke(cid_sample, tid_sample)
        print(f"\n  Trait ({cid_sample}, t{tid_sample}) — vérif des valeurs:")
        for c in range(2, 16):
            actual = float(x_sample[c, 0])  # toutes les positions ont la même valeur
            expected = float(params_expected[c - 2])
            ok = "✓" if abs(actual - expected) < 1e-5 else "✗"
            print(f"    ch {c:2d}: actual={actual:+.4f}, expected={expected:+.4f} {ok}")

        print(f"\n  ✓ Alignment WORKS — Exp C can be launched")

    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
