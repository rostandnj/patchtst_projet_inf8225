"""
scripts/01_preprocess.py

Phase 1 : charge les 24 JSON, applique le pipeline nettoyage/filtrage/
extraction, exporte le dataset prêt pour l'entraînement.

Extraction : SSVn pur (équivalent à l'extracteur de Faci et al. 2021).
    - Paramètres physiologiques : SSVn.parameters.correction
    - SNR et nbLog : SSVn.snr et SSVn.nbLogs

Exports (data/processed/) :
    params.npz             : 14 paramètres par trait, groupés par enfant
    traces.npz             : traces brutes nettoyées (longueur variable)
    metadata.parquet/.csv  : table de métadonnées (un row par trait)
    preprocessing_report.txt : rapport de filtrage

Usage :
    python scripts/01_preprocess.py \
        --raw-dir data/raw \
        --out-dir data/processed \
        --snr-min 15.0 \
        --d-max 500.0 \
        --duration-max 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_all_children
from src.data.sigma_lognormal import PARAMETER_NAMES


def main():
    parser = argparse.ArgumentParser(description="Preprocess TDAH dataset (SSVn extraction)")
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Dossier contenant les SXX.json")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--snr-min", type=float, default=15.0,
                        help="Seuil SNR (SSVn) pour rejet logiciel en dB (défaut: 15.0)")
    parser.add_argument("--d-max", type=float, default=500.0,
                        help="Amplitude D maximale plausible en mm (défaut: 500)")
    parser.add_argument("--duration-max", type=float, default=None,
                        help="Durée max du trait nettoyé en s (optionnel, ex: 3.0)")
    parser.add_argument("--velocity-threshold", type=float, default=5.0,
                        help="Seuil de vitesse pour nettoyage plateaux en mm/s (défaut: 5)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chargement + filtrage
    print(f"Loading children from {args.raw_dir}...")
    filter_desc = f"SNR >= {args.snr_min}dB, D <= {args.d_max}mm"
    if args.duration_max is not None:
        filter_desc += f", duration <= {args.duration_max}s"
    print(f"Filters: {filter_desc}")

    children, log = load_all_children(
        args.raw_dir,
        snr_min=args.snr_min,
        d_max_mm=args.d_max,
        duration_max_s=args.duration_max,
        velocity_threshold_mm_s=args.velocity_threshold,
    )

    # 2. Rapport
    report_lines = [
        "=" * 72,
        "PREPROCESSING REPORT",
        "=" * 72,
        "",
        "Extraction: SSVn (free-number-of-lognormals, equiv. to Faci 2021)",
        "  - Physiological params from SSVn.parameters.correction",
        "  - SNR, nbLog from SSVn global fields",
        "",
        f"Filters applied:",
        f"  R1: SNR >= {args.snr_min} dB",
        f"  R2: NOT (t0 < 0 AND mu > 0)",
        f"  R3: D <= {args.d_max} mm",
        f"  R4: all params finite",
    ]
    if args.duration_max is not None:
        report_lines.append(f"  R5: cleaned duration <= {args.duration_max} s")
    report_lines.extend([
        "",
        f"Children loaded: {len(children)}",
        f"  - Controls (label=0): {sum(1 for c in children if c.label == 0)}",
        f"  - ADHD (label=1):     {sum(1 for c in children if c.label == 1)}",
        "",
        "Rejection log:",
        log.summary(),
        "",
        "Strokes kept per child:",
    ])
    for c in children:
        tag = "CTRL" if c.label == 0 else "ADHD"
        report_lines.append(f"  {c.child_id} [{tag}]: {c.n_strokes} strokes")
    total_kept = sum(c.n_strokes for c in children)
    report_lines.append("")
    report_lines.append(f"GRAND TOTAL strokes kept: {total_kept}")

    lengths = [s.raw_trace.n_samples for c in children for s in c.strokes]
    if lengths:
        lengths = np.array(lengths)
        report_lines.extend([
            "",
            "Cleaned trace length distribution (samples):",
            f"  min:    {lengths.min()}",
            f"  max:    {lengths.max()}",
            f"  mean:   {lengths.mean():.1f}",
            f"  median: {np.median(lengths):.1f}",
            f"  p25:    {np.percentile(lengths, 25):.1f}",
            f"  p75:    {np.percentile(lengths, 75):.1f}",
        ])
    durations = [s.raw_trace.duration_s for c in children for s in c.strokes]
    if durations:
        durations = np.array(durations)
        report_lines.extend([
            "",
            "Cleaned trace duration distribution (seconds):",
            f"  min:    {durations.min():.3f}",
            f"  max:    {durations.max():.3f}",
            f"  mean:   {durations.mean():.3f}",
            f"  median: {np.median(durations):.3f}",
        ])

    # Distribution de nbLog (feature discriminante attendue avec SSVn)
    nblog_values = [(c.label, s.sigma_params.nbLog)
                    for c in children for s in c.strokes]
    if nblog_values:
        nb_ctrl = [n for lbl, n in nblog_values if lbl == 0]
        nb_adhd = [n for lbl, n in nblog_values if lbl == 1]
        report_lines.extend([
            "",
            "nbLog distribution (SSVn free extractor):",
            f"  CTRL: mean={np.mean(nb_ctrl):.2f}, median={np.median(nb_ctrl):.1f}, "
            f"min={min(nb_ctrl)}, max={max(nb_ctrl)}",
            f"  ADHD: mean={np.mean(nb_adhd):.2f}, median={np.median(nb_adhd):.1f}, "
            f"min={min(nb_adhd)}, max={max(nb_adhd)}",
        ])

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (args.out_dir / "preprocessing_report.txt").write_text(report_text)

    # 3. Export paramètres
    print(f"\nExporting to {args.out_dir}...")
    params_dict = {}
    trial_ids_dict = {}
    for c in children:
        params_dict[f"{c.child_id}_params"] = c.params_matrix().astype(np.float32)
        trial_ids_dict[f"{c.child_id}_trial_ids"] = c.trial_ids()

    np.savez(
        args.out_dir / "params.npz",
        **params_dict,
        **trial_ids_dict,
        child_ids=np.array([c.child_id for c in children]),
        labels=np.array([c.label for c in children], dtype=np.int64),
        param_names=np.array(PARAMETER_NAMES),
    )
    total_params_kb = sum(v.nbytes for v in params_dict.values()) / 1024
    print(f"  - params.npz written ({total_params_kb:.1f} KB of params)")

    # 4. Export traces brutes
    traces_dict = {}
    for c in children:
        for s in c.strokes:
            key = f"{c.child_id}_t{s.trial_id:03d}"
            traces_dict[f"{key}_x"] = s.raw_trace.x.astype(np.float32)
            traces_dict[f"{key}_y"] = s.raw_trace.y.astype(np.float32)
            traces_dict[f"{key}_t"] = s.raw_trace.t.astype(np.float32)
    np.savez(args.out_dir / "traces.npz", **traces_dict)
    total_traces_mb = sum(v.nbytes for v in traces_dict.values()) / 1024 / 1024
    print(f"  - traces.npz written ({total_traces_mb:.2f} MB)")

    # 5. Métadonnée tabulaire
    rows = []
    for c in children:
        for pos, s in enumerate(c.strokes):
            row = {
                "child_id": c.child_id,
                "label": c.label,
                "trial_id": s.trial_id,
                "stroke_index_in_child": pos,
                "n_samples": s.raw_trace.n_samples,
                "duration_s": s.raw_trace.duration_s,
            }
            for name in PARAMETER_NAMES:
                row[name] = getattr(s.sigma_params, name)
            rows.append(row)
    metadata = pd.DataFrame(rows)
    metadata.to_parquet(args.out_dir / "metadata.parquet", index=False)
    metadata.to_csv(args.out_dir / "metadata.csv", index=False)
    print(f"  - metadata.parquet + metadata.csv written ({len(metadata)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
