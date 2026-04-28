"""
check_temporal_order.py — Vérifie que les trials dans chaque JSON sont
ordonnés temporellement, condition sine qua non pour Exp A.

Usage: python check_temporal_order.py /chemin/vers/raw
"""
import json
import sys
from pathlib import Path


def check_child(jf):
    with open(jf) as f:
        trials = json.load(f)

    n = len(trials)
    trial_ids = []
    timestamps_start = []
    timestamps_go = []
    recorded_at = []

    for t in trials:
        tmeta = t.get("metaData", {}).get("trial", {})
        trial_ids.append(tmeta.get("trialID"))
        timestamps_start.append(tmeta.get("timeStampStart"))
        timestamps_go.append(tmeta.get("timeStampGo"))
        recorded_at.append(tmeta.get("recordedAt"))

    # Analyses
    report = {
        "child": jf.stem,
        "n_trials": n,
        "trialIDs": trial_ids,
        "trialID_is_monotonic": None,
        "trialID_unique": len(set(trial_ids)) == n,
        "timeStampStart_range": (min(t for t in timestamps_start if t is not None),
                                 max(t for t in timestamps_start if t is not None))
                                if any(t is not None for t in timestamps_start) else None,
        "timeStampStart_all_zero": all(t == 0 for t in timestamps_start if t is not None),
        "recordedAt_sample": recorded_at[:3],
    }

    # Monotonie des trialID
    clean_ids = [tid for tid in trial_ids if tid is not None]
    if len(clean_ids) >= 2:
        monotonic_inc = all(clean_ids[i] <= clean_ids[i + 1] for i in range(len(clean_ids) - 1))
        report["trialID_is_monotonic"] = monotonic_inc

    return report


def main(folder):
    folder = Path(folder)
    json_files = sorted(folder.glob("S*.json"))

    print(f"\n{'=' * 72}")
    print(f"TEMPORAL ORDER CHECK across {len(json_files)} files")
    print(f"{'=' * 72}\n")

    all_monotonic = True
    all_unique = True
    trialID_issue_children = []

    for jf in json_files:
        r = check_child(jf)
        mono = r["trialID_is_monotonic"]
        uniq = r["trialID_unique"]
        ts_all_zero = r["timeStampStart_all_zero"]

        status = "OK" if (mono and uniq) else "PROBLEM"
        print(f"{r['child']:6s} [{status}] n={r['n_trials']:3d}  "
              f"trialID_monotonic={mono}  trialID_unique={uniq}  "
              f"timeStampStart_all_zero={ts_all_zero}")

        print(f"       trialIDs: {r['trialIDs']}")
        if r['timeStampStart_range']:
            print(f"       timeStampStart range: {r['timeStampStart_range']}")
        print(f"       recordedAt sample: {r['recordedAt_sample']}")
        print()

        if not mono:
            all_monotonic = False
            trialID_issue_children.append(r['child'])
        if not uniq:
            all_unique = False

    print(f"{'=' * 72}")
    print(f"SUMMARY")
    print(f"{'=' * 72}")
    print(f"All children have monotonic trialIDs: {all_monotonic}")
    print(f"All children have unique trialIDs:    {all_unique}")
    if trialID_issue_children:
        print(f"Children with non-monotonic trialIDs: {trialID_issue_children}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_temporal_order.py /path/to/json/folder")
        sys.exit(1)
    main(sys.argv[1])