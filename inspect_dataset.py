"""
inspect_dataset.py — Inspection rapide du dataset TDAH de Faci et al. 2021.
Chaque JSON est un array de trials, un fichier par enfant.

Usage: python inspect_dataset.py /chemin/vers/dossier_json
"""
import json
import sys
from pathlib import Path
from collections import Counter
import statistics


def main(folder):
    folder = Path(folder)
    json_files = sorted(folder.glob("S*.json"))
    print(f"\n{'=' * 72}")
    print(f"Found {len(json_files)} JSON files in {folder}")
    print(f"{'=' * 72}\n")

    if not json_files:
        print("No files found. Check the path.")
        return

    # ---- 1. Structure du premier fichier ----
    print(f"--- FIRST FILE INSPECTION: {json_files[0].name} ---\n")
    with open(json_files[0]) as f:
        first = json.load(f)
    print(f"Top-level type: {type(first).__name__}")
    if isinstance(first, list):
        print(f"Number of trials in this child: {len(first)}")
        print(f"Keys of first trial: {list(first[0].keys())}")
        meta_keys = list(first[0].get("metaData", {}).keys())
        print(f"metaData keys: {meta_keys}")
        task_ids_first = [t.get("metaData", {}).get("task", {}).get("taskID") for t in first]
        print(f"Task IDs in this child: {Counter(task_ids_first)}")
    else:
        print("WARNING: top-level is not a list. Check structure.")
        return

    # ---- 2. Statistiques globales ----
    print(f"\n{'=' * 72}")
    print("GLOBAL STATISTICS")
    print(f"{'=' * 72}\n")

    health_per_child = {}
    age_per_child = {}
    gender_per_child = {}
    hand_per_child = {}
    trials_total_per_child = {}
    trials_valid_per_child = {}
    trials_by_task_per_child = {}
    n_samples_distribution = []
    has_pressure_count = 0
    has_elevation_count = 0
    has_azimuth_count = 0
    has_ssvp_count = 0
    has_ssvn_count = 0
    has_iix_count = 0
    errors = []

    for jf in json_files:
        try:
            with open(jf) as f:
                trials = json.load(f)
            if not isinstance(trials, list):
                errors.append(f"{jf.name}: not a list")
                continue

            trials_total_per_child[jf.stem] = len(trials)

            # Méta enfant: on prend du premier trial
            first_trial = trials[0]
            meta = first_trial.get("metaData", {})
            part = meta.get("participant", {})
            health_per_child[jf.stem] = part.get("health", {}).get("healthy")
            age_per_child[jf.stem] = part.get("age")
            gender_per_child[jf.stem] = part.get("gender")
            hand_per_child[jf.stem] = part.get("dominantHand")

            # Stats par trial
            n_valid = 0
            tasks_here = Counter()
            for t in trials:
                tmeta = t.get("metaData", {})
                if tmeta.get("trial", {}).get("validTrial") is True:
                    n_valid += 1
                task_id = tmeta.get("task", {}).get("taskID")
                tasks_here[task_id] += 1

                raw = t.get("rawData", {})
                # longueur x
                x_arr = raw.get("x", [[]])
                if x_arr and isinstance(x_arr, list) and len(x_arr) > 0 and isinstance(x_arr[0], list):
                    n_samples_distribution.append(len(x_arr[0]))

                # canaux disponibles
                if raw.get("pressure") and raw["pressure"] and len(raw["pressure"]) > 0 \
                        and isinstance(raw["pressure"][0], list) and len(raw["pressure"][0]) > 0:
                    has_pressure_count += 1
                if raw.get("elevation") and raw["elevation"] and len(raw["elevation"]) > 0 \
                        and isinstance(raw["elevation"][0], list) and len(raw["elevation"][0]) > 0:
                    has_elevation_count += 1
                if raw.get("azimuth") and raw["azimuth"] and len(raw["azimuth"]) > 0 \
                        and isinstance(raw["azimuth"][0], list) and len(raw["azimuth"][0]) > 0:
                    has_azimuth_count += 1

                # extracteurs
                extracted = t.get("extracted", {})
                if "SSVp" in extracted and extracted["SSVp"].get("extractedData"):
                    has_ssvp_count += 1
                if "SSVn" in extracted and extracted["SSVn"].get("extractedData"):
                    has_ssvn_count += 1
                if "IIX" in extracted and extracted["IIX"].get("extractedData"):
                    has_iix_count += 1

            trials_valid_per_child[jf.stem] = n_valid
            trials_by_task_per_child[jf.stem] = dict(tasks_here)

        except Exception as e:
            errors.append(f"{jf.name}: {type(e).__name__}: {e}")

    # ---- 3. Rapport ----
    print("--- Health distribution ---")
    h_counter = Counter(health_per_child.values())
    print(f"  healthy=True (control): {h_counter.get(True, 0)}")
    print(f"  healthy=False (ADHD):   {h_counter.get(False, 0)}")
    print(f"  None/missing:           {h_counter.get(None, 0)}")

    print(f"\n--- Demographics ---")
    ages_clean = [a for a in age_per_child.values() if a is not None]
    if ages_clean:
        print(f"  Ages: mean={statistics.mean(ages_clean):.2f}, "
              f"min={min(ages_clean)}, max={max(ages_clean)}")
    print(f"  Genders: {Counter(gender_per_child.values())}")
    print(f"  Hands:   {Counter(hand_per_child.values())}")

    print(f"\n--- Trials per child (total / valid) ---")
    for child in sorted(trials_total_per_child):
        health = health_per_child.get(child)
        hstr = "CTRL" if health is True else ("ADHD" if health is False else "????")
        total = trials_total_per_child[child]
        valid = trials_valid_per_child.get(child, 0)
        tasks = trials_by_task_per_child.get(child, {})
        print(f"  {child} [{hstr}]: {total} total, {valid} valid, tasks={tasks}")

    total = sum(trials_total_per_child.values())
    valid = sum(trials_valid_per_child.values())
    print(f"\n  GRAND TOTAL: {total} trials, {valid} valid")

    print(f"\n--- Sample length distribution (raw x signal) ---")
    if n_samples_distribution:
        print(f"  n trials with data: {len(n_samples_distribution)}")
        print(f"  min:    {min(n_samples_distribution)}")
        print(f"  max:    {max(n_samples_distribution)}")
        print(f"  mean:   {statistics.mean(n_samples_distribution):.1f}")
        print(f"  median: {statistics.median(n_samples_distribution):.1f}")
        # histogramme simple
        buckets = [0, 100, 200, 300, 400, 500, 1000, 10000]
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            n = sum(1 for x in n_samples_distribution if lo <= x < hi)
            print(f"  [{lo:5d}, {hi:5d}): {n}")

    print(f"\n--- Channel availability (count of trials with non-empty channel) ---")
    print(f"  pressure:  {has_pressure_count} / {total}")
    print(f"  elevation: {has_elevation_count} / {total}")
    print(f"  azimuth:   {has_azimuth_count} / {total}")

    print(f"\n--- Extractor availability ---")
    print(f"  SSVp: {has_ssvp_count} / {total}")
    print(f"  SSVn: {has_ssvn_count} / {total}")
    print(f"  IIX:  {has_iix_count} / {total}")

    if errors:
        print(f"\n--- ERRORS ---")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_dataset.py /path/to/json/folder")
        sys.exit(1)
    main(sys.argv[1])