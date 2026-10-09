"""
diagnose_short_duplicate_runs.py

diagnose_xjtu_value_by_cell.py just proved the persistent 79-sample
cluster comes from exactly 3 cells: Sim_satellite_1, Sim_satellite_3,
Sim_satellite_7 -- all already confirmed to have had their >=5-row
calibration blocks correctly removed (90/85/90 rows respectively).

Hypothesis: strip_calibration_blocks()'s min_run=5 threshold was chosen
to match the ORIGINAL bug signature (79 samples, runs of 5+), but these
3 cells may ALSO contain SHORTER duplicate runs (2-4 identical
consecutive values) that never got removed, since min_run=5 only
catches runs at or above that length. A shorter flat stretch in the raw
per-cell sequence can still make multiple nearby windows' single-point
target y land on the same repeated value.

This script re-loads XJTU fresh and scans ONLY these 3 specific cells
for ANY duplicate run (min_run=2, not 5) in the capacity column, to see
if a shorter residual pocket exists that would explain the persistent
cluster.

Run:
    python diagnose_short_duplicate_runs.py
"""

import numpy as np
import config
import xjtu_loader

TARGET_CELLS = [
    "XJTU_Batch-6_Sim_satellite_1",
    "XJTU_Batch-6_Sim_satellite_3",
    "XJTU_Batch-6_Sim_satellite_7",
]


def find_runs(values, min_run=2):
    values = np.asarray(values, dtype=float)
    runs = []
    run_start = 0
    n = len(values)
    for i in range(1, n + 1):
        same = i < n and values[i] == values[i - 1]
        if not same:
            run_len = i - run_start
            if run_len >= min_run:
                runs.append((run_start, i, run_len, float(values[run_start])))
            run_start = i
    return runs


def main():
    print("Reloading XJTU fresh (this uses your current xjtu_loader.py, "
          "i.e. AFTER the min_run=5 calibration-block strip already ran)...")
    df = xjtu_loader.load_all()

    for cell_id in TARGET_CELLS:
        sub = df[df["cell_id"] == cell_id].sort_values("cycle")
        if sub.empty:
            print(f"\n{cell_id}: NOT FOUND in loaded data (unexpected)")
            continue

        capacity = sub["capacity"].to_numpy()
        print(f"\n{cell_id}: {len(capacity)} rows remaining after the existing strip")

        # Check every run length from 2 up to 4 specifically (5+ should
        # already be empty, since strip_calibration_blocks() already ran
        # inside xjtu_loader.load_all() itself)
        runs_2plus = find_runs(capacity, min_run=2)
        runs_5plus = find_runs(capacity, min_run=5)

        if runs_5plus:
            print(f"  [!!!] UNEXPECTED: {len(runs_5plus)} run(s) of length >=5 "
                  f"still present -- the existing strip should have caught "
                  f"these. Something is wrong with strip_calibration_blocks() "
                  f"itself, not just the threshold.")
            for start, end, length, val in runs_5plus:
                print(f"      rows[{start}:{end}] len={length} value={val}")

        short_runs = [r for r in runs_2plus if r[2] < 5]
        if short_runs:
            print(f"  Found {len(short_runs)} SHORT duplicate run(s) (length 2-4, "
                  f"below the min_run=5 threshold that strip_calibration_blocks() "
                  f"uses):")
            total_short_rows = 0
            for start, end, length, val in short_runs:
                print(f"      rows[{start}:{end}] len={length} value={val}")
                total_short_rows += length
            print(f"  Total rows in short runs: {total_short_rows}")
        else:
            print(f"  No duplicate runs of length 2-4 found either -- "
                  f"this cell's remaining data has no obvious repeated-value "
                  f"artifact at all. The persistent cluster must have a "
                  f"different cause (e.g. genuinely coincidental physical "
                  f"values, or something in the windowing step itself "
                  f"rather than the raw per-cell data).")


if __name__ == "__main__":
    main()
