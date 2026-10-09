"""
Characterize the LOW-side rows in outputs/calce_v2_verification.csv (capacity
well below the local trend): are they the first group of a new file (an
expected reset point), and what does their terminal voltage look like (near
4.0 V suggests no discharge occurred in that group at all; near the cutoff
suggests a real but very small discharge)?

Reloads each affected cell in full (calce_loader_v2, so cycle_raw, source_file
and terminal_voltage are available) and joins it to the outlier rows on
(cell, cycle) rather than reopening individual files.

Run:  python analyze_low_side_outliers.py
"""

import os

import numpy as np
import pandas as pd

import calce_loader_v2 as ld
import config


def main():
    csv_path = os.path.join(config.OUTPUT_DIR, "calce_v2_verification.csv")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"{csv_path} is missing or empty -- run verify_calce_v2.py first"); return
    out = pd.read_csv(csv_path)
    low = out[out["capacity"] < out["local_median"]].copy()
    if low.empty:
        print("no low-side outlier rows")
        return

    dirs = {}
    for d in [d for lst in config.CALCE_DIRS.values() for d in lst]:
        dirs[os.path.basename(d.rstrip("/\\"))] = d

    rows = []
    for cell in low["cell"].unique():
        d = dirs.get(cell)
        if d is None or not os.path.isdir(d):
            print(f"[{cell}] directory not found -- skipping"); continue
        full = ld.load_calce_cell(d, cell, "CS2" if cell.startswith("CS2") else "CX2")
        full = full.reset_index(drop=True)
        full["pos_in_file"] = full.groupby("source_file", sort=False).cumcount()
        full["n_in_file"] = full.groupby("source_file")["source_file"].transform("size")
        merged = low[low["cell"] == cell].merge(
            full[["cycle", "source_file", "pos_in_file", "n_in_file", "terminal_voltage", "cycle_raw"]],
            on="cycle", how="left", suffixes=("", "_full"))
        rows.append(merged)

    m = pd.concat(rows, ignore_index=True)
    first_of_file = m["pos_in_file"] == 0
    exact_zero = m["capacity"] == 0

    print(f"{len(m)} low-side outlier rows analyzed\n")
    print(f"first group of their file:  {first_of_file.sum()} of {len(m)} ({100*first_of_file.mean():.0f}%)")
    print(f"exact capacity == 0:        {exact_zero.sum()} of {len(m)} ({100*exact_zero.mean():.0f}%)")
    print(f"  of those, first-of-file:  {(exact_zero & first_of_file).sum()} of {exact_zero.sum()}")
    v = m.loc[exact_zero, "terminal_voltage"].dropna()
    if len(v):
        print(f"\nterminal voltage at exact-zero rows (n={len(v)}): "
              f"median {v.median():.2f} V, range [{v.min():.2f}, {v.max():.2f}] "
              f"(near 4.0 V => no discharge happened in that group; near a cutoff ~2.7 V => "
              f"a real but tiny discharge)")
    else:
        print("\nno voltage column recovered for the exact-zero rows -- cannot distinguish the two cases")

    print("\nnot first-of-file (genuinely mid-file, not just a fresh session start):")
    interior = m[~first_of_file]
    if interior.empty:
        print("  none")
    else:
        print(interior[["cell", "file", "cycle", "pos_in_file", "n_in_file", "capacity", "local_median"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
