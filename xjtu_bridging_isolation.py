"""
Which XJTU test windows span removed rows?

xjtu_loader.py removes sentinel readings from the eight Sim_satellite_* cells and keeps the original
cycle numbers, so a window can join readings on either side of a removed block. This script counts,
from the windows' cycle indices, the fraction of test windows with a gap, separately for the
Sim_satellite_* cells and for the other XJTU cells, which no correction touched. A gap in the other
cells reflects the dataset's own sampling; a gap only in the corrected cells is caused by the
correction.

Run:
    python xjtu_bridging_isolation.py
"""

import os
import json

import numpy as np

import config


def main():
    path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found.")
        return
    cache = np.load(path, allow_pickle=True)

    lab_test = cache["lab_test"]
    cell_test = cache["cell_test"]
    window_cycles = cache["window_cycle_indices_test"]

    xjtu_mask = lab_test == "XJTU"
    if xjtu_mask.sum() == 0:
        print("  [!] No XJTU test windows found.")
        return

    is_satellite = np.array(["Sim_satellite" in str(c) for c in cell_test])

    diffs = np.diff(window_cycles, axis=1)
    bridging = np.any(diffs != 1, axis=1)

    sat_mask = xjtu_mask & is_satellite
    other_mask = xjtu_mask & ~is_satellite

    n_sat = int(sat_mask.sum())
    n_other = int(other_mask.sum())
    frac_bridging_sat = float(bridging[sat_mask].mean()) if n_sat > 0 else float("nan")
    frac_bridging_other = float(bridging[other_mask].mean()) if n_other > 0 else float("nan")

    print(f"{'='*78}")
    print("XJTU BRIDGING ISOLATION: Sim_satellite cells vs. other 47 cells")
    print(f"{'='*78}\n")
    print(f"Sim_satellite_* cells:  n={n_sat:5d}  bridging fraction={frac_bridging_sat:.4f}")
    print(f"Other 47 cells:         n={n_other:5d}  bridging fraction={frac_bridging_other:.4f}")

    if n_sat > 0 and n_other > 0:
        if frac_bridging_sat < 0.02:
            print("\nSim_satellite bridging is near-zero, consistent with the "
                  "documented renumbering-erases-the-gap issue (this confirms "
                  "the erasure, it does not mean satellite cells are actually "
                  "free of bridging risk -- see preprocessing.py's own "
                  "docstring).")
        if abs(frac_bridging_other - 0.1443) < 0.03:
            print("\nOther-47-cells bridging is close to the overall 14.43% "
                  "figure -- this supports the hypothesis that XJTU's "
                  "bridging is MOSTLY native cycle-logging sparsity, not "
                  "correction-related, similar to CALCE/NASA/Oxford.")
        elif frac_bridging_other < 0.05:
            print("\nOther-47-cells bridging is LOW -- most of the overall "
                  "14.43% must be concentrated somewhere unexpected; "
                  "investigate further before drawing a conclusion.")

    result = {
        "n_sim_satellite": n_sat, "frac_bridging_sim_satellite": frac_bridging_sat,
        "n_other_47_cells": n_other, "frac_bridging_other_47_cells": frac_bridging_other,
    }
    out_path = os.path.join(config.OUTPUT_DIR, "xjtu_bridging_isolation.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
