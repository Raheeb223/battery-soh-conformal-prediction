"""
diagnose_satellite_bridging.py

xjtu_bridging_isolation.py found the OPPOSITE of what preprocessing.py's
own docstring predicted: Sim_satellite_* cells show 100% bridging (not
near-zero, as the "renumbering erases the gap" documentation expected),
while the other 47 XJTU cells show a clean 0%. This directly contradicts
the documented expectation and needs to be looked at directly, not
guessed about.

This prints the raw cycle values for several Sim_satellite test windows,
so we can see exactly what pattern is actually there.

Run from your project root:
    python diagnose_satellite_bridging.py
"""

import os
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

    is_satellite = np.array(["Sim_satellite" in str(c) for c in cell_test])
    xjtu_mask = lab_test == "XJTU"
    sat_mask = xjtu_mask & is_satellite

    n_sat = int(sat_mask.sum())
    print(f"Total Sim_satellite test windows: {n_sat}\n")

    idx = np.where(sat_mask)[0]
    unique_cells = np.unique(cell_test[idx])
    print(f"Distinct Sim_satellite cells in test split: {list(unique_cells)}\n")

    for uc in unique_cells:
        cell_idx = idx[cell_test[idx] == uc]
        print(f"--- {uc} ({len(cell_idx)} test windows) -- first 5 ---")
        for i in cell_idx[:5]:
            cw = window_cycles[i]
            diffs = np.diff(cw)
            print(f"  cycles={cw}")
            print(f"  step-diffs={diffs}")
        print()

    print("="*78)
    print("If the cycle values here show clean 1,2,3,4...N with no visible "
          "gap, but the bridging detector STILL flags them, that would mean "
          "a bug in the detector logic itself for this specific case -- "
          "check the printed step-diffs directly against what's expected. "
          "If instead there IS a real, visible jump (e.g. a gap of 2+ "
          "somewhere in the middle), that means the documented "
          "'renumbering erases the gap' claim does not match what "
          "xjtu_loader.py actually does -- worth checking that function "
          "directly, since this is better news for the manuscript (the "
          "correction's effect turns out to be directly measurable after "
          "all), not worse.")


if __name__ == "__main__":
    main()
