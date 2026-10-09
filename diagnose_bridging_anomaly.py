"""
diagnose_bridging_anomaly.py

windowing_bridge_sensitivity.py reported EXACTLY 100.00% bridging for
CALCE, NASA, and Oxford -- three datasets with NO documented row-removal
history (only XJTU and RWTH have that, per Section 3.2). A 100% rate
across three unrelated datasets is far more consistent with the
bridging DETECTOR being miscalibrated for how these datasets' 'cycle'
column is actually numbered, than with a genuine finding.

This prints the raw cycle-index values for a handful of windows from
each of these three datasets, so we can see directly what pattern is
actually there -- rather than guess.

Run from your project root:
    python diagnose_bridging_anomaly.py
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
    window_cycles = cache["window_cycle_indices_test"]
    cell_test = cache["cell_test"]

    for ds in ("CALCE", "NASA", "Oxford", "RWTH"):  # RWTH included as a known-good baseline
        mask = lab_test == ds
        n = int(mask.sum())
        if n == 0:
            print(f"--- {ds}: no test windows found ---\n")
            continue
        idx = np.where(mask)[0][: min(5, n)]
        print(f"--- {ds} (n={n} test windows) -- first {len(idx)} windows' raw cycle values ---")
        for i in idx:
            cw = window_cycles[i]
            diffs = np.diff(cw)
            print(f"  cell={cell_test[i]}  cycles={cw}  step-diffs={diffs}")
        print()

    print("="*78)
    print("Look at the step-diffs above for CALCE/NASA/Oxford. If they are "
          "consistently something OTHER than 1 (e.g. always 5, or always a "
          "fraction, or wildly irregular non-1 values), that confirms this "
          "dataset's 'cycle' column does not use a simple +1-per-row "
          "convention, and the bridging detector needs to compare against "
          "THAT dataset-specific step size, not a hardcoded '!= 1' check. "
          "Compare directly against RWTH's diffs (the known-good case, "
          "correctly near-zero bridging) as a sanity reference.")


if __name__ == "__main__":
    main()
