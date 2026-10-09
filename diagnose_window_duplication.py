"""
diagnose_window_duplication.py

diagnose_cache_directly.py proved the 79-sample cluster exists in
cache/processed.npz itself, even after the min_run=2 fix removed far
more rows than before. In a normal sliding window, each window's target
y = soh_norm[i+w+h-1] maps to a UNIQUE array position -- no two
different windows from the same cell should share a target index. So
27 different windows from one cell landing on the exact same y value
means one of two very different things:

  (a) The FULL INPUT WINDOW (X, not just y) is bit-identical across
      those samples too -- meaning whole windows are being duplicated
      somewhere (a real indexing/loop bug in make_windows() or
      upstream), not a data coincidence.
  (b) Only y matches, but X differs across the 79 samples -- meaning
      27+ genuinely different rows in these cells' surviving data
      independently hold nearly the same capacity reading. Odd, but a
      real (if strange) data characteristic rather than a bug.

This script checks which one it is, directly from the cache.

Run:
    python diagnose_window_duplication.py
"""

import numpy as np
import config
import os

def main():
    cache_path = os.path.join(config.CACHE_DIR, "processed.npz")
    data = np.load(cache_path, allow_pickle=True)

    lab_test = data["lab_test"]
    cell_test = data["cell_test"]
    y_test = data["y_test"]
    X_test = data["X_test"]

    mask = lab_test == "XJTU"
    yt = y_test[mask]
    ct = cell_test[mask]
    Xt = X_test[mask]

    min_val = yt.min()
    exact_match = np.isclose(yt, min_val, atol=1e-4)
    idx = np.where(exact_match)[0]
    print(f"Found {len(idx)} samples at the minimum value ({min_val:.6f})\n")

    Xs = Xt[idx]  # shape (79, window, 1)
    cs = ct[idx]

    # Check 1: are ALL 79 input windows bit-identical to each other?
    first = Xs[0]
    all_identical = all(np.array_equal(Xs[i], first) for i in range(len(Xs)))
    print(f"Are ALL {len(Xs)} input windows bit-identical to sample 0? {all_identical}")

    if not all_identical:
        # Check pairwise identical groups instead -- maybe they cluster
        # into a few distinct repeated windows rather than one giant
        # duplicate, or maybe genuinely all different
        n_unique = len(np.unique(Xs.reshape(len(Xs), -1), axis=0))
        print(f"Number of UNIQUE input windows among these {len(Xs)} samples: {n_unique}")
        if n_unique < len(Xs):
            print("  -> Some duplication exists but not total -- multiple "
                  "distinct repeated-window groups.")
        else:
            print("  -> All 79 input windows are genuinely DIFFERENT from "
                  "each other, despite sharing the same target y. This "
                  "means 79 different actual rows in the surviving data "
                  "independently hold a near-identical capacity reading "
                  "-- NOT a duplication/indexing bug. Look at the raw "
                  "per-cell values directly next.")

    # Per-cell breakdown of X uniqueness
    print("\nPer-cell breakdown:")
    for cell_id in np.unique(cs):
        cell_mask = cs == cell_id
        cell_X = Xs[cell_mask]
        n_unique_cell = len(np.unique(cell_X.reshape(len(cell_X), -1), axis=0))
        print(f"  {cell_id}: {cell_mask.sum()} samples, {n_unique_cell} unique input windows")

    # Print the actual last-timestep values (closest proxy to the raw
    # value feeding the target, in normalized space) for a handful of
    # samples to see if the FULL window content varies smoothly (real
    # data) or is flat/repeated (artifact)
    print("\nFirst 5 samples' full window (last 5 timesteps shown), to eyeball:")
    for i in range(min(5, len(Xs))):
        vals = Xs[i, -5:, 0]
        print(f"  sample {i} (cell={cs[i]}): last 5 timesteps = {vals}")


if __name__ == "__main__":
    main()
