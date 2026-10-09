"""
diagnose_sensitivity_anomaly.py

The XJTU before/after sensitivity comparison shows a counterintuitive
result: disabling the sentinel-value fix (which ONLY removes rows, never
adds them) somehow DECREASED XJTU's test-window count (2605 -> 2387 at
window=20), despite raw row count correctly increasing (20566 -> 27642).

Since cell-to-split assignment is a seeded shuffle independent of the
fix, the same cells land in train/val/test either way -- so this can
only happen if a Sim_satellite cell in the TEST split lost more windows
than it gained, most plausibly via add_soh()'s existing (SOH > 0) &
(SOH < 1.10) bounds filter: if the fix is disabled, a cell's first 1-3
raw rows (used to compute its reference capacity) could themselves be
contaminated sentinel values, corrupting that cell's ENTIRE SOH curve
and pushing many of its rows outside the bounds filter -- a real,
explainable side effect, not a bug in the toggle itself.

This script tests that hypothesis directly: for each Sim_satellite cell
that lands in the TEST split, it prints the computed reference capacity
and resulting SOH range with the fix ON vs OFF, so we can confirm or
rule this out with actual numbers rather than argue from theory.

Run from your project root (same folder as config.py):
    python diagnose_sensitivity_anomaly.py
"""

import os
import numpy as np
import pandas as pd
import config
import preprocessing
import xjtu_loader


def compute_reference_and_soh(df, cell_id):
    """Mirrors preprocessing.py's add_soh() reference-capacity logic
    exactly, for one cell, so we can inspect it directly."""
    sub = df[df["cell_id"] == cell_id].sort_values("cycle")
    if sub.empty:
        return None
    capacity = sub["capacity"].to_numpy()
    ref = np.median(capacity[: min(3, len(capacity))])
    soh = capacity / ref
    n_out_of_bounds = int(np.sum((soh <= 0) | (soh >= 1.10)))
    return {
        "cell_id": cell_id,
        "n_rows": len(capacity),
        "first_3_raw_values": capacity[: min(3, len(capacity))].tolist(),
        "reference_capacity": float(ref),
        "soh_min": float(soh.min()),
        "soh_max": float(soh.max()),
        "n_rows_out_of_bounds": n_out_of_bounds,
        "n_rows_surviving_bounds_filter": len(capacity) - n_out_of_bounds,
    }


def main():
    print("Step 1: determine which XJTU cells land in the TEST split "
          "(seeded, independent of the fix toggle)...")
    raw_all = preprocessing.load_all_raw()
    df_all = preprocessing.add_soh(raw_all)
    train_ids, val_ids, test_ids = preprocessing.cell_independent_split(df_all)

    xjtu_test_cells = [c for c in test_ids if c.startswith("XJTU")]
    satellite_test_cells = [c for c in xjtu_test_cells if "Sim_satellite" in c]
    print(f"  XJTU cells in TEST split: {len(xjtu_test_cells)}")
    print(f"  Of those, Sim_satellite cells: {satellite_test_cells}")

    if not satellite_test_cells:
        print("\n  [!] No Sim_satellite cells are in the TEST split. This "
              "means the fix toggle should NOT have been able to change "
              "the test-window count at all -- if you're still seeing a "
              "difference, the cause is something else entirely (e.g. a "
              "non-deterministic split, or the two runs used different "
              "RANDOM_SEED values). Check config.RANDOM_SEED was identical "
              "for both runs before investigating further.")
        return

    print(f"\nStep 2: comparing reference-capacity computation WITH vs "
          f"WITHOUT the fix, for each affected test-split cell...")

    for fix_enabled in (True, False):
        os.environ["XJTU_APPLY_SENTINEL_FIX"] = "1" if fix_enabled else "0"
        import importlib
        importlib.reload(xjtu_loader)  # re-read the env var fresh

        label = "WITH fix" if fix_enabled else "WITHOUT fix"
        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"{'='*70}")

        xjtu_df = xjtu_loader.load_all()
        for cell_id in satellite_test_cells:
            result = compute_reference_and_soh(xjtu_df, cell_id)
            if result is None:
                print(f"  {cell_id}: NOT FOUND in this run's data")
                continue
            print(f"  {cell_id}:")
            print(f"    rows={result['n_rows']}  "
                  f"first_3_raw_values={[round(v,4) for v in result['first_3_raw_values']]}")
            print(f"    reference_capacity={result['reference_capacity']:.4f}  "
                  f"SOH range=[{result['soh_min']:.4f}, {result['soh_max']:.4f}]")
            print(f"    rows OUT of (0, 1.10) SOH bounds: "
                  f"{result['n_rows_out_of_bounds']} of {result['n_rows']} "
                  f"(these would be silently dropped by add_soh()'s existing filter)")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("If 'WITHOUT fix' shows a reference_capacity wildly different from "
          "'WITH fix' for the same cell, and/or a much higher "
          "'rows OUT of bounds' count, this CONFIRMS the hypothesis: "
          "disabling the sentinel-value fix corrupts that cell's reference "
          "capacity (because a sentinel value landed in its first 3 raw "
          "rows), which cascades into many of its OTHER rows failing the "
          "existing SOH bounds filter -- a real, explainable, and reportable "
          "side effect, not a bug in the sensitivity toggle. If the "
          "reference capacities are similar and bounds-filter counts are "
          "similar, this hypothesis is NOT the explanation and the anomaly "
          "needs a different investigation (e.g. inspect make_windows() "
          "directly for this specific cell in both cache versions).")


if __name__ == "__main__":
    main()
