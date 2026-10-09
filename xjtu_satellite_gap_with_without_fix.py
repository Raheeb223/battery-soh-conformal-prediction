"""
xjtu_satellite_gap_with_without_fix.py

diagnose_satellite_bridging.py revealed that Sim_satellite_* cells' raw
cycle values are NOT a clean renumbered 1..N sequence (contradicting
preprocessing.py's own docstring claim that renumbering erases the
gap). Instead there's a baseline step of ~5-6 (plausibly native to the
satellite duty-cycle simulation protocol) plus larger, irregular jumps
(11-28) scattered throughout (plausibly where sentinel-value calibration
rows were removed).

This isolates the two by re-running preprocessing WITH and WITHOUT the
sentinel-value fix (config.XJTU_APPLY_SENTINEL_FIX, the same toggle used
in the earlier XJTU sensitivity experiment) and comparing the SAME
satellite cells' cycle-gap patterns under both conditions:

  - A gap present in BOTH conditions is native to the protocol, not
    correction-related.
  - A gap present ONLY with the fix ON (i.e. it disappears when the fix
    is OFF, because the previously-removed sentinel rows are back,
    filling in what looked like a gap) is direct, empirical evidence of
    exactly where the correction removed rows.

This does NOT retrain any model or touch conformal calibration -- it
only re-runs preprocessing's data loading and windowing step twice,
which is fast, and compares raw cycle-index arrays.

Run from your project root:
    python xjtu_satellite_gap_with_without_fix.py
"""

import os

import numpy as np

import config
import preprocessing


def get_satellite_windows(fix_enabled: bool):
    os.environ["XJTU_APPLY_SENTINEL_FIX"] = "1" if fix_enabled else "0"
    import importlib
    import xjtu_loader
    importlib.reload(xjtu_loader)
    importlib.reload(preprocessing)

    raw = preprocessing.load_all_raw()
    df = preprocessing.add_soh(raw)
    train_ids, val_ids, test_ids = preprocessing.cell_independent_split(df)

    # make_windows() needs real soh_mean/soh_std dicts to normalize SOH
    # (not needed for THIS comparison's purpose, but required by the
    # function signature) -- compute them properly from train cells,
    # matching what run() already does, rather than pass empty dicts.
    train_df = df[df["cell_id"].isin(train_ids)]
    soh_mean = train_df.groupby("dataset")["soh"].mean().to_dict()
    soh_std = train_df.groupby("dataset")["soh"].std().to_dict()
    for k in soh_std:
        if soh_std[k] == 0 or np.isnan(soh_std[k]):
            soh_std[k] = 1.0

    X, y, labels, cells, cyc = preprocessing.make_windows(
        df, test_ids, soh_mean, soh_std, window=config.INPUT_WINDOW
    )

    is_satellite = np.array(["Sim_satellite" in str(c) for c in cells])
    xjtu_mask = labels == "XJTU"
    sat_mask = xjtu_mask & is_satellite

    return cells[sat_mask], cyc[sat_mask]


def main():
    print("Running preprocessing WITH the sentinel-value fix...")
    cells_with, cyc_with = get_satellite_windows(fix_enabled=True)
    print(f"  {len(cells_with)} Sim_satellite test windows found\n")

    print("Running preprocessing WITHOUT the sentinel-value fix...")
    cells_without, cyc_without = get_satellite_windows(fix_enabled=False)
    print(f"  {len(cells_without)} Sim_satellite test windows found\n")

    common_cells = sorted(set(cells_with) & set(cells_without))
    print(f"Cells present in both conditions' test split: {common_cells}\n")

    for cell_id in common_cells:
        idx_with = np.where(cells_with == cell_id)[0][:3]
        idx_without = np.where(cells_without == cell_id)[0][:3]
        print(f"--- {cell_id} ---")
        print("  WITH fix:")
        for i in idx_with:
            print(f"    cycles={cyc_with[i]}")
        print("  WITHOUT fix:")
        for i in idx_without:
            print(f"    cycles={cyc_without[i]}")
        print()

    print("="*78)
    print("Compare the WITH vs WITHOUT cycle values directly for the SAME "
          "cell. Any gap value that appears WITH the fix but is FILLED IN "
          "(smaller step, or an intermediate cycle number appears) WITHOUT "
          "the fix is direct evidence that gap was caused by the "
          "correction removing rows -- not native protocol structure. Any "
          "gap that looks IDENTICAL in both conditions is native, "
          "unrelated to the correction.")


if __name__ == "__main__":
    main()
