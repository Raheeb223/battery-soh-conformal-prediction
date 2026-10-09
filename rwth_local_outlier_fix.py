"""
rwth_local_outlier_fix.py

Confirmed finding: ALL 48 RWTH cells contain isolated single-row capacity
outliers -- most consistently at cycle=1 (a value ~26-27% of the cell's
real baseline capacity, remarkably consistent across independent cells --
e.g. sanyo_030: 0.288/1.076=26.8%, sanyo_004: 0.281/1.059=26.5%,
sanyo_006: 0.283/1.064=26.6%, sanyo_009: 0.293/1.069=27.4%), plus ~15-18
MORE such outliers scattered through each cell's later sequence (count
roughly tracks the number of distinct session files per cell, 34-41),
plus one severe case (RWTH_ep_sanyo_019) with several consecutive
near-zero readings.

The consistency of the ~26-27% ratio across unrelated cells rules out
random sensor noise -- this is very likely a systematic measurement
artifact at the START of each concatenated session file (a formation/
characterization-style partial reading, consistent with the "BOL"
(Beginning Of Life) session-type codes seen in RWTH's raw filenames),
not a broken sensor. Total impact: 787 of 114,716 rows (0.686%).

This module provides strip_local_outliers(df), using the EXACT verified
detector from diagnose_rwth_local_outliers.py (same thresholds, same
edge-case handling for first/last rows in a cell -- tested against 5
scenarios including the real cycle=1 pattern found in this data).

--- HOW TO INTEGRATE INTO YOUR REAL rwth_loader.py ---
Add near the top:
    from rwth_local_outlier_fix import strip_local_outliers

Then, in load_all(), right before the final `return pd.concat(frames, ...)`
(or equivalently, right after building each per-cell DataFrame, before
appending it to `frames` -- whichever matches your actual loader's
structure), call:
    df = strip_local_outliers(df)

Since this project's convention (established in xjtu_loader.py) is to
strip per-cell right after each cell's DataFrame is built, the cleanest
integration point is likely wherever your loader currently builds one
cell's cycle/capacity DataFrame, analogous to xjtu_loader.py's
load_xjtu_cell() calling strip_calibration_blocks() before returning.
Paste your actual rwth_loader.py here if you want the exact insertion
point identified rather than guessed.
"""

import numpy as np
import pandas as pd

LOCAL_OUTLIER_THRESHOLD = 0.15


def find_local_outliers(capacity, threshold=LOCAL_OUTLIER_THRESHOLD):
    """Identical logic to diagnose_rwth_local_outliers.py's verified
    detector -- flags interior single-point spikes/dips, first/last-row
    edge cases (this is what catches the cycle=1 artifact), and any
    exact zero/negative capacity value."""
    vals = np.asarray(capacity, dtype=float)
    n = len(vals)
    outlier_idx = []

    for i in range(1, n - 1):
        left, mid, right = vals[i - 1], vals[i], vals[i + 1]
        neighbor_diff = abs(left - right)
        mid_dev_left = abs(mid - left)
        mid_dev_right = abs(mid - right)
        scale = max(abs(left), abs(right), 1e-9)
        if (mid_dev_left / scale > threshold and
                mid_dev_right / scale > threshold and
                neighbor_diff / scale < threshold / 2):
            outlier_idx.append(i)

    if n >= 3:
        first, second, third = vals[0], vals[1], vals[2]
        scale = max(abs(second), abs(third), 1e-9)
        if (abs(first - second) / scale > threshold and
                abs(second - third) / scale < threshold / 2):
            outlier_idx.append(0)

        last, second_last, third_last = vals[-1], vals[-2], vals[-3]
        scale = max(abs(second_last), abs(third_last), 1e-9)
        if (abs(last - second_last) / scale > threshold and
                abs(second_last - third_last) / scale < threshold / 2):
            outlier_idx.append(n - 1)

    zero_idx = [i for i in range(n) if vals[i] <= 0]
    return sorted(set(outlier_idx) | set(zero_idx))


def strip_local_outliers(df: pd.DataFrame, cell_id_col="cell_id",
                          capacity_col="capacity", cycle_col="cycle",
                          verbose=True) -> pd.DataFrame:
    """Drops rows flagged by find_local_outliers(), computed per-cell on
    the cycle-sorted capacity sequence. Applies to ALL cells (unlike the
    XJTU fix, this artifact affects all 48/48 RWTH cells, not a specific
    sub-batch) -- if you reuse this module for a different dataset,
    consider whether a per-cell or per-dataset scope restriction is
    appropriate there instead."""
    cleaned_groups = []
    total_dropped = 0

    for cell_id, group in df.groupby(cell_id_col):
        g = group.sort_values(cycle_col).reset_index(drop=True)
        outlier_positions = find_local_outliers(g[capacity_col].to_numpy())
        if outlier_positions:
            n_dropped = len(outlier_positions)
            total_dropped += n_dropped
            if verbose:
                print(f"  [RWTH-fix] {cell_id}: removed {n_dropped} local "
                      f"outlier row(s) (session-boundary transients / "
                      f"zero-capacity glitches, not real degradation)")
            drop_mask = np.zeros(len(g), dtype=bool)
            drop_mask[outlier_positions] = True
            g = g.loc[~drop_mask].reset_index(drop=True)
        cleaned_groups.append(g)

    result = pd.concat(cleaned_groups, ignore_index=True)
    if verbose:
        print(f"  [RWTH-fix] TOTAL: removed {total_dropped} rows across "
              f"{df[cell_id_col].nunique()} cells "
              f"({100*total_dropped/len(df):.3f}% of all RWTH rows)")
    return result


if __name__ == "__main__":
    # Smoke test reproducing the exact real pattern: cycle=1 artifact
    # present, plus a mid-sequence spike, plus a genuine zero-run
    rng = np.random.default_rng(0)
    n = 100
    capacity = 1.06 - 0.0003 * np.arange(n) + rng.normal(0, 0.001, n)
    capacity[0] = 0.28  # the cycle=1 artifact, ~26% of baseline
    capacity[50] = 0.3  # a mid-sequence single-point spike
    capacity[70:73] = 0.0  # a genuine zero-run (like sanyo_019)

    df = pd.DataFrame({
        "cycle": np.arange(1, n + 1),
        "capacity": capacity,
        "cell_id": "TEST_CELL",
    })

    cleaned = strip_local_outliers(df)
    print(f"\nBefore: {len(df)} rows, after: {len(cleaned)} rows")
    assert len(df) - len(cleaned) == 1 + 1 + 3  # cycle1 + spike + 3 zeros
    assert (cleaned["capacity"] <= 0).sum() == 0
    assert 0.28 not in cleaned["capacity"].values
    print("Smoke test PASSED: cycle-1 artifact, mid-sequence spike, and "
          "zero-run all correctly removed; clean baseline values retained.")
