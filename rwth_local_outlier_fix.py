"""
Removal of isolated invalid capacity readings in RWTH cells.

The raw RWTH data contains isolated single-row capacity values that deviate
sharply from both neighbours while the neighbours agree with each other: the
signature of a single bad reading inside a smooth trend, as opposed to genuine
fast degradation. Most occur at session starts (see rwth_loader.py); one cell
(RWTH_ep_sanyo_019) also contains several consecutive near-zero readings.

strip_local_outliers(df) flags a row as an outlier when it deviates from both
immediate neighbours by more than LOCAL_OUTLIER_THRESHOLD (defined below), with
explicit handling of the first and last row of each cell. It is called per
cell by rwth_loader.load_rwth_cell().

Running this file directly executes the built-in test scenarios.
"""

import numpy as np
import pandas as pd

LOCAL_OUTLIER_THRESHOLD = 0.15


def find_local_outliers(capacity, threshold=LOCAL_OUTLIER_THRESHOLD):
    """Flags interior single-point spikes/dips, first/last-row
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
    sub-batch) -- when reusing this module for a different dataset,
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
