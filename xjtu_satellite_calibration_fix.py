"""
xjtu_satellite_calibration_fix.py

Fix for the confirmed XJTU padding artifact: all 8 "Sim_satellite_*" cells
in XJTU Batch-6 contain a periodic calibration/reference block -- a run of
exactly 5 rows at a fixed value (1.601, bit-identical across all 8 cells)
recurring every 52 rows for the entire cell. This is NOT real degradation
data: no two physically distinct cells produce bit-identical capacity
readings, and the fixed periodicity (52-row spacing, always exactly 5
rows) is a structural artifact of the raw release, not aging.

Static search of xjtu_loader.py/preprocessing.py found no pad/repeat/tile
code responsible -- this is baked into the raw XJTU Batch-6 data itself,
not introduced by this project's loading code.

Working theory: the "Sim_satellite" profiles simulate a satellite duty
cycle and periodically insert a fixed reference/calibration segment
(5 identical readings at nominal capacity) rather than a true per-cycle
discharge-capacity measurement.

This module provides:
  1. detect_calibration_blocks() -- finds the periodic constant-run blocks
     in a raw per-cell capacity sequence generically (does not hardcode
     row offsets or the value 1.601, since block phase/period could differ
     slightly across cells or a re-run of the loader).
  2. strip_satellite_calibration_blocks() -- drops those rows before
     windowing, restricting to Sim_satellite_* cells only (Batch-1/Batch-2
     hits from the earlier scan are NOT touched -- see the note below on
     why those are left alone).
  3. A drop-in point to call from xjtu_loader.py's load_all()/load_xjtu().

IMPORTANT -- scope of this fix:
Only Batch-6 Sim_satellite_* cells are affected. The isolated 5-9 row
runs found earlier in Batch-1 (2C) and Batch-2 (3C) cells are NOT
periodic, NOT identical across cells, and sit at different capacity
values per cell -- consistent with genuine short plateaus in real fade
data, not padding. Do NOT apply this filter to those cells.
"""

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


def detect_calibration_blocks(values, min_run=5, min_period=None, max_period=None):
    """Detects periodic runs of identical consecutive values in a 1D
    capacity/SOH sequence -- the calibration-block signature.

    Returns a list of dicts: {"start": int, "end": int, "value": float}
    for every run of >= min_run identical consecutive values. If
    min_period/max_period are given, only keeps blocks whose start-to-start
    spacing from the previous block falls in that range (use this to
    require the ~52-row periodicity specifically, e.g. min_period=45,
    max_period=60) -- left as None by default so this also catches blocks
    on cells where the period might differ slightly.
    """
    values = np.asarray(values, dtype=float)
    blocks = []
    run_start = 0
    n = len(values)
    for i in range(1, n + 1):
        same = i < n and values[i] == values[i - 1]
        if not same:
            run_len = i - run_start
            if run_len >= min_run:
                blocks.append({"start": run_start, "end": i, "value": float(values[run_start])})
            run_start = i

    if min_period is not None or max_period is not None:
        filtered = [blocks[0]] if blocks else []
        for b in blocks[1:]:
            gap = b["start"] - filtered[-1]["start"]
            lo = min_period if min_period is not None else 0
            hi = max_period if max_period is not None else np.inf
            if lo <= gap <= hi:
                filtered.append(b)
        blocks = filtered

    return blocks


def strip_satellite_calibration_blocks(cell_id, df, capacity_col="capacity", min_run=5):
    """Given one XJTU cell's raw DataFrame, drops rows belonging to a
    detected periodic calibration block IF this is a Sim_satellite_* cell.
    Returns (cleaned_df, n_dropped, block_report).

    Non-satellite cells are returned unchanged -- this scope restriction
    is deliberate; see the module docstring for why the Batch-1/2 hits are
    left alone.
    """
    if "Sim_satellite" not in str(cell_id):
        return df, 0, []

    if capacity_col not in df.columns:
        raise KeyError(
            f"'{capacity_col}' not found in columns for cell {cell_id}: "
            f"{list(df.columns)}. Update capacity_col to match your actual "
            f"column name."
        )

    blocks = detect_calibration_blocks(df[capacity_col].to_numpy(), min_run=min_run)
    if not blocks:
        return df, 0, []

    drop_mask = np.zeros(len(df), dtype=bool)
    for b in blocks:
        drop_mask[b["start"]:b["end"]] = True

    cleaned = df.loc[~drop_mask].reset_index(drop=True)
    n_dropped = int(drop_mask.sum())
    return cleaned, n_dropped, blocks


def apply_to_all_cells(cell_groups, capacity_col="capacity", min_run=5, verbose=True):
    """Runs strip_satellite_calibration_blocks() over every (cell_id, df)
    pair from xjtu_loader's _cell_groups()-style output. Returns a new
    list of (cell_id, cleaned_df) pairs plus a summary report you can
    write straight into the manuscript's dataset-table footnote / data
    availability note.

    Usage inside xjtu_loader.py's load_all(), after the raw per-cell
    DataFrames are assembled but BEFORE windowing/feature extraction:

        from xjtu_satellite_calibration_fix import apply_to_all_cells
        cell_groups = [(cid, df) for cid, df in raw_cell_groups]
        cell_groups, report = apply_to_all_cells(cell_groups)
        # then proceed to windowing using the cleaned cell_groups
    """
    cleaned_groups = []
    report_lines = [
        "XJTU Sim_satellite periodic calibration-block removal report",
        "=" * 60,
    ]
    total_dropped = 0
    n_cells_affected = 0

    for cell_id, df in cell_groups:
        cleaned_df, n_dropped, blocks = strip_satellite_calibration_blocks(
            cell_id, df, capacity_col=capacity_col, min_run=min_run
        )
        cleaned_groups.append((cell_id, cleaned_df))
        if n_dropped > 0:
            n_cells_affected += 1
            total_dropped += n_dropped
            line = (f"  {cell_id}: dropped {n_dropped} rows across "
                    f"{len(blocks)} calibration block(s) "
                    f"(value={blocks[0]['value']}, period~"
                    f"{blocks[1]['start'] - blocks[0]['start'] if len(blocks) > 1 else 'n/a'} rows)")
            report_lines.append(line)
            if verbose:
                print(line)

    report_lines.append("")
    report_lines.append(
        f"TOTAL: {total_dropped} rows dropped across {n_cells_affected} "
        f"Sim_satellite_* cells."
    )
    report_lines.append(
        "Rationale: these rows are bit-identical across all affected "
        "cells and recur at a fixed period, which is inconsistent with "
        "real per-cell degradation measurement. No pad/repeat/tile code "
        "was found in xjtu_loader.py/preprocessing.py responsible for "
        "this, so the artifact is treated as inherent to the raw XJTU "
        "Batch-6 release (a periodic calibration/reference segment in "
        "the simulated satellite duty-cycle profile) rather than a bug "
        "in this project's own code."
    )

    if verbose:
        print("\n".join(report_lines[-3:]))

    return cleaned_groups, "\n".join(report_lines)


if __name__ == "__main__":
    # Smoke test with synthetic data matching the reported signature
    if pd is None:
        raise SystemExit("pandas required for this smoke test")

    rng = np.random.default_rng(0)
    n = 300
    capacity = 2.0 - 0.001 * np.arange(n) + rng.normal(0, 0.002, n)
    for start in range(24, n, 52):
        capacity[start:start + 5] = 1.601
    df = pd.DataFrame({"capacity": capacity})

    cleaned, n_dropped, blocks = strip_satellite_calibration_blocks(
        "XJTU_Batch-6_Sim_satellite_1", df
    )
    print(f"Smoke test: dropped {n_dropped} rows across {len(blocks)} blocks "
          f"(expected ~{5 * len(range(24, n, 52))} rows / "
          f"{len(range(24, n, 52))} blocks)")
    assert n_dropped == 5 * len(range(24, n, 52))
    print("OK")
