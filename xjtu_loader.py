"""
XJTU (Xi'an Jiaotong University) battery dataset loader.

Folder structure: Battery Dataset/Batch-N/<protocol>_battery-M.mat
  Batch-1: 2C_battery-1..8.mat            (8 cells, 2C constant discharge)
  Batch-2: 3C_battery-1..15.mat           (15 cells, 3C constant discharge)
  Batch-3: R2.5_battery-1..8.mat          (8 cells, randomised ~2.5C profile)
  Batch-4: R3_battery-1..8.mat            (8 cells, randomised ~3C profile)
  Batch-5: RW_battery-1..8.mat            (8 cells, real-world profile)
  Batch-6: Sim_satellite_battery-1..8.mat (8 cells, simulated satellite profile)

Each per-cell .mat file has two top-level keys:
  'data'    -- per-cycle raw voltage/current/temperature traces (not used)
  'summary' -- per-cycle summary arrays, including 'discharge_capacity_Ah'
               (index i = cycle i+1), which is the capacity-fade curve used here.

Sentinel-value correction (Batch-6 Sim_satellite_* cells only): these cells
contain periodic calibration/reference values that are not degradation
measurements. They occur as flat runs (e.g. value 1.601), as shorter fragments
of the same value, and as isolated non-consecutive single rows (e.g. 0.111 and
1.579) recurring at a regular spacing. detect_and_strip_sentinel_values()
removes any capacity value that occurs at least min_occurrences times anywhere
in a cell's sequence, which catches all three forms without assuming a period.
Genuine measurements essentially never repeat the exact same floating-point
value that often; the identified sentinel values each recur about 18 times.

Sensitivity toggle: set the environment variable XJTU_APPLY_SENTINEL_FIX to
"0" (or "false") to disable the correction for a run, e.g.

    # PowerShell:
    $env:XJTU_APPLY_SENTINEL_FIX = "0"
    python preprocessing.py
    python experiments.py
    Remove-Item Env:XJTU_APPLY_SENTINEL_FIX

    # bash:
    XJTU_APPLY_SENTINEL_FIX=0 python preprocessing.py

Unset (the default), the correction is applied. No other correction is
affected. run_xjtu_sensitivity_experiment.py automates the full comparison.
"""

import os
import re
from collections import Counter

import numpy as np
import scipy.io as sio
import pandas as pd
import config

APPLY_SENTINEL_FIX = os.environ.get("XJTU_APPLY_SENTINEL_FIX", "1").lower() not in ("0", "false")


def detect_and_strip_sentinel_values(df: pd.DataFrame, cell_id: str,
                                      min_occurrences: int = 4) -> pd.DataFrame:
    """Drops rows whose capacity value recurs >= min_occurrences times
    ANYWHERE in this cell's sequence (not just consecutively) -- catches
    flat calibration runs of any length AND non-consecutive periodic
    sentinel values in one pass. Sim_satellite_* cells only; no-op for
    every other cell.

    Honors the XJTU_APPLY_SENTINEL_FIX environment variable (see module
    docstring) -- if set to "0"/"false", this is a no-op for ALL cells,
    including Sim_satellite_* ones, enabling a clean before/after
    sensitivity comparison."""
    if not APPLY_SENTINEL_FIX:
        return df
    if "Sim_satellite" not in cell_id:
        return df

    capacity = df["capacity"].to_numpy()
    counts = Counter(capacity.tolist())
    suspect_values = {v for v, c in counts.items() if c >= min_occurrences}

    if not suspect_values:
        return df

    drop_mask = np.isin(capacity, list(suspect_values))
    n_dropped = int(drop_mask.sum())
    if n_dropped > 0:
        value_summary = ", ".join(
            f"{v:.4g} (x{counts[v]})" for v in sorted(suspect_values)
        )
        print(f"  [XJTU] {cell_id}: removed {n_dropped} sentinel/calibration "
              f"rows -- values [{value_summary}] each recur >= "
              f"{min_occurrences} times and are not real per-cycle "
              f"degradation readings")
        df = df.loc[~drop_mask].reset_index(drop=True)
    return df


def load_xjtu_cell(mat_path: str, cell_id: str, protocol: str) -> pd.DataFrame:
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "summary" not in mat:
        raise KeyError(f"'summary' key not found in {mat_path}; "
                        f"top-level keys were: {list(mat.keys())}")

    summary = mat["summary"]
    if not hasattr(summary, "discharge_capacity_Ah"):
        raise KeyError(f"'discharge_capacity_Ah' field not found under 'summary' "
                        f"in {mat_path}; available fields: "
                        f"{[f for f in dir(summary) if not f.startswith('_')]}")

    capacity = np.atleast_1d(summary.discharge_capacity_Ah).astype(float)
    cycles = np.arange(1, len(capacity) + 1)

    df = pd.DataFrame({"cycle": cycles, "capacity": capacity})
    df["cell_id"] = cell_id
    df["dataset"] = "XJTU"
    df["protocol"] = protocol

    df = detect_and_strip_sentinel_values(df, cell_id)

    return df


def load_all() -> pd.DataFrame:
    if not os.path.isdir(config.XJTU_DIR):
        raise FileNotFoundError(f"XJTU_DIR not found: {config.XJTU_DIR}")

    mode = "APPLIED" if APPLY_SENTINEL_FIX else "DISABLED (sensitivity-analysis mode)"
    print(f"  [XJTU] sentinel-value fix: {mode}")

    frames = []
    batch_dirs = sorted(
        d for d in os.listdir(config.XJTU_DIR)
        if os.path.isdir(os.path.join(config.XJTU_DIR, d)) and d.lower().startswith("batch")
    )
    for batch in batch_dirs:
        batch_dir = os.path.join(config.XJTU_DIR, batch)
        mat_files = sorted(f for f in os.listdir(batch_dir) if f.endswith(".mat"))
        print(f"  [XJTU] {batch}: found {len(mat_files)} cell files")
        for fname in mat_files:
            # filename like "2C_battery-1.mat" or "Sim_satellite_battery-3.mat"
            m = re.match(r"(.+)_battery-(\d+)\.mat$", fname, re.IGNORECASE)
            protocol = m.group(1) if m else "unknown"
            battery_num = m.group(2) if m else fname
            cell_id = f"XJTU_{batch}_{protocol}_{battery_num}"
            path = os.path.join(batch_dir, fname)
            try:
                df = load_xjtu_cell(path, cell_id, protocol)
                if len(df) > 5:
                    frames.append(df)
            except Exception as e:
                print(f"  [XJTU] skipped {fname}: {e}")

    if not frames:
        raise RuntimeError("XJTU: no cells parsed — check XJTU_DIR and .mat structure")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())