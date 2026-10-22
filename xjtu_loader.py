"""
XJTU (Xi'an Jiaotong University) battery dataset loader.

Folder structure: Battery Dataset/Batch-N/<protocol>_battery-M.mat
  Batch-1: 2C_battery-1..8.mat            (8 cells, 2C constant discharge)
  Batch-2: 3C_battery-1..15.mat           (15 cells, 3C constant discharge)
  Batch-3: R2.5_battery-1..8.mat          (8 cells, randomized ~2.5C profile)
  Batch-4: R3_battery-1..8.mat            (8 cells, randomized ~3C profile)
  Batch-5: RW_battery-1..8.mat            (8 cells, real-world profile)
  Batch-6: Sim_satellite_battery-1..8.mat (8 cells, simulated satellite profile)

Each per-cell .mat file has two top-level keys:
  'data'    — per-cycle raw voltage/current/temperature traces (not used here)
  'summary' — a struct with per-cycle SUMMARY arrays already computed,
              including 'discharge_capacity_Ah' (index i = cycle i+1) —
              this is exactly the capacity-fade curve we need, no
              reconstruction from raw traces required.

Chemistry: NCM (per the original dataset description; see the "Data
Introduction" PDF in the download for the exact composition).

--- SENTINEL VALUES ---
Batch-6 Sim_satellite_* cells contain periodic calibration/reference
sentinel values mixed into the degradation curve, in three forms: long
flat runs of the value 1.601, shorter fragments of the same value, and
single-row values (0.111 and 1.579) that recur about every 41 rows and
are never consecutive, so a consecutive-run detector cannot catch them.

detect_and_strip_sentinel_values() therefore treats any capacity value
that occurs >= min_occurrences times anywhere in a cell's sequence as a
sentinel and removes it, which catches all three forms without assuming
a period. Real degradation
measurements essentially never hit the exact same floating-point value
4+ times independently across hundreds of cycles, so this threshold has
wide safety margin (both confirmed real sentinel values recurred ~18
times each -- 4.5x the threshold).

Applied to Sim_satellite_* cells only.

--- SENSITIVITY-ANALYSIS TOGGLE ---
To let the correction be switched off for a clean before/after sensitivity
comparison without editing this file, this module honors the environment
variable XJTU_APPLY_SENTINEL_FIX. Set it to "0" (or "false") to disable
detect_and_strip_sentinel_values() entirely for this run, e.g.:

    # PowerShell:
    $env:XJTU_APPLY_SENTINEL_FIX = "0"
    python preprocessing.py
    python experiments.py
    Remove-Item Env:XJTU_APPLY_SENTINEL_FIX   # unset when done

    # cmd.exe:
    set XJTU_APPLY_SENTINEL_FIX=0
    python preprocessing.py && python experiments.py
    set XJTU_APPLY_SENTINEL_FIX=

Leaving the variable unset (the default) applies the correction. This
does not touch any other correction (RWTH, LODO normalization) -- only
the XJTU sentinel-value strip is toggled, giving an isolated ablation.
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