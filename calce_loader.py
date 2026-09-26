"""
CALCE CS2 / CX2 loader.

Each cell folder (e.g. CS2_33/) contains one Arbin-exported .xlsx file per
test session, each with a "Channel_x-yyy" sheet holding per-row measurements
including Cycle_Index and Discharge_Capacity(Ah). The maximum discharge
capacity recorded per Cycle_Index (the point right before the cell switches
back to charging) is taken across all files for that cell, sorted
chronologically.

If the column names differ, inspect_data.py prints the actual sheet names and
columns; adjust the CANDIDATE_* lists below to match.
"""

import os
import pandas as pd
import config

CANDIDATE_CYCLE_COLS = ["Cycle_Index", "Cycle_Index(#)", "Cycle Index"]
CANDIDATE_CAP_COLS = ["Discharge_Capacity(Ah)", "Discharge_Capacity", "Discharge Capacity (Ah)"]


def _find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_calce_cell(cell_dir: str, cell_id: str, chemistry: str) -> pd.DataFrame:
    # some extracted zips put the .xlsx files directly in cell_dir, others nest
    # them one folder deeper (e.g. CS2_8/CS2_8/*.xlsx) — search recursively so
    # both layouts work without editing config.py per-cell.
    file_paths = []
    for dirpath, _, filenames in os.walk(cell_dir):
        for f in filenames:
            if f.lower().endswith((".xlsx", ".xls")):
                file_paths.append(os.path.join(dirpath, f))
    file_paths.sort(key=os.path.basename)

    if not file_paths:
        raise FileNotFoundError(f"No excel files found under {cell_dir} (searched recursively)")

    all_rows = []
    for path in file_paths:
        fname = os.path.basename(path)
        try:
            xls = pd.ExcelFile(path)
        except Exception as e:
            print(f"  [CALCE] could not open {path}: {e}")
            continue
        for sheet in xls.sheet_names:
            if "channel" not in sheet.lower():
                continue
            df = pd.read_excel(path, sheet_name=sheet)
            cyc_col = _find_col(df.columns, CANDIDATE_CYCLE_COLS)
            cap_col = _find_col(df.columns, CANDIDATE_CAP_COLS)
            if cyc_col is None or cap_col is None:
                continue
            sub = df[[cyc_col, cap_col]].dropna()
            sub.columns = ["cycle_raw", "capacity"]
            sub["source_file"] = fname
            all_rows.append(sub)

    if not all_rows:
        raise RuntimeError(f"Parsed nothing for {cell_dir} — check column names via inspect_data.py")

    combined = pd.concat(all_rows, ignore_index=True)
    # per-file cycle numbers restart at each test session; renumber globally
    # by taking the max capacity per (source_file, cycle_raw) then concatenating
    # sessions in file order (files sorted by name === chronological for CALCE).
    per_cycle = (
        combined.groupby(["source_file", "cycle_raw"], sort=False)["capacity"]
        .max()
        .reset_index()
    )
    per_cycle["cycle"] = range(1, len(per_cycle) + 1)
    per_cycle["cell_id"] = cell_id
    # Grouped as one "CALCE" dataset (not split CS2 vs CX2) so the
    # cell-independent split/normalization has 10 cells to work with
    # instead of 4 and 6 separately — small groups made test-set MAE
    # and conformal calibration unstable for each subgroup on its own.
    # Chemistry is kept as a separate column for chemistry-level breakdowns.
    per_cycle["dataset"] = "CALCE"
    per_cycle["chemistry"] = chemistry
    return per_cycle[["cycle", "capacity", "cell_id", "dataset", "chemistry"]]


def load_all() -> pd.DataFrame:
    frames = []
    for chemistry, dirs in config.CALCE_DIRS.items():
        for d in dirs:
            cell_id = os.path.basename(d.rstrip("/\\"))
            if not os.path.isdir(d):
                print(f"  [CALCE] folder not found, skipping: {d}")
                continue
            try:
                df = load_calce_cell(d, cell_id, chemistry)
                frames.append(df)
            except Exception as e:
                print(f"  [CALCE] skipped {cell_id}: {e}")
    if not frames:
        raise RuntimeError("CALCE: no cells parsed — run inspect_data.py and fix calce_loader.py")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())