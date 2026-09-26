"""
Beijing Institute of Technology (BIT) dataset loader (Mendeley Data kw34hhw7xg).

Folder structure: kw34hhw7xg-3/
  Cycled with Arbitrary Uses Profiles/#N/
    LR1865SZ_cyclesYYMMDD_XXX_Y.xlsx        <- full cycle-life data (used)
    LR1865SZ_first20cycleYYMMDD_XXX_Y.xlsx  <- first 20 cycles only; skipped
                                               because it duplicates the start
                                               of the main file
  Cycled with Fixed Current Profiles/#N/    (same file pattern)

Each main "*_cycles*.xlsx" file has one sheet ('记录表', "record sheet") with
columns including Cycle_Index and Capacity(Ah). There are no separate
charge/discharge capacity columns, so the maximum Capacity(Ah) recorded per
Cycle_Index is taken as that cycle's capacity (as for CALCE, capacity
accumulates through a cycle and peaks near end-of-discharge).

Cell model: LR1865SZ, 18650 form factor.
"""

import os
import re
import pandas as pd
import config

BIT_ROOT = getattr(config, "BIT_DIR", None)


def _find_main_cycle_file(cell_dir: str):
    """Pick the '*_cycles*.xlsx' file, explicitly skipping '*_first20cycle*'
    files which are a redundant subset."""
    candidates = [
        f for f in os.listdir(cell_dir)
        if f.lower().endswith(".xlsx") and "first20cycle" not in f.lower()
    ]
    return os.path.join(cell_dir, candidates[0]) if candidates else None


def load_bit_cell(cell_dir: str, cell_id: str, protocol_group: str) -> pd.DataFrame:
    path = _find_main_cycle_file(cell_dir)
    if path is None:
        raise FileNotFoundError(f"No main '*_cycles*.xlsx' file found in {cell_dir}")

    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]  # single sheet, name is Chinese ('记录表')
    df = pd.read_excel(path, sheet_name=sheet)

    if "Cycle_Index" not in df.columns or "Capacity(Ah)" not in df.columns:
        raise KeyError(f"Expected columns not found in {path}; "
                        f"got columns: {list(df.columns)}")

    per_cycle = (
        df.groupby("Cycle_Index")["Capacity(Ah)"]
        .max()
        .reset_index()
        .rename(columns={"Cycle_Index": "cycle", "Capacity(Ah)": "capacity"})
    )
    per_cycle["cell_id"] = cell_id
    per_cycle["dataset"] = "BIT"
    per_cycle["protocol_group"] = protocol_group  # 'arbitrary' or 'fixed'
    return per_cycle[["cycle", "capacity", "cell_id", "dataset", "protocol_group"]]


def load_all() -> pd.DataFrame:
    if not BIT_ROOT or not os.path.isdir(BIT_ROOT):
        raise FileNotFoundError(
            f"BIT_DIR not set or not found: {BIT_ROOT}. Add BIT_DIR to config.py."
        )

    subfolders = {
        "arbitrary": "Cycled with Arbitrary Uses Profiles",
        "fixed": "Cycled with Fixed Current Profiles",
    }

    frames = []
    total_cells = 0
    for group_name, subfolder in subfolders.items():
        group_dir = os.path.join(BIT_ROOT, subfolder)
        if not os.path.isdir(group_dir):
            print(f"  [BIT] subfolder not found, skipping: {group_dir}")
            continue
        cell_folders = sorted(f for f in os.listdir(group_dir)
                              if os.path.isdir(os.path.join(group_dir, f)))
        print(f"  [BIT] {group_name}: found {len(cell_folders)} cell folders, starting parse...")
        for i, cell_folder in enumerate(cell_folders, 1):
            cell_dir = os.path.join(group_dir, cell_folder)
            cell_id = f"BIT_{cell_folder.strip('#')}_{group_name}"
            print(f"  [BIT] ({i}/{len(cell_folders)}) parsing {cell_folder} ...", flush=True)
            try:
                df = load_bit_cell(cell_dir, cell_id, group_name)
                if len(df) > 5:
                    frames.append(df)
                    total_cells += 1
            except Exception as e:
                print(f"  [BIT] skipped {cell_folder}: {e}")
    print(f"  [BIT] done — {total_cells} cells parsed successfully")

    if not frames:
        raise RuntimeError("BIT: no cells parsed — check BIT_DIR and file structure")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all()
    print(df.groupby("cell_id").size())
    print(df.head())