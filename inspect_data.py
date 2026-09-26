"""
Checks that every raw dataset is found and parses. Run this first.

Battery datasets are inconsistent in internal structure (field names, sheet
names, JSON keys). This script prints the actual structure of the raw files
for NASA, CALCE, Oxford and MIT, and runs the full load_all() of the BIT,
XJTU and RWTH loaders. If a printed field name does not match what a loader
expects, adjust that loader (the relevant lines are marked in each loader).

Because it runs the full BIT/XJTU/RWTH loaders (about 1,600 RWTH CSVs, 70 BIT
Excel files and 55 XJTU .mat files), this takes several minutes.

Usage:
    python inspect_data.py
"""

import os
import json
import scipy.io as sio
import pandas as pd
import config


def inspect_nasa():
    print("\n=== NASA ===")
    if not os.path.isdir(config.NASA_DIR):
        print(f"  [!] Folder not found: {config.NASA_DIR}")
        return
    files = []
    for dirpath, _, filenames in os.walk(config.NASA_DIR):
        for f in filenames:
            if f.endswith(".mat"):
                files.append(os.path.join(dirpath, f))
    print(f"  Found {len(files)} .mat files (searched recursively):",
          [os.path.basename(f) for f in files[:5]], "...")
    if not files:
        # fallback: show what's actually inside NASA_DIR, and search the
        # whole ROOT_DIR in case the folder name doesn't match what config.py assumes
        print(f"  [!] Raw contents of {config.NASA_DIR}:")
        try:
            for entry in os.listdir(config.NASA_DIR):
                print(f"      {entry}")
        except Exception as e:
            print(f"      (could not list: {e})")
        print(f"  [!] Searching entire ROOT_DIR ({config.ROOT_DIR}) for any .mat files...")
        root_mats = []
        for dirpath, _, filenames in os.walk(config.ROOT_DIR):
            for f in filenames:
                if f.endswith(".mat") and "oxford" not in f.lower():
                    root_mats.append(os.path.join(dirpath, f))
        print(f"      Found {len(root_mats)} elsewhere:", root_mats[:10])
        return
    mat = sio.loadmat(files[0])
    key = os.path.basename(files[0]).replace(".mat", "")
    if key not in mat:
        key = [k for k in mat.keys() if not k.startswith("__")][0]
    print(f"  Top-level keys: {[k for k in mat.keys() if not k.startswith('__')]}")
    try:
        cycle = mat[key][0, 0]['cycle'][0]
        print(f"  Number of cycles in {files[0]}: {len(cycle)}")
        print(f"  First cycle type: {cycle[0]['type'][0]}")
        print(f"  First cycle fields: {cycle[0]['data'][0, 0].dtype.names}")
    except Exception as e:
        print(f"  [!] Structure differs from expected — inspect manually. Error: {e}")


def inspect_calce():
    print("\n=== CALCE (CS2 / CX2) ===")
    for chem, dirs in config.CALCE_DIRS.items():
        for d in dirs[:1]:  # just check the first cell of each chemistry
            if not os.path.isdir(d):
                print(f"  [!] Folder not found: {d}")
                continue
            xlsx_files = []
            for dirpath, _, filenames in os.walk(d):
                for f in filenames:
                    if f.lower().endswith((".xlsx", ".xls")):
                        xlsx_files.append(os.path.join(dirpath, f))
            print(f"  {d}: {len(xlsx_files)} excel files (searched recursively), "
                  f"e.g. {[os.path.basename(f) for f in xlsx_files[:3]]}")
            if not xlsx_files:
                print(f"    [!] Raw contents of {d}:")
                try:
                    for entry in os.listdir(d):
                        print(f"        {entry}")
                except Exception as e:
                    print(f"        (could not list: {e})")
                continue
            if xlsx_files:
                path = xlsx_files[0]
                xls = pd.ExcelFile(path)
                print(f"    Sheets: {xls.sheet_names}")
                for sheet in xls.sheet_names:
                    if sheet.lower().startswith("channel"):
                        df = pd.read_excel(path, sheet_name=sheet, nrows=3)
                        print(f"    Columns in '{sheet}': {list(df.columns)}")
                        break


def inspect_oxford():
    print("\n=== Oxford ===")
    if not os.path.isfile(config.OXFORD_MAT):
        print(f"  [!] File not found: {config.OXFORD_MAT}")
        return
    mat = sio.loadmat(config.OXFORD_MAT, squeeze_me=True, struct_as_record=False)
    top_keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"  Top-level keys: {top_keys}")
    root_key = top_keys[0]
    root = mat[root_key]
    print(f"  Cell field names under '{root_key}': {[f for f in dir(root) if not f.startswith('_')][:20]}")


def inspect_mit():
    print("\n=== MIT / Severson (FastCharge JSON) ===")
    if not os.path.isdir(config.MIT_DIR):
        print(f"  [!] Folder not found: {config.MIT_DIR}")
        return
    files = []
    for dirpath, _, filenames in os.walk(config.MIT_DIR):
        for f in filenames:
            if f.endswith("_structure.json"):
                files.append(os.path.join(dirpath, f))
    print(f"  Found {len(files)} JSON files (searched recursively), "
          f"e.g. {[os.path.basename(f) for f in files[:3]]}")
    if not files:
        return
    # pick the first non-empty file (some of yours were 0 KB / still downloading)
    good_file = next((f for f in files if os.path.getsize(f) > 0), None)
    if good_file is None:
        print("  [!] All found JSON files are 0 KB — re-download FastCharge.zip")
        return
    with open(good_file) as fh:
        d = json.load(fh)
    print(f"  Top-level keys: {list(d.keys())}")
    if "summary" in d:
        print(f"  'summary' keys: {list(d['summary'].keys())}")


def inspect_bit():
    print("\n=== BIT (Beijing Institute of Technology) ===")
    bit_dir = getattr(config, "BIT_DIR", None)
    if not bit_dir or not os.path.isdir(bit_dir):
        print(f"  [!] BIT_DIR not found: {bit_dir}")
        return
    subfolders = ["Cycled with Arbitrary Uses Profiles", "Cycled with Fixed Current Profiles"]
    total_cells = 0
    for sf in subfolders:
        path = os.path.join(bit_dir, sf)
        if not os.path.isdir(path):
            print(f"  [!] Subfolder not found: {path}")
            continue
        cell_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(f"  {sf}: {len(cell_folders)} cell folders")
        total_cells += len(cell_folders)
    print(f"  Total cell folders found: {total_cells}")
    try:
        import bit_loader
        df = bit_loader.load_all()
        print(f"  Loader check: {df['cell_id'].nunique()} cells parsed successfully, "
              f"{len(df)} total rows")
    except Exception as e:
        print(f"  [!] bit_loader.load_all() failed: {e}")


def inspect_xjtu():
    print("\n=== XJTU (Xi'an Jiaotong University) ===")
    xjtu_dir = getattr(config, "XJTU_DIR", None)
    if not xjtu_dir or not os.path.isdir(xjtu_dir):
        print(f"  [!] XJTU_DIR not found: {xjtu_dir}")
        return
    batch_dirs = sorted(d for d in os.listdir(xjtu_dir)
                        if os.path.isdir(os.path.join(xjtu_dir, d)) and d.lower().startswith("batch"))
    print(f"  Found {len(batch_dirs)} batch folders: {batch_dirs}")
    for b in batch_dirs:
        n_mat = len([f for f in os.listdir(os.path.join(xjtu_dir, b)) if f.endswith(".mat")])
        print(f"    {b}: {n_mat} .mat files")
    try:
        import xjtu_loader
        df = xjtu_loader.load_all()
        print(f"  Loader check: {df['cell_id'].nunique()} cells parsed successfully, "
              f"{len(df)} total rows")
    except Exception as e:
        print(f"  [!] xjtu_loader.load_all() failed: {e}")


def inspect_rwth():
    print("\n=== RWTH Aachen ===")
    rwth_dir = getattr(config, "RWTH_DIR", None)
    if not rwth_dir or not os.path.isdir(rwth_dir):
        print(f"  [!] RWTH_DIR not found: {rwth_dir}")
        return
    csv_count = len([f for f in os.listdir(rwth_dir) if f.lower().endswith(".csv")])
    print(f"  Found {csv_count} CSV files (raw session files) in {rwth_dir}")
    try:
        import rwth_loader
        df = rwth_loader.load_all()
        print(f"  Loader check: {df['cell_id'].nunique()} cells parsed successfully, "
              f"{len(df)} total rows")
    except Exception as e:
        print(f"  [!] rwth_loader.load_all() failed: {e}")


if __name__ == "__main__":
    inspect_nasa()
    inspect_calce()
    inspect_oxford()
    inspect_mit()
    inspect_bit()
    inspect_xjtu()
    inspect_rwth()
    print("\nDone. Fix any parser noted with [!] before running preprocessing.py")