"""
RUN THIS FIRST for the two new datasets (XJTU, BIT).

Generic inspector — since we don't yet know either dataset's internal file
format, this walks the folder structure, counts files by extension, and
peeks inside one example of each format type it finds (CSV header row,
Excel sheet names/columns, .mat top-level keys, JSON top-level keys).

Edit the two paths below to point at your actual folders, then run:
    python inspect_new_datasets.py
Paste the FULL output back — that's what the real loaders get built from.
"""

import os
import json

# ---- EDIT THESE TWO PATHS to match where you put the folders ----
XJTU_DIR = r"C:\Users\User\PycharmProjects\Battery Dataset"
BIT_DIR  = r"C:\Users\User\PycharmProjects\kw34hhw7xg-3"


def walk_summary(root_dir, max_depth=3, max_files_per_dir=10):
    print(f"\n{'='*70}\nFOLDER STRUCTURE: {root_dir}\n{'='*70}")
    if not os.path.isdir(root_dir):
        print(f"  [!] Folder not found: {root_dir}")
        return

    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []  # stop descending further
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/  ({len(filenames)} files, {len(dirnames)} subfolders)")
        for f in sorted(filenames)[:max_files_per_dir]:
            print(f"{indent}  {f}")
        if len(filenames) > max_files_per_dir:
            print(f"{indent}  ... and {len(filenames) - max_files_per_dir} more files")


def extension_counts(root_dir):
    counts = {}
    examples = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            counts[ext] = counts.get(ext, 0) + 1
            if ext not in examples:
                examples[ext] = os.path.join(dirpath, f)
    print(f"\n--- File extension counts in {root_dir} ---")
    for ext, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no extension)'}: {n} files, e.g. {examples[ext]}")
    return examples


def peek_csv(path):
    print(f"\n  Peeking CSV: {path}")
    try:
        with open(path, "r", errors="replace") as fh:
            for i, line in enumerate(fh):
                print(f"    line {i}: {line.strip()[:300]}")
                if i >= 3:
                    break
    except Exception as e:
        print(f"    [!] could not read: {e}")


def peek_excel(path):
    print(f"\n  Peeking Excel: {path}")
    try:
        import pandas as pd
        xls = pd.ExcelFile(path)
        print(f"    Sheets: {xls.sheet_names}")
        for sheet in xls.sheet_names[:3]:
            df = pd.read_excel(path, sheet_name=sheet, nrows=3)
            print(f"    Columns in '{sheet}': {list(df.columns)}")
    except Exception as e:
        print(f"    [!] could not read: {e}")


def peek_mat(path):
    print(f"\n  Peeking .mat: {path}")
    try:
        import scipy.io as sio
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        top_keys = [k for k in mat.keys() if not k.startswith("__")]
        print(f"    Top-level keys: {top_keys}")
        for k in top_keys[:2]:
            val = mat[k]
            fields = [f for f in dir(val) if not f.startswith("_")]
            print(f"    Fields under '{k}': {fields[:20]}")
    except Exception as e:
        print(f"    [!] could not read: {e}")


def peek_json(path):
    print(f"\n  Peeking JSON: {path}")
    try:
        with open(path) as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            print(f"    Top-level keys: {list(d.keys())}")
        elif isinstance(d, list):
            print(f"    Top-level: list of {len(d)} items; first item keys: "
                  f"{list(d[0].keys()) if d and isinstance(d[0], dict) else type(d[0])}")
    except Exception as e:
        print(f"    [!] could not read: {e}")


def inspect_all(root_dir, label):
    print(f"\n\n{'#'*70}\n# {label}\n{'#'*70}")
    walk_summary(root_dir)
    examples = extension_counts(root_dir)

    for ext, path in examples.items():
        if ext in (".csv", ".txt"):
            peek_csv(path)
        elif ext in (".xlsx", ".xls"):
            peek_excel(path)
        elif ext == ".mat":
            peek_mat(path)
        elif ext == ".json":
            peek_json(path)


if __name__ == "__main__":
    inspect_all(XJTU_DIR, "XJTU DATASET")
    inspect_all(BIT_DIR, "BIT DATASET")
    print("\n\nDone. Paste this FULL output back for loader code.")
