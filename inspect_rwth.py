"""
RUN THIS FIRST for the RWTH Aachen dataset — same discipline as every other
dataset in this project: inspect real structure before writing any loader.

Edit RWTH_DIR below to point at wherever you extracted the 5.1GB download,
then run and paste the FULL output back.
"""

import os
import json

RWTH_DIR = r"C:\Users\User\PycharmProjects\RWTH-2021-04545_818642"   # <-- EDIT THIS


def walk_summary(root_dir, max_depth=3, max_files_per_dir=10):
    print(f"\n{'='*70}\nFOLDER STRUCTURE: {root_dir}\n{'='*70}")
    if not os.path.isdir(root_dir):
        print(f"  [!] Folder not found: {root_dir}")
        return
    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/  ({len(filenames)} files, {len(dirnames)} subfolders)")
        for f in sorted(filenames)[:max_files_per_dir]:
            print(f"{indent}  {f}")
        if len(filenames) > max_files_per_dir:
            print(f"{indent}  ... and {len(filenames) - max_files_per_dir} more files")


def extension_counts(root_dir):
    counts, examples = {}, {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            counts[ext] = counts.get(ext, 0) + 1
            if ext not in examples:
                examples[ext] = os.path.join(dirpath, f)
    print(f"\n--- File extension counts ---")
    for ext, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no extension)'}: {n} files, e.g. {examples[ext]}")
    return examples


def peek_csv(path):
    print(f"\n  Peeking CSV/TXT: {path}")
    try:
        with open(path, "r", errors="replace") as fh:
            for i, line in enumerate(fh):
                print(f"    line {i}: {line.strip()[:300]}")
                if i >= 5:
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
            print(f"    List of {len(d)} items")
    except Exception as e:
        print(f"    [!] could not read: {e}")


walk_summary(RWTH_DIR)
examples = extension_counts(RWTH_DIR)
for ext, path in examples.items():
    if ext in (".csv", ".txt"):
        peek_csv(path)
    elif ext in (".xlsx", ".xls"):
        peek_excel(path)
    elif ext == ".mat":
        peek_mat(path)
    elif ext == ".json":
        peek_json(path)

print("\n\nDone. Paste this FULL output back for loader code.")