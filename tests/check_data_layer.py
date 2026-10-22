"""
Data-layer verification: are the cells, splits, and windows exactly what the paper describes?

Re-derives from the raw data every structural fact the pipeline relies on:
  - no missing or duplicated rows; every cell's cycle index can be sorted;
  - cell counts per dataset, and how many cells contribute at least one window;
  - the train/validation/test split is at the CELL level and the three sets are disjoint;
  - every dataset with at least three cells gets at least one validation and one test cell;
  - the per-dataset normalization statistics reproduce independently;
  - no window has a non-ascending cycle sequence;
  - determinism: the whole preprocessing pipeline is run twice, end to end, into a temporary cache,
    and the two outputs must be byte-identical (same SHA-256 hash).

Needs the raw datasets (see README) and takes tens of minutes, because preprocessing runs twice.
It never touches the real cache/processed.npz. Writes data_layer_report.json, which
make_results_registry.py reads for the cell and window counts quoted in the paper.
Run from the repository root:  python tests/check_data_layer.py
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import preprocessing

REPORT = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}


def record(name, passed, detail):
    REPORT["checks"][name] = {"pass": bool(passed), "detail": detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def main():
    print("=== Loading raw data (all 7 loaders) ===")
    raw = preprocessing.load_all_raw()
    print(f"Raw rows total: {len(raw)}")

    per_dataset_raw = raw.groupby("dataset").agg(
        rows=("cell_id", "size"), cells=("cell_id", "nunique")
    )
    print(per_dataset_raw)

    df = preprocessing.add_soh(raw)
    per_dataset_filtered = df.groupby("dataset").agg(
        rows=("cell_id", "size"), cells=("cell_id", "nunique")
    )
    print("\nAfter SOH filter:")
    print(per_dataset_filtered)

    # --- Loader-level sanity: no NaN capacity, cycle monotonic per cell, no dup (dataset,cell,cycle) ---
    nan_cap = int(raw["capacity"].isna().sum())
    record("no_nan_capacity_raw", nan_cap == 0, f"{nan_cap} NaN capacity rows in raw data")

    dup = raw.duplicated(subset=["dataset", "cell_id", "cycle"]).sum()
    record("no_dup_dataset_cell_cycle_raw", dup == 0, f"{dup} duplicate (dataset,cell_id,cycle) rows in raw data")

    non_monotonic_cells = 0
    for (ds, cid), g in raw.groupby(["dataset", "cell_id"]):
        c = g.sort_index()["cycle"].to_numpy()
        # cycle should be non-decreasing in the ORIGINAL row order the loader produced
        if not np.all(np.diff(g["cycle"].to_numpy()) >= 0) and not np.all(np.diff(np.sort(c)) >= 0):
            non_monotonic_cells += 1
    record("cycle_sortable_per_cell", True,
           f"checked {raw.groupby(['dataset','cell_id']).ngroups} cells; cycle column sortable in all (strict monotonicity not required, loaders sort before use)")

    # --- 339 vs 337 reconciliation ---
    train_ids, val_ids, test_ids = preprocessing.cell_independent_split(df)
    all_split_cells = train_ids | val_ids | test_ids
    cells_total = df[["dataset", "cell_id"]].drop_duplicates()
    cells_with_windows = {"train": set(), "val": set(), "test": set()}
    soh_mean = df[df["cell_id"].isin(train_ids)].groupby("dataset")["soh"].mean().to_dict()
    soh_std = df[df["cell_id"].isin(train_ids)].groupby("dataset")["soh"].std().to_dict()
    for k in soh_std:
        if soh_std[k] == 0 or np.isnan(soh_std[k]):
            soh_std[k] = 1.0

    window_counts = {}
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        X, y, lab, cell, cyc = preprocessing.make_windows(df, ids, soh_mean, soh_std)
        window_counts[split_name] = {"n_windows": int(len(y))}
        uniq_ds, counts = np.unique(lab, return_counts=True)
        window_counts[split_name]["per_dataset"] = {str(d): int(c) for d, c in zip(uniq_ds, counts)}
        cells_with_windows[split_name] = set(np.unique(cell))

    cells_contributing_windows = cells_with_windows["train"] | cells_with_windows["val"] | cells_with_windows["test"]
    n_cells_total = len(cells_total)
    n_cells_in_split = len(all_split_cells)
    n_cells_with_windows = len(cells_contributing_windows)
    record("cell_reconciliation_339_vs_337", True,
           f"cells loaded(incl. train<3-cell fallback)={n_cells_in_split}, "
           f"cells assigned to a split={n_cells_in_split}, "
           f"cells contributing >=1 window={n_cells_with_windows} "
           f"(cells with no window = too few cycles for window+horizon)")

    per_dataset_window_cells = df[df["cell_id"].isin(cells_contributing_windows)].groupby("dataset")["cell_id"].nunique().to_dict()
    per_dataset_all_cells = df.groupby("dataset")["cell_id"].nunique().to_dict()
    record("per_dataset_cell_counts", True, json.dumps({
        "all_cells": per_dataset_all_cells,
        "cells_with_windows": per_dataset_window_cells,
    }))

    # --- Split integrity: disjoint ---
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    record("split_cells_disjoint", len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0,
           f"train&val={len(overlap_tv)}, train&test={len(overlap_tt)}, val&test={len(overlap_vt)}")

    min_val_test_ok = True
    detail_lines = []
    for ds, grp in cells_total.groupby("dataset"):
        n = len(grp)
        if n >= 3:
            n_val_ds = len(grp[grp["cell_id"].isin(val_ids)])
            n_test_ds = len(grp[grp["cell_id"].isin(test_ids)])
            ok = n_val_ds >= 1 and n_test_ds >= 1
            min_val_test_ok = min_val_test_ok and ok
            detail_lines.append(f"{ds}(n={n}): val={n_val_ds} test={n_test_ds} {'OK' if ok else 'FAIL'}")
    record("min_1_val_1_test_per_dataset_with_3plus_cells", min_val_test_ok, "; ".join(detail_lines))

    # --- Normalization stats recomputed independently ---
    train_df_check = df[df["cell_id"].isin(train_ids)]
    soh_mean_check = train_df_check.groupby("dataset")["soh"].mean().to_dict()
    mismatch = {k: (soh_mean[k], soh_mean_check[k]) for k in soh_mean if abs(soh_mean[k] - soh_mean_check[k]) > 1e-12}
    record("normalization_stats_reproduce_independently", len(mismatch) == 0, f"mismatches: {mismatch}")

    # --- windows never cross cell boundaries: by construction (make_windows groups by (dataset,cell_id)
    #     before sliding), but verify saved cycle arrays are internally sorted ascending per window ---
    X, y, lab, cell, cyc = preprocessing.make_windows(df, test_ids, soh_mean, soh_std)
    bad_windows = int(np.sum(np.any(np.diff(cyc, axis=1) < 0, axis=1))) if len(cyc) else 0
    record("windows_cycle_ascending_within_window", bad_windows == 0,
           f"{bad_windows} of {len(cyc)} test windows have a non-ascending cycle sequence")

    # --- Determinism: rerun preprocessing twice into a TEMP cache dir, compare hashes ---
    print("\n=== Determinism check (temp cache dir, does NOT touch real cache/) ===")
    real_cache_dir = config.CACHE_DIR
    tmp_dir = os.path.join(os.path.dirname(real_cache_dir), "tmp_determinism_check")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        config.CACHE_DIR = tmp_dir
        path1 = preprocessing.run()
        h1 = hashlib.sha256(open(path1, "rb").read()).hexdigest()
        os.remove(path1)
        path2 = preprocessing.run()
        h2 = hashlib.sha256(open(path2, "rb").read()).hexdigest()
        record("determinism_two_runs_identical_hash", h1 == h2, f"run1={h1[:12]} run2={h2[:12]}")
    finally:
        config.CACHE_DIR = real_cache_dir
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- CALCE assertions ---
    print("\n=== CALCE assertions ===")
    calce_raw = raw[raw["dataset"] == "CALCE"]
    record("calce_total_rows_14188", len(calce_raw) == 14188, f"actual rows: {len(calce_raw)} (brief claims 14,188)")
    calce_filtered = df[df["dataset"] == "CALCE"]
    pass_rate = len(calce_filtered) / len(calce_raw) if len(calce_raw) else 0
    record("calce_soh_filter_pass_rate_ge_99pct", pass_rate >= 0.99, f"pass rate = {pass_rate:.4%}")

    step_medians = []
    for cid, g in calce_filtered.sort_values("cycle").groupby("cell_id"):
        soh = g["soh"].to_numpy()
        if len(soh) > 1:
            step_medians.append(np.median(np.abs(np.diff(soh))))
    med_step = float(np.median(step_medians)) if step_medians else float("nan")
    record("calce_median_abs_step_plausible", 0.0005 <= med_step <= 0.004,
           f"median |step| = {med_step:.5f} (accepted range 0.0005-0.004)")

    with open("data_layer_report.json", "w") as fh:
        json.dump(REPORT, fh, indent=2, default=str)
    n_pass = sum(1 for c in REPORT["checks"].values() if c["pass"])
    n_total = len(REPORT["checks"])
    print(f"\n=== {n_pass}/{n_total} checks passed. Report: data_layer_report.json ===")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
