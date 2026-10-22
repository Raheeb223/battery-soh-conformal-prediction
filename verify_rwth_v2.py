"""
Re-checks RWTH's two data-quality corrections against the raw CSV files: dropping the partial first
cycle of each session file (drop_first) and the local-outlier filter (strip_local_outliers).

rwth_loader.load_all() is run with each of the four combinations of the two corrections, and for
each the script reports the number of rows and cells, the median absolute capacity step between
consecutive cycles (smooth degradation gives small steps; remaining artifacts inflate it), and how
many rows each correction removes.

Requires the RWTH raw CSV files under config.RWTH_DIR.
Run:  python verify_rwth_v2.py
Out:  outputs/rwth_verification_v2.json
"""
import json
import os

import numpy as np

import config
import rwth_loader


def median_abs_step(df):
    steps = []
    for _, g in df.groupby("cell_id"):
        c = g.sort_values("cycle")["capacity"].to_numpy()
        if len(c) > 1:
            steps.append(np.abs(np.diff(c)))
    return float(np.median(np.concatenate(steps))) if steps else float("nan")


def main():
    if not config.RWTH_DIR or not os.path.isdir(config.RWTH_DIR):
        print(f"  [!] RWTH_DIR not found ({config.RWTH_DIR}) -- cannot re-verify against raw files.")
        return

    configs = {
        "production (drop_first=T, outlier_fix=T)": dict(drop_first=True, apply_local_outlier_fix=True),
        "no drop_first (F, outlier_fix=T)": dict(drop_first=False, apply_local_outlier_fix=True),
        "no outlier_fix (drop_first=T, F)": dict(drop_first=True, apply_local_outlier_fix=False),
        "neither correction (F, F)": dict(drop_first=False, apply_local_outlier_fix=False),
    }

    results = {}
    for label, kwargs in configs.items():
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        df = rwth_loader.load_all(**kwargs)
        n_cells = df["cell_id"].nunique()
        n_rows = len(df)
        med_step = median_abs_step(df)
        results[label] = {"n_rows": n_rows, "n_cells": n_cells, "median_abs_step": med_step}
        print(f"  -> n_rows={n_rows}  n_cells={n_cells}  median|step|={med_step:.5f}")

    baseline = results["neither correction (F, F)"]["n_rows"]
    prod = results["production (drop_first=T, outlier_fix=T)"]["n_rows"]
    results["summary"] = {
        "rows_removed_by_drop_first_alone": baseline - results["no outlier_fix (drop_first=T, F)"]["n_rows"],
        "rows_removed_by_outlier_fix_alone": baseline - results["no drop_first (F, outlier_fix=T)"]["n_rows"],
        "rows_removed_by_both_combined": baseline - prod,
        "pct_removed_by_both_combined": 100.0 * (baseline - prod) / baseline,
    }
    print(f"\n{'='*70}\nSummary\n{'='*70}")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")

    out_path = os.path.join(config.OUTPUT_DIR, "rwth_verification_v2.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
