"""
Real-unit conversion for the classical baselines (Table "tab:baselines").

The stored baseline result JSONs (baseline_svr_results.json,
baseline_xgboost_results.json, baseline_linear_results.json) report only
pooled and per-dataset MAE in per-dataset z-scored units; no real-unit values
or per-dataset RMSE were saved for these runs, and their raw predictions were
not saved either, so RMSE in real units cannot be recovered here.

This script converts what IS available -- per-dataset MAE -- to physical SOH
units using the per-dataset SOH standard deviation from cache/processed.npz
(per_dataset_mae[d] * soh_std[d]), then computes a pooled real-unit MAE
weighted by each dataset's TEST WINDOW COUNT from the cache.

ASSUMPTION, stated explicitly: this assumes the baselines were evaluated on
the SAME cell-level test split (seed 42) as the main model and the cached
window counts, consistent with every other cross-model comparison in this
project, but not independently re-verified here since the baselines' raw
per-window predictions were not saved.

Requires: cache/processed.npz, baseline_svr_results.json,
baseline_xgboost_results.json, baseline_linear_results.json.
Run:  python real_unit_baseline_conversion.py
Out:  outputs/real_unit_baselines.json
"""

import json
import os

import numpy as np

import config

DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
FILES = {"Linear regression": "baseline_linear_results.json",
        "SVR (RBF)": "baseline_svr_results.json",
        "XGBoost": "baseline_xgboost_results.json"}


def main():
    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    std = cache["soh_std"].item()
    lab = cache["lab_test"]
    n = {d: int(np.sum(lab == d)) for d in DATASETS}

    out = {}
    print(f"{'Model':18s}" + "".join(f"{d:>9s}" for d in DATASETS) + f"{'pooled':>9s}")
    for name, fname in FILES.items():
        path = os.path.join(config.OUTPUT_DIR, fname)
        if not os.path.exists(path):
            path = fname   # fallback: project root, in case the JSON was saved there instead
        with open(path) as fh:
            j = json.load(fh)
        pm_z = j["per_dataset_mae"]
        real = {d: pm_z[d] * std[d] for d in DATASETS}
        pooled = sum(real[d] * n[d] for d in DATASETS) / sum(n.values())
        out[name] = {"per_dataset_mae_real": real, "pooled_mae_real": pooled,
                     "pooled_mae_z_scored_as_stored": j["test_mae"]}
        print(f"{name:18s}" + "".join(f"{real[d]:9.5f}" for d in DATASETS) + f"{pooled:9.5f}")

    path = os.path.join(config.OUTPUT_DIR, "real_unit_baselines.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()