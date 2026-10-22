"""
How many calibration cells does groupwise calibration need?

For a fixed test set and a fixed pool of validation cells, each dataset is calibrated on random
subsets of b of its validation cells (cells, not windows, are drawn without replacement, because
windows from one cell are correlated), for every b below the number available, and the resulting
coverage on the unchanged test set is recorded. Repeating the draw many times at each b gives the
distribution of coverage with b calibration cells. Oxford and NASA have one validation cell each and
are left out. No model is retrained.

Requires: outputs/conformal_groupwise_predictions_val.npz,
outputs/conformal_groupwise_predictions.npz, cache/processed.npz, config.py.
Run:  python calibration_size_sensitivity.py
Out:  outputs/calibration_size_sensitivity.json
"""

import json
import os

import numpy as np

import config

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
SEED = 0
N_REPEATS = 500
BUDGET_GRID = [1, 2, 3, 5, 7, 10, 15, 20]
ALL = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]


def fsq(r, alpha=ALPHA):
    n = len(r)
    return float(np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))


def budget_sweep(cache_path, val_path, test_path, datasets=ALL):
    """The whole sweep on one set of files; main() runs it on the main split, field_check.py on the
    field data."""
    rng = np.random.default_rng(SEED)
    cache = np.load(cache_path, allow_pickle=True)
    val = np.load(val_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    std = cache["soh_std"].item()

    out = {"alpha": ALPHA, "n_repeats": N_REPEATS, "budget_grid": BUDGET_GRID, "per_dataset": {}}

    print("Calibration-cell-budget sensitivity (groupwise conformal), 90% nominal target")
    print(f"{'Dataset':8s}{'n_cells_avail':>14s}  budgets swept")
    for d in datasets:
        mv = val["lab_val"] == d
        rv = np.abs(val["y_val"][mv] - val["mean_pred"][mv]) * std[d]
        cell_v = val["cell_val"][mv]
        cells_avail = np.unique(cell_v)
        n_avail = len(cells_avail)

        mt = test["lab_test"] == d
        rt = np.abs(test["y_test"][mt] - test["mean_pred"][mt]) * std[d]
        cov = lambda h: float(np.mean(rt <= h))

        idx_by_cell = {c: np.where(cell_v == c)[0] for c in cells_avail}
        budgets = [b for b in BUDGET_GRID if b < n_avail]
        print(f"{d:8s}{n_avail:14d}  {budgets + ([n_avail] if n_avail not in budgets else [])}")

        per_budget = {}
        for b in budgets:
            covs, qs = np.empty(N_REPEATS), np.empty(N_REPEATS)
            for i in range(N_REPEATS):
                pick = rng.choice(cells_avail, size=b, replace=False)
                r_sub = rv[np.concatenate([idx_by_cell[c] for c in pick])]
                q = fsq(r_sub)
                qs[i] = q
                covs[i] = cov(q)
            per_budget[str(b)] = {
                "coverage_mean": float(covs.mean()), "coverage_std": float(covs.std()),
                "coverage_p05": float(np.percentile(covs, 5)), "coverage_p95": float(np.percentile(covs, 95)),
                "q_mean": float(qs.mean()), "q_std": float(qs.std()),
            }

        # Full-budget (all available cells) -- deterministic, no repeats needed.
        q_full = fsq(rv)
        per_budget[str(n_avail)] = {
            "coverage_mean": cov(q_full), "coverage_std": 0.0,
            "coverage_p05": cov(q_full), "coverage_p95": cov(q_full),
            "q_mean": q_full, "q_std": 0.0,
        }

        out["per_dataset"][d] = {"n_cells_available": n_avail, "n_test": int(len(rt)), "per_budget": per_budget}

    print("\nPer-dataset, per-budget mean coverage (%):")
    for d in datasets:
        pb = out["per_dataset"][d]["per_budget"]
        row = f"{d:8s}"
        for b in sorted(pb, key=int):
            row += f"  b={b:<3s}{100*pb[b]['coverage_mean']:5.1f}%"
        print(row)
    return out


def main():
    out = budget_sweep(os.path.join(config.CACHE_DIR, "processed.npz"),
                       os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"),
                       os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"))
    path = os.path.join(config.OUTPUT_DIR, "calibration_size_sensitivity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
