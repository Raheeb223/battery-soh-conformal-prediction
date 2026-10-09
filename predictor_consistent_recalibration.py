"""
Predictor-consistent recalibration of pooled and groupwise conformal
intervals.

WHY THIS EXISTS. In train.py, conformal quantiles are calibrated on
DETERMINISTIC validation predictions (model.eval(), predict_in_batches), but
the reported test intervals are centred on the MC-Dropout MEAN prediction
(mc_dropout_predict). Calibration scores and test scores therefore come from
different predictors, which departs from the split-conformal assumption that
calibration and test nonconformity scores are exchangeable. Here both use the
MC-Dropout-mean predictor: conformal_groupwise_predictions_val.npz (validation,
MC-Dropout mean) is used to calibrate, and conformal_groupwise_predictions.npz
(test, MC-Dropout mean) is used to evaluate.

"POOLED" HAS TWO MEANINGS AND BOTH ARE REPORTED. The pipeline z-scores SOH per
dataset (training-cell mean/std), so a pooled quantile computed on z-scored
residuals is a single number in per-dataset standard-deviation units; the same
z-quantile corresponds to a different physical half-width in each dataset.
"Pooled (real units)" pools residuals in physical SOH units instead, so one
physical half-width applies to every dataset. Results can differ substantially.

Nothing here retrains a model. Inputs: cache/processed.npz (normalization
statistics), outputs/conformal_groupwise_predictions_val.npz,
outputs/conformal_groupwise_predictions.npz.

Run from the project root:
    python predictor_consistent_recalibration.py
Output:
    outputs/predictor_consistent_recalibration.json
"""

import json
import os

import numpy as np
from scipy.stats import beta

import config
from cell_clustered_bootstrap import cell_clustered_bootstrap_coverage

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]


def finite_sample_quantile(residuals, alpha=ALPHA):
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level))


def clopper_pearson(k, n, conf=0.95):
    """WINDOW-LEVEL binomial CI: treats each of the n test WINDOWS as an
    independent Bernoulli trial. This is WRONG whenever windows come from a
    small number of distinct cells (the brief's own rule: "windows from the
    same cell are not independent") -- for CALCE (2 test cells), Oxford (1),
    NASA (1), this CI is overconfident, sometimes drastically (a dataset with
    1 test cell has an effective cell-level sample size of 1, not n_windows).
    Reported here ONLY for reference/comparison against the cell-clustered CI
    computed alongside it below -- do NOT use this one alone as the reported
    per-dataset coverage uncertainty in the manuscript; use cell_bootstrap_95ci."""
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    val = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
    test = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    std = cache["soh_std"].item()

    r_val = np.abs(val["y_val"] - val["mean_pred"])
    lab_val = val["lab_val"]
    r_test = np.abs(test["y_test"] - test["mean_pred"])
    lab_test = test["lab_test"]

    r_val_real = np.array([r_val[i] * std[lab_val[i]] for i in range(len(r_val))])
    q_pooled_z = finite_sample_quantile(r_val)
    q_pooled_real = finite_sample_quantile(r_val_real)
    cell_test = test["cell_test"] if "cell_test" in test.files else None

    out = {"alpha": ALPHA, "q_pooled_z": q_pooled_z, "q_pooled_real": q_pooled_real, "per_dataset": {}}
    print(f"pooled quantile: z-units = {q_pooled_z:.4f}, real SOH units = {q_pooled_real:.5f}\n")
    print(f"{'Dataset':8s}{'n_test':>8s}{'groupwise':>11s}{'window CP CI':>20s}{'cell-boot CI':>20s}{'pooled(z)':>11s}{'pooled(real)':>14s}")
    for d in DATASETS:
        mt, mv = lab_test == d, lab_val == d
        n = int(mt.sum())
        q_g = finite_sample_quantile(r_val[mv])
        covered_g = r_test[mt] <= q_g
        cov_g = float(np.mean(covered_g))
        k = int(round(cov_g * n))
        lo, hi = clopper_pearson(k, n)

        cell_ci = None
        if cell_test is not None:
            cells_d = cell_test[mt]
            n_cells = len(np.unique(cells_d))
            if n_cells >= 2:
                # reuse the already-verified cell-clustered bootstrap (K6's
                # cell_clustered_bootstrap.py) -- same logic, applied here so
                # THIS script's per-dataset coverage CI is cell-valid too,
                # not just the main results table's. cell_clustered_bootstrap_
                # coverage computes (y>=lower)&(y<=upper); encode the existing
                # 0/1 "covered" hit indicator as y=covered_g, lower=0.5,
                # upper=1.5 so that condition reduces to exactly covered_g==1.
                hit = covered_g.astype(float)
                boot = cell_clustered_bootstrap_coverage(
                    hit, np.full(n, 0.5), np.full(n, 1.5), cells_d)
                cell_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

        cov_pz = float(np.mean(r_test[mt] <= q_pooled_z))
        cov_pr = float(np.mean(r_test[mt] * std[d] <= q_pooled_real))
        out["per_dataset"][d] = {"n_test": n, "n_val": int(mv.sum()), "q_groupwise_z": q_g,
                                 "cov_groupwise": cov_g,
                                 "cp95_window_level_see_docstring_caveat": [lo, hi],
                                 "cell_bootstrap_95ci": cell_ci,
                                 "cov_pooled_z": cov_pz, "cov_pooled_real": cov_pr}
        cell_ci_str = ('[%5.1f, %5.1f]' % (100*cell_ci[0], 100*cell_ci[1])) if cell_ci else "n/a (<2 cells)"
        print(f"{d:8s}{n:8d}{100*cov_g:10.1f}%{'[%5.1f, %5.1f]' % (100*lo, 100*hi):>20s}"
              f"{cell_ci_str:>20s}{100*cov_pz:10.1f}%{100*cov_pr:13.1f}%")

    path = os.path.join(config.OUTPUT_DIR, "predictor_consistent_recalibration.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
