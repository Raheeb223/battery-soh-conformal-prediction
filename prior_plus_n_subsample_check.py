"""
Subsample check for the hierarchical "prior + n target samples" scenario.

ERROR_REGISTER E6 fix: this previously imported hierarchical_calibration_corrected,
which uses the withdrawn shrinkage weight w = n/(n + tau2/sigma2_g) (see E5) and
operates on z-scored quantiles derived from predicted-interval half-widths.
Rewritten against hierarchical_calibration_v2's actual model: real-SOH-unit
residuals, and the correct empirical-Bayes weight w = tau2/(tau2+sigma2_g),
where sigma2_g is the CELL-CLUSTERED BOOTSTRAP variance of the quantile
estimated from whichever n-window subsample is being evaluated (not a
function of n alone -- under the correct model, sample size enters only
through its effect on that bootstrap variance).

For each budget n (windows, not cells -- windows from the same cell are
correlated, so this is not n independent calibration points; CALCE's own
validation set is 125 windows from 2 cells), draws random n-window subsets of
the target's own validation residuals, estimates the subsample's own quantile
AND its cell-clustered bootstrap variance, shrinks toward the leave-one-
dataset-out prior with the weight implied by THAT variance, and evaluates
coverage on the target's test set. Reports mean and SD over repeated draws.

A subsample whose n windows happen to come from only 1 unique cell has an
undefined cell-bootstrap variance (no between-cell spread to resample); such
draws are skipped and the skip count is reported, rather than silently
substituted with a proxy.

Requires hierarchical_calibration_v2.py and the same input files it uses.

Run from the project root:
    python prior_plus_n_subsample_check.py
"""

import os

import numpy as np

import hierarchical_calibration_v2 as h

TARGET = "CALCE"
BUDGETS = [5, 10, 25, 50, 100]
N_DRAWS = 500
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    cache = np.load(os.path.join(h.config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    val = np.load(os.path.join(h.config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
    test = np.load(os.path.join(h.config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    std = cache["soh_std"].item()

    rv, cells = {}, {}
    for d in h.ALL:
        m = val["lab_val"] == d
        rv[d] = np.abs(val["y_val"][m] - val["mean_pred"][m]) * std[d]
        cells[d] = val["cell_val"][m]
    rt = {}
    for d in h.ALL:
        m = test["lab_test"] == d
        rt[d] = np.abs(test["y_test"][m] - test["mean_pred"][m]) * std[d]

    q, s2 = {}, {}
    for d in h.ALL:
        q[d] = h.fsq(rv[d])
        s2[d] = h.cell_bootstrap_variance(rv[d], cells[d], rng)
    fit = [d for d in h.ALL if s2[d] is not None]
    others = [d for d in fit if d != TARGET]
    mu_loo, tau2_loo = h.dersimonian_laird([q[d] for d in others], [s2[d] for d in others])

    target_residuals = rv[TARGET]
    target_cells = cells[TARGET]
    n_windows = len(target_residuals)
    n_cells_total = len(np.unique(target_cells))
    print(f"{TARGET}: {n_windows} calibration windows from {n_cells_total} cells "
          f"(leave-one-dataset-out prior: mu_hat={mu_loo:.5f}, tau2_hat={tau2_loo:.3e})")

    def coverage(threshold):
        return float(np.mean(rt[TARGET] <= threshold))

    print(f"Reference points -- true groupwise (full-set quantile): {coverage(q[TARGET]):.3f}   "
          f"Prior-only: {coverage(mu_loo):.3f}\n")

    print(f"{'n windows':>10} {'draws used':>11} {'draws skipped (1 cell)':>24} "
          f"{'w (mean)':>9} {'coverage mean':>14} {'sd':>7}")
    for n in BUDGETS:
        covs, ws = [], []
        n_skipped = 0
        for _ in range(N_DRAWS):
            idx = rng.choice(n_windows, size=n, replace=False)
            sub_residuals = target_residuals[idx]
            sub_cells = target_cells[idx]
            q_n = h.fsq(sub_residuals)
            s2_n = h.cell_bootstrap_variance(sub_residuals, sub_cells, rng)
            if s2_n is None or s2_n <= 0:
                n_skipped += 1
                continue
            w = tau2_loo / (tau2_loo + s2_n) if tau2_loo > 0 else 0.0
            covs.append(coverage(w * q_n + (1 - w) * mu_loo))
            ws.append(w)
        if covs:
            print(f"{n:>10} {len(covs):>11} {n_skipped:>24} {np.mean(ws):>9.3f} "
                  f"{np.mean(covs):>14.3f} {np.std(covs):>7.3f}")
        else:
            print(f"{n:>10} {0:>11} {n_skipped:>24}  -- all draws had <2 unique cells, undefined --")


if __name__ == "__main__":
    main()
