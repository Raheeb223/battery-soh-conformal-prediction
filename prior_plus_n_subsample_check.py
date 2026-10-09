"""
Subsample check for the hierarchical "prior + n target samples" scenario.

In hierarchical_calibration_corrected.py, column (iii) ("Prior+100") shrinks
the held-out dataset's FULL-calibration-set quantile using the shrinkage
weight computed for n=100. That is a weighting choice, not a quantile
estimated from 100 samples. This script computes the scenario directly:
for each budget n, it draws random n-window subsets of the target's own
validation residuals, estimates the conformal quantile FROM THE SUBSET (with
the finite-sample correction), shrinks it toward the leave-one-dataset-out
prior with the weight for that n, and evaluates coverage on the target's
test set. It reports mean and standard deviation over repeated draws.

Windows from the same cell are correlated, so n counts windows, not
independent samples; CALCE's calibration set is 125 windows from 2 cells.

Requires hierarchical_calibration_corrected.py and config.py in the same
project root, and the same input files that script already uses.

Run from the project root:
    python prior_plus_n_subsample_check.py
"""

import numpy as np

import hierarchical_calibration_corrected as h

TARGET = "CALCE"
BUDGETS = [5, 10, 25, 50, 100]
N_DRAWS = 500
SEED = 0


def finite_sample_quantile(residuals, alpha):
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(residuals, level)


def main():
    preds = h._load_predictions()
    val = h._load_val_predictions()
    cache = h._load_cache()

    qhat = h.derive_qhat_per_dataset(preds)
    datasets = [d for d in h.DATASET_ORDER if d in qhat]
    sigma_g2, _ = h.derive_sigma_g2_per_dataset(val, datasets)

    others = [d for d in datasets if d != TARGET]
    mu_loo, tau2_loo = h.dersimonian_laird([qhat[d] for d in others],
                                           [sigma_g2[d] for d in others])

    y_test = preds["y_test"]
    pred_test = (preds["lower"] + preds["upper"]) / 2
    lab_test = preds["lab_test"]

    mask = val["lab_val"] == TARGET
    residuals = np.abs(val["y_val"][mask] - val["mean_pred"][mask])
    n_windows = len(residuals)
    n_cells = len(np.unique(val["cell_val"][mask]))
    print(f"{TARGET}: {n_windows} calibration windows from {n_cells} cells")

    def coverage(q):
        return h.evaluate_coverage(y_test, pred_test, lab_test, TARGET, q)

    w100 = h.shrinkage_weight(100, tau2_loo, sigma_g2[TARGET])
    as_implemented = coverage(w100 * qhat[TARGET] + (1 - w100) * mu_loo)
    print(f"Column (iii) as implemented (full-set quantile, weight for n=100): {as_implemented:.3f}")
    print(f"True groupwise: {coverage(qhat[TARGET]):.3f}   Prior-only: {coverage(mu_loo):.3f}\n")

    rng = np.random.default_rng(SEED)
    print(f"{'n windows':>10} {'w_n':>7} {'coverage mean':>14} {'sd':>7}")
    for n in BUDGETS:
        w = h.shrinkage_weight(n, tau2_loo, sigma_g2[TARGET])
        covs = []
        for _ in range(N_DRAWS):
            idx = rng.choice(n_windows, size=n, replace=False)
            q_n = finite_sample_quantile(residuals[idx], h.ALPHA)
            covs.append(coverage(w * q_n + (1 - w) * mu_loo))
        print(f"{n:>10} {w:>7.3f} {np.mean(covs):>14.3f} {np.std(covs):>7.3f}")


if __name__ == "__main__":
    main()
