"""
Critical fix A: validate the hierarchical calibration model's sigma_g^2 proxy
against a direct estimate of calibration-quantile uncertainty.

THE PROBLEM BEING ADDRESSED: Section 5.3.6 and Appendix A.3 estimate
sigma_g^2 (the sampling variance feeding the shrinkage weight) from the
Clopper-Pearson interval width of TEST-set COVERAGE -- a different quantity
from what shrinkage actually needs, which is the sampling variance of the
CALIBRATION QUANTILE itself. The appendix already states this is a proxy;
this script tests directly whether it is a GOOD one, by estimating quantile
uncertainty the correct way -- resampling calibration CELLS (not windows)
with replacement and recomputing the conformal quantile on each resample --
and comparing the result to the existing proxy and to a simple fixed-
shrinkage baseline that ignores dataset-specific variance information
entirely.

WHAT THIS DOES NOT DO: this script validates the sigma_g^2 INPUT to the
existing hierarchical_calibration.py pipeline; it does not re-run that
pipeline's own mu_hat/tau2_hat/shrinkage-weight computation, which already
exists as a separate, tested script. If sigma_g^2 estimates change
substantially here, that pipeline should be re-run with the corrected
values, which is a natural next step after this validation rather than
something duplicated here.

USAGE: python validate_sigma_g_proxy.py
"""

import numpy as np

PROCESSED_NPZ_PATH = r"C:\Users\User\PycharmProjects\cache\processed.npz"
VAL_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions_val.npz"

ALPHA = 0.10
N_BOOTSTRAP = 2000
SEED = 0
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]

# The existing proxy: sigma_g^2 derived from Table 7's Clopper-Pearson CI
# width on TEST coverage (copied from the manuscript's own reported values).
EXISTING_PROXY_SIGMA_G2 = {
    "MIT": (0.009 / (2 * 1.96)) ** 2,
    "RWTH": (0.012 / (2 * 1.96)) ** 2,
    "XJTU": (0.015 / (2 * 1.96)) ** 2,
    "BIT": (0.048 / (2 * 1.96)) ** 2,
    "Oxford": (0.064 / (2 * 1.96)) ** 2,
    "CALCE": (0.187 / (2 * 1.96)) ** 2,
    "NASA": (0.897 / (2 * 1.96)) ** 2,
}


def get_val_residuals_and_cells(dataset: str):
    val = np.load(VAL_PRED_PATH, allow_pickle=True)
    cache = np.load(PROCESSED_NPZ_PATH, allow_pickle=True)
    soh_mean = cache["soh_mean"].item()
    soh_std = cache["soh_std"].item()
    m, s = soh_mean[dataset], soh_std[dataset]

    mask = val["lab_val"] == dataset
    y_real = val["y_val"][mask] * s + m
    pred_real = val["mean_pred"][mask] * s + m
    residuals = np.abs(y_real - pred_real)
    cell_ids = val["cell_val"][mask]
    return residuals, cell_ids


def pooled_quantile_at_alpha(residuals: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(residuals)
    if n == 0:
        return np.nan
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(residuals, level)


def direct_quantile_bootstrap(residuals: np.ndarray, cell_ids: np.ndarray,
                               n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    """Resamples whole calibration CELLS with replacement (matching the
    same cell-clustered logic already validated for Priority 1's coverage
    bootstrap), recomputes the conformal QUANTILE on each resample, and
    returns the array of bootstrap quantile estimates -- this is the
    directly-relevant uncertainty for shrinkage, unlike test-coverage CI
    width."""
    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    if n_cells < 2:
        return None  # undefined, same limitation as the coverage bootstrap

    cell_to_indices = {c: np.where(cell_ids == c)[0] for c in unique_cells}
    boot_quantiles = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_cells = rng.choice(unique_cells, size=n_cells, replace=True)
        idx = np.concatenate([cell_to_indices[c] for c in sampled_cells])
        boot_quantiles[b] = pooled_quantile_at_alpha(residuals[idx])
    return boot_quantiles


def main():
    print(f"{'Dataset':<8}{'n_cells':<9}{'Point q':<10}{'Direct sigma_g':<16}"
          f"{'Existing proxy':<16}{'Ratio (direct/proxy)'}")

    results = {}
    for dataset in DATASETS:
        residuals, cell_ids = get_val_residuals_and_cells(dataset)
        n_cells = len(np.unique(cell_ids))
        point_q = pooled_quantile_at_alpha(residuals)

        boot = direct_quantile_bootstrap(residuals, cell_ids)
        if boot is None:
            print(f"{dataset:<8}{n_cells:<9}{point_q:<10.4f}"
                  f"{'undefined (<2 cells)':<16}{EXISTING_PROXY_SIGMA_G2[dataset]:<16.6f}{'--'}")
            continue

        direct_sigma_g2 = np.var(boot)
        existing_proxy = EXISTING_PROXY_SIGMA_G2[dataset]
        ratio = direct_sigma_g2 / existing_proxy if existing_proxy > 0 else float("nan")
        results[dataset] = {"direct": direct_sigma_g2, "proxy": existing_proxy, "ratio": ratio}
        print(f"{dataset:<8}{n_cells:<9}{point_q:<10.4f}"
              f"{direct_sigma_g2:<16.8f}{existing_proxy:<16.8f}{ratio:.2f}x")

    print("\nA ratio near 1.0 supports the existing proxy as reasonable; a ratio far")
    print("from 1.0 (either direction) indicates the proxy under- or over-states")
    print("true calibration-quantile uncertainty for that dataset, and the")
    print("hierarchical shrinkage weights derived from it should be revisited.")

    print("\n--- Simple fixed-shrinkage baseline for comparison ---")
    print("A baseline that ignores all dataset-specific variance information,")
    print("shrinking every group toward the pooled mean by the same fixed 50%,")
    print("regardless of that group's own calibration-set size or variance:")
    all_residuals = []
    for dataset in DATASETS:
        r, _ = get_val_residuals_and_cells(dataset)
        all_residuals.append(r)
    pooled_q = pooled_quantile_at_alpha(np.concatenate(all_residuals))
    print(f"Pooled quantile (shrinkage target): {pooled_q:.4f}")
    for dataset in DATASETS:
        residuals, _ = get_val_residuals_and_cells(dataset)
        own_q = pooled_quantile_at_alpha(residuals)
        fixed_shrunk_q = 0.5 * own_q + 0.5 * pooled_q
        print(f"  {dataset}: own q={own_q:.4f}, fixed-50%-shrunk q={fixed_shrunk_q:.4f}")


if __name__ == "__main__":
    main()
