"""
hierarchical_calibration.py  (Experiment A -- CORRECTED variance input)

This is the user's original hierarchical_calibration.py with exactly one
change: sigma_g^2 (the calibration-quantile variance feeding the
DerSimonian-Laird shrinkage estimator) is now computed via a direct,
cell-clustered bootstrap of the calibration quantile itself, instead of
being derived from Table 7's per-window coverage confidence-interval width.

WHY THE ORIGINAL PROXY WAS WRONG, NOT JUST IMPRECISE: the original
sigma2s = (CI_WIDTH_PP[d] / (2*1.96) / 100) ** 2 line converts a COVERAGE
confidence-interval width (a percentage-point quantity, e.g. "0.9" meaning
0.9 percentage points of coverage precision) into something treated as
directly comparable to qhat_g (a normalized SOH RESIDUAL/quantile value).
These are two different quantities on two different scales -- a units
mismatch, not merely an approximation. The corrected sigma_g^2 below is
computed as the actual sampling variance of qhat_g itself (resampling
calibration cells with replacement and recomputing the quantile on each
resample), in the SAME normalized units qhat_g already uses, so it is
finally the right quantity on the right scale.

Oxford has only 1 unique validation cell, making direct cell-clustered
estimation mathematically undefined for it (no between-cell variability
exists to resample). Oxford therefore falls back to the original proxy,
explicitly flagged in this script's own printed output, not silently
substituted.

Everything else -- qhat_g derivation, n_g derivation, DerSimonian-Laird,
the four-way comparison, calibration-level LODO refitting, and the JSON
output schema -- is UNCHANGED from the original script.

Run from your project root:
    python hierarchical_calibration_corrected.py
Output:
    outputs/hierarchical_calibration_results_corrected.json
"""

import os
import json

import numpy as np

import config

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
DATASET_ORDER = ["MIT", "RWTH", "XJTU", "BIT", "Oxford", "CALCE"]  # NASA excluded, per established convention

# Retained ONLY as Oxford's fallback and as a printed sensitivity comparison
# -- no longer the primary sigma_g^2 source. See module docstring.
CI_WIDTH_PP_PROXY = {
    "MIT": 0.9, "RWTH": 1.2, "XJTU": 1.5,
    "BIT": 4.8, "Oxford": 6.4, "CALCE": 18.7,
}

N_BOOTSTRAP_SIGMA = 2000


def _load_predictions():
    path = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- run the main groupwise conformal "
              f"pipeline first.")
        return None
    return np.load(path, allow_pickle=True)


def _load_val_predictions():
    """NEW: loads the validation-set predictions needed for the direct
    sigma_g^2 bootstrap -- these were not needed by the original proxy,
    which only used already-published Table 7 numbers."""
    path = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- needed for the direct sigma_g^2 "
              f"bootstrap. Falling back to the proxy for ALL datasets if "
              f"this file is missing.")
        return None
    return np.load(path, allow_pickle=True)


def _load_cache():
    path = os.path.join(config.CACHE_DIR, "processed.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- run preprocessing.py first.")
        return None
    return np.load(path, allow_pickle=True)


def derive_qhat_per_dataset(preds):
    y_test = preds["y_test"]
    lower = preds["lower"]
    upper = preds["upper"]
    lab_test = preds["lab_test"]

    qhat_per_dataset = {}
    for ds in DATASET_ORDER:
        mask = lab_test == ds
        if mask.sum() == 0:
            print(f"  [!] No test rows found for {ds} in this predictions file.")
            continue
        widths = (upper[mask] - lower[mask]) / 2
        qhat_mean = float(widths.mean())
        qhat_std = float(widths.std())
        if qhat_std > 1e-4 * max(abs(qhat_mean), 1e-8):
            print(f"  [!] WARNING: {ds}'s interval half-width is NOT constant "
                  f"across its test rows (mean={qhat_mean:.6f}, "
                  f"std={qhat_std:.6f}) -- proceeding with the mean as an "
                  f"approximation.")
        qhat_per_dataset[ds] = qhat_mean
    return qhat_per_dataset


def derive_calibration_set_sizes(cache):
    lab_val = cache["lab_val"]
    n_g = {}
    for ds in DATASET_ORDER:
        n_g[ds] = int((lab_val == ds).sum())
    return n_g


def compute_direct_sigma_g2(residuals: np.ndarray, cell_ids: np.ndarray,
                             alpha: float = ALPHA, n_bootstrap: int = N_BOOTSTRAP_SIGMA,
                             seed: int = 0):
    """Direct, cell-clustered bootstrap estimate of qhat_g's own sampling
    variance, in the SAME (normalized) units as qhat_g itself. Returns
    (sigma_g2, method_string); sigma_g2 is None if fewer than 2 unique
    cells exist (undefined, not a silently-chosen fallback value)."""
    unique_cells = np.unique(cell_ids)
    if len(unique_cells) < 2:
        return None, f"undefined ({len(unique_cells)} unique cell(s), need >=2)"

    rng = np.random.default_rng(seed)
    cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}

    def pooled_quantile(r, a):
        n = len(r)
        level = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
        return np.quantile(r, level)

    boot_q = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_cells = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        idx = np.concatenate([cell_to_idx[c] for c in sampled_cells])
        boot_q[b] = pooled_quantile(residuals[idx], alpha)

    return float(np.var(boot_q)), "direct cell-clustered bootstrap"


def derive_sigma_g2_per_dataset(val_preds, datasets):
    """For each dataset, compute direct sigma_g^2 from VALIDATION residuals
    in NORMALIZED units (matching qhat_g's own units exactly -- NOT
    de-normalized to real SOH units, since qhat_g is not de-normalized
    either). Falls back to the proxy, explicitly flagged, when direct
    estimation is undefined (Oxford)."""
    sigma_g2 = {}
    methods = {}
    if val_preds is None:
        for ds in datasets:
            sigma_g2[ds] = (CI_WIDTH_PP_PROXY[ds] / (2 * 1.96) / 100) ** 2
            methods[ds] = "PROXY (val predictions file not found)"
        return sigma_g2, methods

    for ds in datasets:
        mask = val_preds["lab_val"] == ds
        residuals = np.abs(val_preds["y_val"][mask] - val_preds["mean_pred"][mask])
        cell_ids = val_preds["cell_val"][mask]
        direct, method = compute_direct_sigma_g2(residuals, cell_ids)
        if direct is not None:
            sigma_g2[ds] = direct
            methods[ds] = method
        else:
            sigma_g2[ds] = (CI_WIDTH_PP_PROXY[ds] / (2 * 1.96) / 100) ** 2
            methods[ds] = f"PROXY fallback ({method})"
    return sigma_g2, methods


def dersimonian_laird(qhats, sigma2s):
    qhats = np.array(qhats)
    sigma2s = np.array(sigma2s)
    weights = 1.0 / sigma2s
    mu_fixed = np.sum(weights * qhats) / np.sum(weights)
    Q = np.sum(weights * (qhats - mu_fixed) ** 2)
    k = len(qhats)
    df = k - 1
    c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    weights_re = 1.0 / (sigma2s + tau2)
    mu_hat = np.sum(weights_re * qhats) / np.sum(weights_re)
    return mu_hat, tau2


def bootstrap_tau2_ci(qhats, sigma2s, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    qhats, sigma2s = np.array(qhats), np.array(sigma2s)
    k = len(qhats)
    boot_tau2s = []
    for _ in range(n_boot):
        idx = rng.choice(k, size=k, replace=True)
        if len(set(idx)) < 2:
            continue
        _, tau2_b = dersimonian_laird(qhats[idx], sigma2s[idx])
        boot_tau2s.append(tau2_b)
    boot_tau2s = np.array(boot_tau2s)
    return float(np.percentile(boot_tau2s, 2.5)), float(np.percentile(boot_tau2s, 97.5))


def shrinkage_weight(n_g, tau2, sigma_g2):
    if tau2 <= 0:
        return 0.0
    denom = n_g + tau2 / sigma_g2
    return n_g / denom if denom > 0 else 0.0


def evaluate_coverage(y_test, pred_test, lab_test, ds, qhat):
    mask = lab_test == ds
    if mask.sum() == 0:
        return float("nan")
    lo = pred_test[mask] - qhat
    hi = pred_test[mask] + qhat
    return float(np.mean((y_test[mask] >= lo) & (y_test[mask] <= hi)))


def main():
    preds = _load_predictions()
    val_preds = _load_val_predictions()
    cache = _load_cache()
    if preds is None or cache is None:
        return

    print(f"{'='*78}")
    print("EXPERIMENT A: HIERARCHICAL PARTIAL-POOLING CALIBRATION (CORRECTED sigma_g^2)")
    print(f"{'='*78}\n")

    qhat_per_dataset = derive_qhat_per_dataset(preds)
    n_g_per_dataset = derive_calibration_set_sizes(cache)
    datasets = [d for d in DATASET_ORDER if d in qhat_per_dataset]

    sigma_g2_per_dataset, sigma_methods = derive_sigma_g2_per_dataset(val_preds, datasets)

    print("Derived inputs:")
    print(f"{'Dataset':10s} {'q_hat (derived)':>18s} {'n_g':>8s} {'sigma_g2 (corrected)':>22s} {'method':>34s}")
    for ds in datasets:
        print(f"{ds:10s} {qhat_per_dataset[ds]:18.5f} {n_g_per_dataset[ds]:8d} "
              f"{sigma_g2_per_dataset[ds]:22.8f} {sigma_methods[ds]:>34s}")

    qhats = [qhat_per_dataset[d] for d in datasets]
    sigma2s = [sigma_g2_per_dataset[d] for d in datasets]
    sigma2s_proxy = [(CI_WIDTH_PP_PROXY[d] / (2 * 1.96) / 100) ** 2 for d in datasets]

    mu_hat, tau2 = dersimonian_laird(qhats, sigma2s)
    tau2_ci_lo, tau2_ci_hi = bootstrap_tau2_ci(qhats, sigma2s)
    mu_hat_proxy, tau2_proxy = dersimonian_laird(qhats, sigma2s_proxy)

    print(f"\nPooled mean quantile (mu_hat), CORRECTED: {mu_hat:.5f}")
    print(f"Between-group variance (tau2), CORRECTED: {tau2:.8f}  "
          f"(bootstrap 95% CI: [{tau2_ci_lo:.8f}, {tau2_ci_hi:.8f}])")
    print(f"[sensitivity comparison] mu_hat under original proxy: {mu_hat_proxy:.5f}, tau2: {tau2_proxy:.8f}")

    y_test = preds["y_test"]
    pred_test = (preds["lower"] + preds["upper"]) / 2
    lab_test = preds["lab_test"]

    print(f"\n{'='*78}")
    print("FOUR-WAY COMPARISON (calibration-level LODO, CORRECTED sigma_g^2)")
    print(f"{'='*78}")
    header = f"{'Dataset':10s} {'(i) Pooled':>12s} {'(ii) Prior-only':>16s} {'(iii) Prior+100':>16s} {'(iv) True group.':>17s} {'Shrinkage w_g':>15s}"
    print(header)

    results = {
        "mu_hat_all_corrected": mu_hat, "tau2_all_corrected": tau2,
        "tau2_ci_corrected": [tau2_ci_lo, tau2_ci_hi],
        "mu_hat_all_proxy_sensitivity": mu_hat_proxy, "tau2_all_proxy_sensitivity": tau2_proxy,
        "sigma_g2_methods": sigma_methods,
        "per_dataset": {},
    }

    pooled_qhat_approx = float(np.average(qhats, weights=[n_g_per_dataset[d] for d in datasets]))
    print(f"\n  [!] NOTE: (i) Pooled uses an n-weighted average of each group's "
          f"OWN qhat as an approximation to true single-quantile pooled "
          f"calibration (Table 6's actual pooled numbers remain the "
          f"authoritative reference for that quantity).\n")

    for held_out in datasets:
        other_datasets = [d for d in datasets if d != held_out]
        other_qhats = [qhat_per_dataset[d] for d in other_datasets]
        other_sigma2s = [sigma_g2_per_dataset[d] for d in other_datasets]

        mu_hat_loo, tau2_loo = dersimonian_laird(other_qhats, other_sigma2s)
        sigma2_held = sigma_g2_per_dataset[held_out]

        cov_pooled = evaluate_coverage(y_test, pred_test, lab_test, held_out, pooled_qhat_approx)
        cov_prior_only = evaluate_coverage(y_test, pred_test, lab_test, held_out, mu_hat_loo)

        w_100 = shrinkage_weight(100, tau2_loo, sigma2_held)
        qhat_prior_100 = w_100 * qhat_per_dataset[held_out] + (1 - w_100) * mu_hat_loo
        cov_prior_100 = evaluate_coverage(y_test, pred_test, lab_test, held_out, qhat_prior_100)

        cov_true_groupwise = evaluate_coverage(y_test, pred_test, lab_test, held_out, qhat_per_dataset[held_out])

        w_full = shrinkage_weight(n_g_per_dataset[held_out], tau2_loo, sigma2_held)

        print(f"{held_out:10s} {cov_pooled*100:11.1f}% {cov_prior_only*100:15.1f}% "
              f"{cov_prior_100*100:15.1f}% {cov_true_groupwise*100:16.1f}% {w_full:15.4f}")

        results["per_dataset"][held_out] = {
            "q_hat_true": qhat_per_dataset[held_out],
            "n_g": n_g_per_dataset[held_out],
            "sigma_g2_used": sigma_g2_per_dataset[held_out],
            "sigma_g2_method": sigma_methods[held_out],
            "mu_hat_loo": mu_hat_loo, "tau2_loo": tau2_loo,
            "coverage_pooled_approx": cov_pooled,
            "coverage_prior_only": cov_prior_only,
            "coverage_prior_plus_100": cov_prior_100,
            "coverage_true_groupwise": cov_true_groupwise,
            "shrinkage_weight_full_n": w_full,
        }

    out_path = os.path.join(config.OUTPUT_DIR, "hierarchical_calibration_results_corrected.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
