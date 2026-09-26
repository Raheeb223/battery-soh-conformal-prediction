"""
Hierarchical (shrinkage) groupwise conformal calibration.

Inputs, derived from existing outputs:
  - Per-dataset calibration quantile q_hat_g: recovered from
    outputs/conformal_groupwise_predictions.npz as (upper - lower) / 2, which
    is constant within each dataset.
  - Per-dataset calibration-set size n_g: counted from the lab_val field of
    cache/processed.npz.
  - Per-dataset coverage confidence-interval widths (sigma_g^2): fixed
    constants (CI_WIDTH_PP below), taken from the paper's per-dataset
    coverage table.

Shrinkage estimator:
    q_hat_g^shrunk = w_g * q_hat_g + (1 - w_g) * mu_hat
    w_g = n_g / (n_g + tau2 / sigma_g2)
with mu_hat and tau2 estimated by DerSimonian-Laird (random-effects
method-of-moments, well defined with as few as 6-7 groups).

Four-way comparison:
  (i)   pooled calibration (approximation, see console output)
  (ii)  hierarchical prior only (n_g = 0 -> pure mu_hat)
  (iii) hierarchical prior + 100 target-specific calibration samples
  (iv)  per-dataset groupwise quantile (upper reference)

Validation: calibration-level leave-one-dataset-out. For each dataset, mu_hat
and tau2 are refit from the other datasets' quantiles only, and all four
variants are evaluated on the held-out dataset's test set.

Decision criteria (fixed before the analysis):
  - Success: (ii) is meaningfully closer to 90% coverage than (i) for CALCE,
    BIT and XJTU.
  - Refutation: (ii) is no better than (i), or worse.
  - Ambiguous: improvement for some datasets and degradation for others;
    reported per dataset, together with each dataset's shrinkage weight.

Run:
    python hierarchical_calibration.py
Output:
    outputs/hierarchical_calibration_results.json
"""

import os
import json

import numpy as np

import config

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
DATASET_ORDER = ["MIT", "RWTH", "XJTU", "BIT", "Oxford", "CALCE"]  # NASA excluded, per established convention

CI_WIDTH_PP = {
    "MIT": 0.9, "RWTH": 1.2, "XJTU": 1.5,
    "BIT": 4.8, "Oxford": 6.4, "CALCE": 18.7,
}


def _load_predictions():
    path = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz")
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- run the main groupwise conformal "
              f"pipeline first (experiments.py's conformal-ablation section).")
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
                  f"std={qhat_std:.6f}) -- this predictions file may not be "
                  f"plain (non-normalized) groupwise calibration. Proceeding "
                  f"with the mean as an approximation, but treat this "
                  f"dataset's result with extra caution.")
        qhat_per_dataset[ds] = qhat_mean
    return qhat_per_dataset


def derive_calibration_set_sizes(cache):
    lab_val = cache["lab_val"]
    n_g = {}
    for ds in DATASET_ORDER:
        n_g[ds] = int((lab_val == ds).sum())
    return n_g


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
    cache = _load_cache()
    if preds is None or cache is None:
        return

    print(f"{'='*78}")
    print("EXPERIMENT A: HIERARCHICAL PARTIAL-POOLING CALIBRATION")
    print(f"{'='*78}\n")

    qhat_per_dataset = derive_qhat_per_dataset(preds)
    n_g_per_dataset = derive_calibration_set_sizes(cache)

    print("Derived inputs (no manual entry -- pulled directly from your files):")
    print(f"{'Dataset':10s} {'q_hat (derived)':>18s} {'n_g (calibration set)':>24s}")
    for ds in DATASET_ORDER:
        if ds in qhat_per_dataset:
            print(f"{ds:10s} {qhat_per_dataset[ds]:18.5f} {n_g_per_dataset[ds]:24d}")

    datasets = [d for d in DATASET_ORDER if d in qhat_per_dataset]
    qhats = [qhat_per_dataset[d] for d in datasets]
    sigma2s = [(CI_WIDTH_PP[d] / (2 * 1.96) / 100) ** 2 for d in datasets]

    mu_hat, tau2 = dersimonian_laird(qhats, sigma2s)
    tau2_ci_lo, tau2_ci_hi = bootstrap_tau2_ci(qhats, sigma2s)
    print(f"\nPooled mean quantile (mu_hat): {mu_hat:.5f}")
    print(f"Between-group variance (tau2): {tau2:.8f}  "
          f"(bootstrap 95% CI: [{tau2_ci_lo:.8f}, {tau2_ci_hi:.8f}])")

    y_test = preds["y_test"]
    pred_test = (preds["lower"] + preds["upper"]) / 2
    lab_test = preds["lab_test"]

    print(f"\n{'='*78}")
    print("FOUR-WAY COMPARISON (calibration-level LODO)")
    print(f"{'='*78}")
    header = f"{'Dataset':10s} {'(i) Pooled':>12s} {'(ii) Prior-only':>16s} {'(iii) Prior+100':>16s} {'(iv) True group.':>17s} {'Shrinkage w_g':>15s}"
    print(header)

    results = {"mu_hat_all": mu_hat, "tau2_all": tau2, "tau2_ci": [tau2_ci_lo, tau2_ci_hi], "per_dataset": {}}

    pooled_qhat_approx = float(np.average(qhats, weights=[n_g_per_dataset[d] for d in datasets]))
    print(f"\n  [!] NOTE: (i) Pooled uses an n-weighted average of each group's "
          f"OWN qhat as an approximation to true single-quantile pooled "
          f"calibration (Table 6's actual pooled numbers, computed from raw "
          f"pooled residuals directly, remain the authoritative reference -- "
          f"this approximation is only for internal consistency within this "
          f"script's four-way table).\n")

    for held_out in datasets:
        other_datasets = [d for d in datasets if d != held_out]
        other_qhats = [qhat_per_dataset[d] for d in other_datasets]
        other_sigma2s = [(CI_WIDTH_PP[d] / (2 * 1.96) / 100) ** 2 for d in other_datasets]

        mu_hat_loo, tau2_loo = dersimonian_laird(other_qhats, other_sigma2s)
        sigma2_held = (CI_WIDTH_PP[held_out] / (2 * 1.96) / 100) ** 2

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
            "mu_hat_loo": mu_hat_loo, "tau2_loo": tau2_loo,
            "coverage_pooled_approx": cov_pooled,
            "coverage_prior_only": cov_prior_only,
            "coverage_prior_plus_100": cov_prior_100,
            "coverage_true_groupwise": cov_true_groupwise,
            "shrinkage_weight_full_n": w_full,
        }

    out_path = os.path.join(config.OUTPUT_DIR, "hierarchical_calibration_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {out_path}")

    print(f"\n{'='*78}")
    print("INTERPRETATION: compare column (ii) against column (i) for CALCE, "
          "BIT, XJTU specifically -- if (ii) sits meaningfully closer to 90% "
          "than (i), that supports the hierarchical prior as a useful "
          "default for genuinely unseen sources. Check the shrinkage weight "
          "column for CALCE specifically: a LOW weight means CALCE resists "
          "pooling (its own quantile dominates even with a lot of data), a "
          "HIGH weight means it gets pulled toward the pooled mean despite "
          "potentially being an outlier -- report whichever the numbers "
          "actually show, not an assumed direction.")


if __name__ == "__main__":
    main()
