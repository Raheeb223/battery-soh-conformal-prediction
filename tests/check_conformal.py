"""
Unit tests for conformal.py, the split-conformal calibration used by every interval in the paper.

conformal.py is a pure statistics module (no project-specific I/O), so it
can be tested directly with synthetic data rather than real cell/window
data -- these are proper unit tests, not an end-to-end pipeline check.

Covers:
  1. SplitConformal's finite-sample marginal coverage guarantee, verified
     by Monte Carlo over many independent (calibration, test) draws from a
     known distribution -- average coverage across trials should sit at or
     above (1-alpha), per the standard split-conformal theorem (Vovk et al.),
     not just hit the target in one lucky run.
  2. Independent reimplementation of the calibration quantile (manual sort +
     order-statistic indexing, not np.quantile) cross-checked against
     SplitConformal.calibrate()'s qhat.
  3. GroupwiseConformal reduces to running SplitConformal separately per
     group -- same qhat per group, same per-sample interval.
  4. NormalizedSplitConformal reduces to SplitConformal (up to the constant
     mc_std factor) when mc_std is identical for every sample.
  5. GroupwiseConformal's "missing calibration data" fallback actually
     engages (not just present in code, but triggered) when a label appears
     at predict time with zero calibration-time samples.

Synthetic data only; no project files needed. Exits with status 1 if any check fails.
Run from the repository root:  python tests/check_conformal.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformal import SplitConformal, GroupwiseConformal, NormalizedSplitConformal

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def test_marginal_coverage_monte_carlo():
    print("\n1. SplitConformal marginal coverage (Monte Carlo, N(0,1) residuals)")
    rng = np.random.default_rng(0)
    alpha = 0.1
    n_cal, n_test, n_trials = 200, 500, 300
    coverages = []
    for _ in range(n_trials):
        y_cal = rng.normal(0, 1, n_cal)
        pred_cal = np.zeros(n_cal)
        y_test = rng.normal(0, 1, n_test)
        pred_test = np.zeros(n_test)

        sc = SplitConformal(alpha=alpha)
        sc.calibrate(y_cal, pred_cal)
        coverages.append(sc.coverage(y_test, pred_test))

    mean_cov = np.mean(coverages)
    print(f"   mean coverage over {n_trials} trials: {mean_cov:.4f} (target >= {1-alpha:.2f})")
    # Allow a small Monte Carlo tolerance -- the theorem guarantees
    # E[coverage] >= 1-alpha exactly, but with finite trials/finite n_test
    # there's sampling noise; demand it's not meaningfully BELOW target.
    check(f"mean coverage >= {1-alpha-0.02:.2f} (theorem: must be >= {1-alpha:.2f})",
          mean_cov >= 1 - alpha - 0.02)


def test_independent_quantile_reimplementation():
    print("\n2. Independent reimplementation of the calibration quantile")
    print("   NOTE: conformal.py computes qhat via np.quantile(residuals, k/n,")
    print("   method='linear' [numpy default]), NOT the textbook discrete order")
    print("   statistic residuals_sorted[k-1]. np.quantile's linear-interpolation")
    print("   index is h=(k/n)*(n-1), which always falls in [k-1, k) for k<n, so")
    print("   it linearly blends the k-th and (k+1)-th order statistics -- this")
    print("   makes conformal.py's qhat PROVABLY >= the discrete order statistic")
    print("   (since sorted residuals are non-decreasing), i.e. at least as")
    print("   conservative as the textbook recipe, never less. This test checks")
    print("   that inequality directly, rather than exact equality (verified by")
    print("   derivation, not just empirically: this holds for ANY sorted input).")
    rng = np.random.default_rng(1)
    alpha = 0.1
    always_at_least_as_conservative = True
    max_relative_gap = 0.0
    for trial in range(20):
        n = rng.integers(10, 500)
        y_cal = rng.normal(0, rng.uniform(0.5, 2.0), n)
        pred_cal = rng.normal(0, 0.1, n)

        sc = SplitConformal(alpha=alpha)
        qhat = sc.calibrate(y_cal, pred_cal)

        # Textbook discrete order statistic: the ceil((n+1)(1-alpha))-th
        # SMALLEST residual (1-indexed), capped at n.
        residuals = np.sort(np.abs(y_cal - pred_cal))
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(k, n)
        order_stat_qhat = residuals[k - 1]

        if qhat < order_stat_qhat - 1e-9:
            always_at_least_as_conservative = False
            print(f"   trial {trial}: qhat={qhat} < order_stat={order_stat_qhat} (n={n}) -- VIOLATION")
        max_relative_gap = max(max_relative_gap, (qhat - order_stat_qhat) / max(order_stat_qhat, 1e-9))
    print(f"   max observed (qhat - order_stat)/order_stat across trials: {max_relative_gap:.4f}")
    check("conformal.py's qhat is always >= the textbook discrete order statistic (20 trials)",
          always_at_least_as_conservative)


def test_groupwise_matches_separate_split_conformal():
    print("\n3. GroupwiseConformal matches running SplitConformal separately per group")
    rng = np.random.default_rng(2)
    alpha = 0.1
    labels_cal = np.array(["A"] * 100 + ["B"] * 150 + ["C"] * 50)
    y_cal = rng.normal(0, 1, len(labels_cal))
    pred_cal = rng.normal(0, 0.1, len(labels_cal))

    gc = GroupwiseConformal(alpha=alpha)
    group_qhats = gc.calibrate(y_cal, pred_cal, labels_cal)

    all_match = True
    for g in ["A", "B", "C"]:
        mask = labels_cal == g
        sc = SplitConformal(alpha=alpha)
        qhat_separate = sc.calibrate(y_cal[mask], pred_cal[mask])
        if abs(group_qhats[g] - qhat_separate) > 1e-9:
            all_match = False
    check("per-group qhat matches standalone SplitConformal exactly for all 3 groups", all_match)


def test_normalized_reduces_to_unnormalized_with_constant_mc_std():
    print("\n4. NormalizedSplitConformal reduces to SplitConformal when mc_std is constant")
    rng = np.random.default_rng(3)
    alpha = 0.1
    n = 300
    y_cal = rng.normal(0, 1, n)
    pred_cal = rng.normal(0, 0.1, n)
    const_std = 2.5
    mc_std_cal = np.full(n, const_std)

    sc = SplitConformal(alpha=alpha)
    qhat_plain = sc.calibrate(y_cal, pred_cal)

    nsc = NormalizedSplitConformal(alpha=alpha)
    qhat_norm = nsc.calibrate(y_cal, pred_cal, mc_std_cal)

    # normalized residual = |y-pred|/const_std, so its (1-alpha) quantile is
    # qhat_plain/const_std -- and predict_interval multiplies back by
    # const_std, so the realized half-width should match exactly.
    y_test = rng.normal(0, 1, 50)
    pred_test = np.zeros(50)
    lo_plain, hi_plain = sc.predict_interval(pred_test)
    lo_norm, hi_norm = nsc.predict_interval(pred_test, np.full(50, const_std))

    half_width_match = np.allclose(hi_plain - lo_plain, hi_norm - lo_norm, atol=1e-9)
    check("realized interval half-width matches SplitConformal when mc_std is constant",
          half_width_match)


def test_groupwise_missing_calibration_fallback_engages():
    print("\n5. GroupwiseConformal's missing-calibration-data fallback actually engages")
    rng = np.random.default_rng(4)
    alpha = 0.1
    # calibrate on groups A, B only
    labels_cal = np.array(["A"] * 100 + ["B"] * 100)
    y_cal = rng.normal(0, 1, 200)
    pred_cal = rng.normal(0, 0.1, 200)

    gc = GroupwiseConformal(alpha=alpha)
    gc.calibrate(y_cal, pred_cal, labels_cal)

    # predict for a group "C" that had ZERO calibration samples
    labels_test = np.array(["A"] * 10 + ["C"] * 10)
    pred_test = np.zeros(20)
    lower, upper = gc.predict_interval(pred_test, labels_test)

    no_nans = not np.isnan(lower).any() and not np.isnan(upper).any()
    expected_pooled_qhat = np.mean([c.qhat for c in gc.group_conformals.values()])
    c_mask = labels_test == "C"
    c_half_width_correct = np.allclose(upper[c_mask] - pred_test[c_mask], expected_pooled_qhat, atol=1e-9)
    check("fallback produces no NaNs for the unseen group", no_nans)
    check("fallback half-width for unseen group equals the pooled qhat across calibrated groups",
          c_half_width_correct)


if __name__ == "__main__":
    test_marginal_coverage_monte_carlo()
    test_independent_quantile_reimplementation()
    test_groupwise_matches_separate_split_conformal()
    test_normalized_reduces_to_unnormalized_with_constant_mc_std()
    test_groupwise_missing_calibration_fallback_engages()

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
