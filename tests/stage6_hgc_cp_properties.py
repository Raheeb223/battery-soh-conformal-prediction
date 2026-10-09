"""
Stage 6: HGC-CP property tests, beyond what hgc_cp.py's own __main__ block
already covers (MMD ranking sanity, reproducibility of the bandwidth
subsample, inf-width handling -- all independently re-run and confirmed
passing as part of this stage; see TEST_REPORT.md).

New checks here:
  1. compute_group_weights always sums to 1 (by construction, but asserted
     explicitly rather than only implied by the softmax normalization).
  2. Increasing lambda concentrates MORE weight on the closest group
     (monotonic sanity check on the heterogeneity-weighting direction).
  3. Single-group calibration (no real heterogeneity to weight across)
     reduces hgc_cp_interval to the ordinary split-conformal quantile --
     HGC-CP must not do something different from standard conformal
     prediction when there is nothing to adapt to.
  4. corrected_weighted_quantile's returned quantile, when finite, always
     lies within [min(scores), max(scores)] -- a basic sanity bound any
     quantile of the scores must satisfy.

Run:  python tests/stage6_hgc_cp_properties.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hgc_cp import (
    compute_group_weights, corrected_weighted_quantile, hgc_cp_interval,
    pooled_quantile,
)

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def test_weights_sum_to_one():
    print("\n1. compute_group_weights sums to 1")
    rng = np.random.default_rng(1)
    for trial in range(10):
        n_groups = rng.integers(2, 8)
        distances = {f"g{i}": float(rng.uniform(0, 10)) for i in range(n_groups)}
        lam = float(rng.uniform(0.1, 20))
        w = compute_group_weights(distances, lam)
        check(f"trial {trial} (n_groups={n_groups}, lam={lam:.2f}): sum={sum(w.values()):.8f}",
              abs(sum(w.values()) - 1.0) < 1e-8)


def test_lambda_concentrates_weight():
    print("\n2. increasing lambda concentrates weight on the closest group")
    distances = {"close": 0.1, "mid": 2.0, "far": 8.0}
    prev = -1.0
    monotonic = True
    weights_seen = []
    for lam in [0.1, 1.0, 3.0, 10.0, 30.0]:
        w = compute_group_weights(distances, lam)["close"]
        weights_seen.append((lam, w))
        if w < prev - 1e-9:
            monotonic = False
        prev = w
    print(f"   weight on 'close' group vs lambda: {weights_seen}")
    check("weight on closest group is non-decreasing in lambda", monotonic)


def test_single_group_reduces_to_split_conformal():
    print("\n3. single-group calibration reduces to ordinary split-conformal quantile")
    rng = np.random.default_rng(2)
    scores = np.abs(rng.normal(0, 0.1, 300))
    group_ids = np.array(["only"] * len(scores))
    distances = {"only": 0.0}
    alpha = 0.1

    lo, hi, q_star, _ = hgc_cp_interval(0.0, scores, group_ids, distances,
                                         lam=3.0, alpha=alpha, beta=0.0)
    standard_q = pooled_quantile(scores, alpha)
    rel_diff = abs(q_star - standard_q) / standard_q
    print(f"   HGC-CP q*={q_star:.5f} vs standard split-conformal quantile={standard_q:.5f} "
          f"(rel diff={rel_diff:.4f})")
    # Not expected to be bit-identical (corrected_weighted_quantile uses a
    # slightly different finite-sample correction than pooled_quantile's
    # ceil((n+1)(1-alpha))/n formula), but with a single group and n=300,
    # beta=0, lam irrelevant (only one group to weight), they must be close.
    check("single-group HGC-CP quantile matches standard conformal within 5%", rel_diff < 0.05)


def test_quantile_within_score_bounds():
    print("\n4. corrected_weighted_quantile stays within [min(scores), max(scores)] when finite")
    rng = np.random.default_rng(3)
    all_within = True
    for trial in range(20):
        n = rng.integers(20, 200)
        scores = np.abs(rng.normal(0, 1, n))
        weights = rng.uniform(0.1, 1.0, n)
        weights = weights / weights.sum() * rng.uniform(0.5, 1.0)  # leave some mass for test point
        test_weight = float(rng.uniform(0.001, 0.05))
        q = corrected_weighted_quantile(scores, weights, test_weight, alpha=0.1)
        if np.isfinite(q) and not (scores.min() - 1e-9 <= q <= scores.max() + 1e-9):
            all_within = False
    check("all finite quantiles within score bounds (20 random trials)", all_within)


if __name__ == "__main__":
    test_weights_sum_to_one()
    test_lambda_concentrates_weight()
    test_single_group_reduces_to_split_conformal()
    test_quantile_within_score_bounds()

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
