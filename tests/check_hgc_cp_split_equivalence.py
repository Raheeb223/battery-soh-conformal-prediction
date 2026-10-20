"""
Equivalence test for the fast HGC-CP path.

hgc_cp.py computes the weighted quantile once per calibration set (compute_hgc_cp_qstar) and then
applies it per test point (apply_hgc_cp_interval), instead of recomputing everything for each point
(hgc_cp_interval). This test checks that the fast path gives bit-identical intervals to the
original computation across 250 (test point, random configuration) combinations, including a
configuration that deliberately triggers the infinite-width case.

Synthetic data only; no project files needed. Exits with status 1 if any check fails.
Run from the repository root:  python tests/check_hgc_cp_split_equivalence.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hgc_cp import hgc_cp_interval, compute_hgc_cp_qstar, apply_hgc_cp_interval

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def test_equivalence_random_configs():
    print("1. Split path matches monolithic hgc_cp_interval() across random configs")
    rng = np.random.default_rng(0)
    all_match = True
    n_mismatches = 0
    for trial in range(50):
        n_groups = rng.integers(2, 6)
        groups = [f"g{i}" for i in range(n_groups)]
        distances = {g: float(rng.uniform(0, 10)) for g in groups}
        lam = float(rng.uniform(0.5, 20))
        beta = float(rng.uniform(0, 2))
        alpha = 0.1

        sizes = rng.integers(5, 100, n_groups)  # deliberately uneven, can trigger inf
        calib_scores = np.concatenate([np.abs(rng.normal(0, 1, s)) for s in sizes])
        calib_group_ids = np.concatenate([[g] * s for g, s in zip(groups, sizes)])

        # Precompute once (new path)
        q_star, q_t, weights_by_group = compute_hgc_cp_qstar(
            calib_scores, calib_group_ids, distances, lam, alpha, beta)

        for _ in range(5):  # multiple test points against the same calibration set
            y_hat = float(rng.normal(0, 1))

            lo_new, hi_new, q_new, w_new = apply_hgc_cp_interval(y_hat, q_star, q_t, weights_by_group)
            lo_old, hi_old, q_old, w_old = hgc_cp_interval(
                y_hat, calib_scores, calib_group_ids, distances, lam, alpha, beta)

            same_lo = (lo_new == lo_old) or (np.isnan(lo_new) and np.isnan(lo_old))
            same_hi = (hi_new == hi_old) or (np.isnan(hi_new) and np.isnan(hi_old))
            same_q = (q_new == q_old) or (np.isnan(q_new) and np.isnan(q_old))
            same_w = w_new == w_old
            if not (same_lo and same_hi and same_q and same_w):
                all_match = False
                n_mismatches += 1
                print(f"   trial {trial}: MISMATCH new=({lo_new},{hi_new},{q_new}) "
                      f"old=({lo_old},{hi_old},{q_old})")

    print(f"   250 (test point, config) combinations checked, {n_mismatches} mismatches")
    check("split path bit-identical to monolithic path across 250 combinations", all_match)


def test_inf_case_reachable_and_equivalent():
    print("\n2. inf-width edge case: split path matches monolithic path there too")
    # Deliberately starved near-zero-distance group (same scenario as the
    # module's own built-in inf-handling check).
    calib_group_ids = np.array(["tiny"] * 3 + ["normal"] * 200)
    calib_scores = np.concatenate([np.abs(np.random.default_rng(1).normal(0, 0.05, 3)),
                                    np.abs(np.random.default_rng(2).normal(0, 1, 200))])
    distances = {"tiny": 1e-9, "normal": 5.0}

    q_star, q_t, weights_by_group = compute_hgc_cp_qstar(
        calib_scores, calib_group_ids, distances, lam=3.0, alpha=0.1, beta=0.0)
    lo_new, hi_new, q_new, _ = apply_hgc_cp_interval(0.0, q_star, q_t, weights_by_group)
    lo_old, hi_old, q_old, _ = hgc_cp_interval(
        0.0, calib_scores, calib_group_ids, distances, lam=3.0, alpha=0.1, beta=0.0)

    check("inf case actually triggered in this test (q_t is inf)", np.isinf(q_t))
    check("split path matches monolithic path in the inf case",
          lo_new == lo_old and hi_new == hi_old and q_new == q_old)


if __name__ == "__main__":
    test_equivalence_random_configs()
    test_inf_case_reachable_and_equivalent()

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
