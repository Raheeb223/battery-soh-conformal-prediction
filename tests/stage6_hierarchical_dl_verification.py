"""
Stage 6: independent verification of hierarchical_calibration_v2.py's
DerSimonian-Laird (1986) random-effects estimator.

Two checks:
  1. Hand-calculated test cases (worked by hand below, not derived from the
     module under test) -- the standard DL textbook formula applied to
     small, exactly-computable inputs.
  2. A from-scratch reimplementation (written independently in this file,
     not imported from hierarchical_calibration_v2.py) cross-checked
     against the module's dersimonian_laird() on random inputs.

This directly targets the brief's requirement: "hierarchical partial
pooling verification (DerSimonian-Laird random-effects estimator ... vs
an independent implementation)".

Run:  python tests/stage6_hierarchical_dl_verification.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hierarchical_calibration_v2 import dersimonian_laird

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def independent_dl(q, s2):
    """Written from scratch here, NOT imported -- same textbook DerSimonian-
    Laird formula, independently coded to cross-check the module's version."""
    q = np.asarray(q, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    k = len(q)
    weights_fixed = 1.0 / s2
    mu_fixed = np.sum(weights_fixed * q) / np.sum(weights_fixed)
    cochrans_q = np.sum(weights_fixed * (q - mu_fixed) ** 2)
    c_term = np.sum(weights_fixed) - np.sum(weights_fixed ** 2) / np.sum(weights_fixed)
    tau_squared = max(0.0, (cochrans_q - (k - 1)) / c_term) if c_term > 0 else 0.0
    weights_random = 1.0 / (s2 + tau_squared)
    mu_random = np.sum(weights_random * q) / np.sum(weights_random)
    return mu_random, tau_squared


def test_hand_calculated_cases():
    print("1. Hand-calculated test cases")

    # Case A: q=[1,2,3], s2=[1,1,1] (equal variances, k=3)
    # mu_fe = 2.0, Q = 1+0+1 = 2, c = 3 - 1 = 2, tau2 = max(0, (2-2)/2) = 0
    # tau2=0 -> mu_re = mu_fe = 2.0
    mu, tau2 = dersimonian_laird([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    print(f"   Case A: mu={mu:.6f} (expected 2.0), tau2={tau2:.6f} (expected 0.0)")
    check("Case A: mu matches hand calculation", abs(mu - 2.0) < 1e-9)
    check("Case A: tau2 matches hand calculation (homogeneous -> 0)", abs(tau2 - 0.0) < 1e-9)

    # Case B: q=[0,5], s2=[1,1] (k=2, heterogeneous)
    # w=[1,1], mu_fe=2.5, Q=6.25+6.25=12.5, c=2-1=1, tau2=max(0,(12.5-1)/1)=11.5
    # wr=1/(1+11.5)=0.08 both (equal) -> mu_re=mu_fe=2.5
    mu, tau2 = dersimonian_laird([0.0, 5.0], [1.0, 1.0])
    print(f"   Case B: mu={mu:.6f} (expected 2.5), tau2={tau2:.6f} (expected 11.5)")
    check("Case B: mu matches hand calculation", abs(mu - 2.5) < 1e-9)
    check("Case B: tau2 matches hand calculation (heterogeneous -> 11.5)", abs(tau2 - 11.5) < 1e-9)

    # Case C: q=[0,1], s2=[1,4] (unequal variances, Q < k-1 -> tau2 clipped to 0)
    # w=[1,0.25], mu_fe=0.2, Q=0.04+0.16=0.2, c=1.25-1.0625/1.25=0.4
    # tau2=max(0,(0.2-1)/0.4)=max(0,-2)=0 -> mu_re=mu_fe=0.2
    mu, tau2 = dersimonian_laird([0.0, 1.0], [1.0, 4.0])
    print(f"   Case C: mu={mu:.6f} (expected 0.2), tau2={tau2:.6f} (expected 0.0, clipped)")
    check("Case C: mu matches hand calculation", abs(mu - 0.2) < 1e-9)
    check("Case C: tau2 correctly clipped to 0 when Q < k-1", abs(tau2 - 0.0) < 1e-9)


def test_independent_reimplementation():
    print("\n2. Independent from-scratch reimplementation, random inputs")
    rng = np.random.default_rng(0)
    all_match = True
    for trial in range(30):
        k = rng.integers(2, 10)
        q = rng.normal(0, 2, k)
        s2 = rng.uniform(0.1, 5.0, k)
        mu_module, tau2_module = dersimonian_laird(q, s2)
        mu_indep, tau2_indep = independent_dl(q, s2)
        if abs(mu_module - mu_indep) > 1e-9 or abs(tau2_module - tau2_indep) > 1e-9:
            all_match = False
            print(f"   trial {trial}: MISMATCH module=({mu_module},{tau2_module}) "
                  f"vs independent=({mu_indep},{tau2_indep})")
    check("module's dersimonian_laird matches independent reimplementation (30 random trials)",
          all_match)


def test_shrinkage_direction():
    print("\n3. Sanity: with tau2>0, the random-effects weight correctly down-weights "
          "the low-variance (high-precision) group LESS than a naive unweighted mean would "
          "over-weight it -- i.e. random-effects mu lies between the fixed-effect mu and the "
          "simple unweighted average, moving toward the unweighted average as tau2 grows")
    q = np.array([0.0, 10.0])
    s2 = np.array([0.01, 100.0])  # group 0 is extremely precise, group 1 extremely noisy
    mu_re, tau2 = dersimonian_laird(q, s2)
    w_fe = 1.0 / s2
    mu_fe = np.sum(w_fe * q) / np.sum(w_fe)
    simple_mean = np.mean(q)
    print(f"   mu_fe (fixed-effect, dominated by precise group)={mu_fe:.4f}")
    print(f"   mu_re (random-effects, tau2={tau2:.4f})={mu_re:.4f}")
    print(f"   simple unweighted mean={simple_mean:.4f}")
    # mu_re should sit between mu_fe and the simple mean (closer to simple mean than
    # fixed-effect is, since tau2>0 inflates the noisy group's effective precision
    # relative to fixed-effect weighting, but not all the way to the naive average
    # unless tau2 dominates both sigma2 terms).
    between = min(mu_fe, simple_mean) <= mu_re <= max(mu_fe, simple_mean)
    check("random-effects mu lies between fixed-effect mu and the simple unweighted mean", between)


if __name__ == "__main__":
    test_hand_calculated_cases()
    test_independent_reimplementation()
    test_shrinkage_direction()

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
