"""
Verification of the Bayesian log-scale borrowing in borrowing_variants.py.

Checks:
  1. A from-scratch brute-force posterior (written here, not imported): for the intercept-only
     normal-normal model with a flat prior on the mean and a half-Cauchy prior on tau, the
     posterior mean of one group's effect is computed by direct numerical integration over a
     (mu, tau) grid and compared with bayes_own() and bayes_new() on random inputs.
  2. Limits: a group measured with (almost) no error keeps its own value; identical groups return
     that value; the posterior weight on the group's own value lies in [0, 1].
  3. Meta-regression: when the covariate explains the group values exactly, the predicted value
     for a new group equals the fitted line at its covariate.

Synthetic data only; no project files needed. Exits with status 1 if any check fails.
Run from the repository root:  python tests/check_bayesian_borrowing.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from borrowing_variants import bayes_new, bayes_own

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def brute_force(y, s2, scale, t=None):
    """Posterior mean of theta_t (t in the data) or of a new group's theta (t=None), by
    integrating p(mu, tau | y) on a grid. Written independently of borrowing_variants.py."""
    mus = np.linspace(y.min() - 6, y.max() + 6, 1201)
    taus = np.linspace(1e-4, 8.0, 1600)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    var = s2[None, None, :] + T[..., None] ** 2
    loglik = -0.5 * np.sum(np.log(var) + (y[None, None, :] - M[..., None]) ** 2 / var, axis=-1)
    logpost = loglik + np.log(1.0 / (1.0 + (T / scale) ** 2))
    w = np.exp(logpost - logpost.max())
    w /= w.sum()
    if t is None:
        return float(np.sum(w * M))
    k = T ** 2 / (T ** 2 + s2[t])
    return float(np.sum(w * (k * y[t] + (1 - k) * M)))


def test_brute_force():
    print("1. Agreement with a brute-force posterior")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(6):
        k = rng.integers(4, 7)
        y = rng.normal(0, 1, k)
        s2 = rng.uniform(0.01, 0.5, k)
        X = np.ones((k, 1))
        for scale in (0.5, 1.0, 2.0):
            t = int(rng.integers(k))
            own, _ = bayes_own(y, s2, X, t, scale)
            worst = max(worst, abs(own - brute_force(y, s2, scale, t)))
            new = bayes_new(y, s2, X, np.ones(1), scale)
            worst = max(worst, abs(new - brute_force(y, s2, scale)))
    print(f"   largest difference over 36 comparisons: {worst:.2e}")
    check("bayes_own and bayes_new match the brute-force posterior to 2e-3", worst < 2e-3)


def test_limits():
    print("2. Limits")
    y = np.array([0.0, 0.2, -0.1, 3.0])
    X = np.ones((4, 1))
    own, w = bayes_own(y, np.array([0.1, 0.1, 0.1, 1e-8]), X, 3, 1.0)
    check("a group with almost no measurement error keeps its own value", abs(own - 3.0) < 1e-4)
    check("its posterior weight on its own value is close to 1", w > 0.999)
    same = np.full(5, 0.7)
    own, w = bayes_own(same, np.full(5, 0.2), np.ones((5, 1)), 2, 1.0)
    check("identical groups return their common value", abs(own - 0.7) < 1e-9)
    rng = np.random.default_rng(2)
    ok = True
    for _ in range(20):
        y = rng.normal(0, 1, 5)
        s2 = rng.uniform(0.01, 1.0, 5)
        _, w = bayes_own(y, s2, np.ones((5, 1)), int(rng.integers(5)), 1.0)
        ok &= 0.0 <= w <= 1.0
    check("posterior weight on the own value lies in [0, 1] (20 random settings)", ok)


def test_meta_regression():
    print("3. Meta-regression")
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    y = 0.3 + 1.7 * x
    X = np.column_stack([np.ones(4), x])
    pred = bayes_new(y, np.full(4, 1e-6), X, np.array([1.0, 1.2]), 1.0)
    check("an exactly linear covariate predicts the fitted line for a new group",
          abs(pred - (0.3 + 1.7 * 1.2)) < 1e-3)


if __name__ == "__main__":
    test_brute_force()
    test_limits()
    test_meta_regression()
    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 60}")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
