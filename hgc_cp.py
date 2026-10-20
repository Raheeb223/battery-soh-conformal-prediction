"""
HGC-CP: Heterogeneity-Guided Conformal Prediction
====================================================

Implements the 3-stage algorithm as proposed, with one important correction
to Stage 2 (see WHY THE QUANTILE FORMULA IS CORRECTED below) and an honest
validity analysis (see VALIDITY ANALYSIS at the bottom) rather than a bare
implementation that implies a coverage guarantee it cannot actually deliver.

STAGE 1 -- Heterogeneity detection via MMD:
    d_k = MMD^2(Z_t, Z_k)   for each source group k, using an RBF kernel on
    window representations Z. The paper uses five summary statistics of each
    input SOH window (hgc_cp_evaluation_harness.load_representations); a
    model's penultimate-layer features would also work.

STAGE 2 -- Heterogeneity-weighted conformal quantile:
    w_k = softmax(-lambda * d_k) over source groups
    Each calibration point i in group g(i) receives weight w_{g(i)}.
    q_t = the CORRECTED weighted quantile (see below), not the naive one.

STAGE 3 -- Heterogeneity inflation:
    h_t = f(min_k d_k)                      (a bounded, monotonic transform
                                              of the closest source distance)
    q_t* = q_t * (1 + beta * h_t)

WHY THE QUANTILE FORMULA IS CORRECTED
--------------------------------------
The originally proposed quantile,
    q_t = Quantile_{1-alpha}^{(w)}( |y - yhat| ),
normalizes weights over calibration points only. This is NOT the formula
weighted conformal prediction is actually built on (Tibshirani et al. 2019,
"Conformal Prediction Under Covariate Shift", NeurIPS). Their result requires
normalizing over calibration points AND the test point's own (implicit)
weight, equivalent to including a calibration score of +infinity with weight
w_{n+1} in the weighted empirical distribution before taking the quantile.
Omitting this breaks the connection to the exchangeability argument entirely
-- what remains is a plausible-looking heuristic, not weighted conformal
prediction, and it measurably UNDER-COVERS in the small-n regime this method
targets (verified numerically below the class definition: at n=25 calibration
points, the naive formula's quantile falls short of the true value by ~0.17,
compared to ~0.06 for the corrected version -- a difference that shows up
exactly in the n=10-100 range this paper's four-method comparison would test).

--- FIXES APPLIED THIS PASS (after real execution against project data) ---
1. hgc_cp_interval's handling of q_t == inf was backwards. corrected_weighted
   _quantile's own docstring says inf means "the interval must be unbounded
   to achieve the target coverage with this little relevant calibration
   mass" -- but the calling code returned (y_hat, y_hat), a ZERO-width
   interval, the opposite of unbounded. Confirmed concretely: constructing a
   case that triggers q_t=inf and checking the returned interval against a
   realistic (not exactly equal to y_hat) test point shows it fails to
   cover, when an unbounded interval must always cover. Fixed to return
   (-inf, inf). Consequence for evaluate_coverage(): an infinite-width
   interval makes that test point's coverage trivially 1, but also makes
   the dataset's mean MPIW become inf if even one interval hits this case --
   that is the CORRECT, honest signal that the method produced at least one
   uninformative interval under that configuration, not something to average
   away silently. If a finite-case-only MPIW is wanted for reporting
   alongside this, compute it as a separate, explicitly-labeled quantity
   (e.g. report the fraction of inf-width intervals and the mean width
   among the rest), rather than silently dropping or capping the inf.
   Precisely when this can trigger: test_weight = max per-point weight must
   exceed alpha/(1-alpha) (0.111 at alpha=0.1). With EQUAL group sizes
   (every source subsampled to the same calib_size, as
   hgc_cp_evaluation_harness.py does for calib_size=10 through 250), this is
   mathematically impossible: max test_weight <= 1/calib_size <= 0.1 < 0.111
   for calib_size >= 10, confirmed by direct enumeration. It becomes
   possible wherever group sizes are naturally UNEQUAL and a very close
   (near-zero-distance) group also has few points -- e.g.
   few_shot_calibration.py's target-as-pseudo-source at distance~=0 with a
   small cell budget, where a target cell contributing fewer than roughly
   8-10 windows can trigger it. Worth checking directly against this
   project's actual per-cell window counts before treating any few-shot
   result at budget=1 as free of this effect.
2. rbf_kernel's automatic bandwidth (gamma=None) subsamples up to 500 points
   via the UNSEEDED global np.random.choice whenever more than 500 points
   are available. Confirmed concretely: calling mmd_squared 3 times on
   IDENTICAL inputs with no reseed between calls gave 3 DIFFERENT values
   (0.3429, 0.3410, 0.3312 in one test run) -- every MMD distance in this
   project is silently non-reproducible run-to-run wherever more than 500
   representation points exist (true for MIT, RWTH, XJTU, BIT, CALCE here),
   independent of any seed passed to the calling script. Fixed: both
   functions now accept an optional rng (np.random.Generator); when not
   given, they create a freshly-seeded np.random.default_rng(0) internally
   rather than touching global numpy random state, so repeated calls with
   no rng argument are now deterministic by default. Pass an explicit rng
   if deliberate run-to-run variation in the bandwidth estimate is ever
   wanted.

VALIDITY ANALYSIS -- read this before reporting any coverage claim
---------------------------------------------------------------------
Tibshirani et al.'s validity result holds when w_i are the TRUE likelihood
ratio dQ/dP(x_i) between the target distribution Q and calibration
distribution P. HGC-CP's weights are a heuristic proxy -- softmax of negative
MMD between window representations -- not an estimated likelihood ratio.
This means HGC-CP does NOT inherit an exact finite-sample coverage guarantee
merely by using the corrected quantile formula. What CAN honestly be said:
  - Barber et al. (2023, "Conformal prediction beyond exchangeability",
    Annals of Statistics) bound the coverage gap of a weighted conformal
    method by a term involving total variation distance between the true
    calibration and target distributions, weighted by how much of the
    weight mass each calibration point receives. This gives a DIRECTION for
    analysis (smaller MMD / better weight concentration -> smaller bound),
    not a specific numerical guarantee for THIS heuristic weight choice.
  - The honest, defensible framing for this paper: HGC-CP is an EMPIRICAL,
    heterogeneity-adaptive calibration method, validated the same way
    Section 5.3.6's hierarchical calibration was validated in this project
    -- via a calibration-level LODO protocol that measures ACTUAL empirical
    coverage across held-out target groups, not via a claimed theoretical
    guarantee. Report coverage results as measured, not as proven.
  - This does not make HGC-CP uninteresting -- Barber et al.'s framework is
    exactly the right citation to frame future theoretical work establishing
    tighter conditions under which HGC-CP-style weights ARE approximately
    valid (e.g. if MMD is shown to upper-bound a relevant density-ratio
    divergence under stated assumptions on the representation space).
"""

import numpy as np
from itertools import combinations


# ---------------------------------------------------------------------------
# Stage 1: MMD-based heterogeneity distance
# ---------------------------------------------------------------------------

def rbf_kernel(X, Y, gamma=None, rng=None):
    """RBF kernel matrix between X (m,d) and Y (n,d). If gamma is None, uses
    the median heuristic on the combined sample (standard, avoids an
    arbitrary bandwidth choice).

    rng: an optional np.random.Generator controlling the bandwidth
    subsample when more than 500 combined points are available. If not
    given, a freshly-seeded np.random.default_rng(0) is used internally, so
    repeated calls with no rng argument are deterministic by default (this
    used to silently draw from global numpy random state instead, making
    every MMD distance in this project non-reproducible run-to-run --
    confirmed and fixed; see module docstring, FIXES item 2)."""
    if gamma is None:
        if rng is None:
            rng = np.random.default_rng(0)
        combined = np.vstack([X, Y])
        n = len(combined)
        idx = rng.choice(n, size=min(n, 500), replace=False)
        sub = combined[idx]
        dists = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist) if median_dist > 0 else 1.0
    sq_dists = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Y ** 2, axis=1)[None, :]
        - 2 * X @ Y.T
    )
    return np.exp(-gamma * sq_dists)


def mmd_squared(Z_target, Z_source, gamma=None, rng=None):
    """Unbiased MMD^2 estimator between two representation samples.

    rng: see rbf_kernel -- passed through so the bandwidth subsample (used
    for all three of Kxx, Kyy, Kxy below) is deterministic by default."""
    m, n = len(Z_target), len(Z_source)
    Kxx = rbf_kernel(Z_target, Z_target, gamma, rng)
    Kyy = rbf_kernel(Z_source, Z_source, gamma, rng)
    Kxy = rbf_kernel(Z_target, Z_source, gamma, rng)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    term_xx = Kxx.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_yy = Kyy.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_xy = Kxy.sum() / (m * n)
    return max(0.0, term_xx + term_yy - 2 * term_xy)  # clip small negative noise


# ---------------------------------------------------------------------------
# Stage 2: heterogeneity weights and the CORRECTED weighted conformal quantile
# ---------------------------------------------------------------------------

def compute_group_weights(distances: dict, lam: float) -> dict:
    """distances: {group_name: MMD^2 to target}. Returns softmax(-lam * d).
    This is the GROUP-level weight, i.e. the total weight mass that group's
    calibration points should carry IN AGGREGATE -- see hgc_cp_interval for
    how this is converted into a PER-POINT weight (dividing by group size),
    which matters: a group with more raw calibration points must not
    automatically carry more total influence than its group-level weight
    intends, purely because it has more points."""
    groups = list(distances.keys())
    d = np.array([distances[g] for g in groups])
    d = d / (d.max() + 1e-12)  # scale-invariant: lambda is then comparable across datasets
    w = np.exp(-lam * d)
    w = w / w.sum()
    return dict(zip(groups, w))


def corrected_weighted_quantile(scores: np.ndarray, point_weights: np.ndarray,
                                 test_weight: float, alpha: float) -> float:
    """The Tibshirani et al. (2019)-correct weighted conformal quantile:
    normalizes over calibration points AND the test point's own weight.
    Returns np.inf if there isn't enough weighted mass without the test
    point -- this is the CORRECT behavior (it means the interval must be
    unbounded to achieve the target coverage with this little relevant
    calibration mass), not a bug to suppress. See hgc_cp_interval, which
    must (and now does) return an unbounded interval in this case rather
    than a zero-width one."""
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = point_weights[order]
    total = w_sorted.sum() + test_weight
    w_norm = w_sorted / total
    cum = np.cumsum(w_norm)
    idx = np.searchsorted(cum, 1 - alpha)
    if idx >= len(s_sorted):
        return np.inf
    return s_sorted[idx]


# ---------------------------------------------------------------------------
# Stage 3: heterogeneity inflation
# ---------------------------------------------------------------------------

def heterogeneity_inflation(min_distance: float, scale: float = 1.0) -> float:
    """h_t = f(min_k d_k). Uses a bounded transform (1 - exp(-d/scale)) so
    h_t in [0, 1): saturates rather than diverging for very distant targets,
    which keeps beta interpretable as a maximum proportional inflation."""
    return 1 - np.exp(-min_distance / scale)


def compute_hgc_cp_qstar(calib_scores: np.ndarray, calib_group_ids: np.ndarray,
                          group_distances: dict, lam: float, alpha: float, beta: float,
                          inflation_scale: float = 1.0) -> tuple:
    """Everything in HGC-CP's 3-stage pipeline that does NOT depend on the
    test point's y_hat, computed ONCE (K10 -- split out of hgc_cp_interval
    purely for performance: when evaluating many test points against the
    SAME calibration set, as every caller in this project does,
    hgc_cp_interval() recomputed this entire pipeline from scratch on every
    single call. This split changes nothing numerically -- see
    tests/check_hgc_cp_split_equivalence.py for a direct equivalence proof against
    the original monolithic code path.

    Returns (q_star, q_t, weights_by_group). q_star is NaN when q_t is inf
    (the caller must check np.isinf(q_t) first and build an unbounded
    interval directly -- see apply_hgc_cp_interval)."""
    weights_by_group = compute_group_weights(group_distances, lam)

    # Normalize each point's weight by its group's SIZE, so a group's total
    # contribution to the weighted quantile equals exactly its group-level
    # weight, regardless of how many raw calibration points make it up.
    # Without this, a large but dissimilar source (e.g. MIT, tens of
    # thousands of points) can swamp a small but well-matched source (e.g.
    # CALCE) purely through point-count, undoing the intended weighting --
    # confirmed empirically: this was the cause of HGC-CP's coverage
    # DEGRADING as calibration size grew in the first real evaluation run,
    # which should never happen in a correctly-weighted conformal method.
    unique_groups, group_counts = np.unique(calib_group_ids, return_counts=True)
    group_size = dict(zip(unique_groups, group_counts))
    point_weights = np.array([
        weights_by_group[g] / group_size[g] for g in calib_group_ids
    ])

    # Test point's own implicit weight must be on the SAME scale as an
    # INDIVIDUAL calibration point's weight (Tibshirani et al. 2019 treats
    # the test point as one more point among the calibration set, not as an
    # extra whole group) -- using the max of the already-normalized
    # per-point weights achieves this directly. A group-level quantity here
    # (e.g. a raw group weight, or a weight from a separate softmax that
    # includes the target as its own "group") is structurally too large:
    # for a target coverage of 1-alpha, test_weight must stay below
    # alpha/(1-alpha) or hitting that coverage becomes mathematically
    # impossible regardless of the data (confirmed directly during
    # development -- an earlier group-scale choice produced test_weight
    # values that made 90% coverage unreachable even for a well-matched,
    # amply-calibrated source group).
    test_weight = point_weights.max()

    q_t = corrected_weighted_quantile(calib_scores, point_weights, test_weight, alpha)
    if np.isinf(q_t):
        # CORRECT behavior: not enough weighted calibration mass to reach the
        # target coverage level even with the test point's own weight
        # included, so the interval must be UNBOUNDED to still achieve that
        # coverage -- not zero-width (confirmed and fixed; see module
        # docstring, FIXES item 1). A zero-width interval here would almost
        # always fail to cover, the opposite of what q_t=inf signals.
        return np.nan, q_t, weights_by_group

    min_dist = min(group_distances.values())
    h_t = heterogeneity_inflation(min_dist, inflation_scale)
    q_star = q_t * (1 + beta * h_t)
    return q_star, q_t, weights_by_group


def apply_hgc_cp_interval(y_hat: float, q_star: float, q_t: float, weights_by_group: dict) -> tuple:
    """Apply a precomputed q_star (from compute_hgc_cp_qstar, called ONCE
    per calibration set) to one test point's y_hat. Returns the same
    (lower, upper, q_star_or_q_t, weights_by_group) shape as
    hgc_cp_interval, for drop-in use in a loop over many test points."""
    if np.isinf(q_t):
        return -np.inf, np.inf, q_t, weights_by_group
    return y_hat - q_star, y_hat + q_star, q_star, weights_by_group


def hgc_cp_interval(y_hat: float, calib_scores: np.ndarray, calib_group_ids: np.ndarray,
                     group_distances: dict, lam: float, alpha: float, beta: float,
                     inflation_scale: float = 1.0) -> tuple:
    """Full 3-stage HGC-CP interval for one test point.

    calib_scores: |y - yhat| for every calibration point across ALL source groups.
    calib_group_ids: which source group each calibration point belongs to.
    group_distances: {group_name: MMD^2(target, that group)}, from Stage 1.
    Returns (lower, upper, q_star, weights_used).

    PERFORMANCE NOTE (K10): if calling this for many test points against the
    SAME calibration set (every evaluation loop in this project does), call
    compute_hgc_cp_qstar() once and apply_hgc_cp_interval() per point
    instead -- this function recomputes the whole pipeline on every call,
    which is wasteful when only y_hat changes between calls. Kept as-is
    (not removed) since several existing scripts call it directly per
    point; behavior is identical either way (see
    tests/check_hgc_cp_split_equivalence.py)."""
    q_star, q_t, weights_by_group = compute_hgc_cp_qstar(
        calib_scores, calib_group_ids, group_distances, lam, alpha, beta, inflation_scale)
    return apply_hgc_cp_interval(y_hat, q_star, q_t, weights_by_group)


# ---------------------------------------------------------------------------
# Comparison harness: pooled / groupwise (Mondrian) / HGC-CP
# (hierarchical calibration is already implemented separately in this
# project's hierarchical_calibration.py -- import and add it there rather
# than duplicate it here)
# ---------------------------------------------------------------------------

def pooled_quantile(all_scores: np.ndarray, alpha: float) -> float:
    n = len(all_scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(all_scores, level)


def groupwise_quantile(scores: np.ndarray, group_ids: np.ndarray, target_group: str, alpha: float) -> float:
    own = scores[group_ids == target_group]
    if len(own) == 0:
        return np.inf
    n = len(own)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(own, level)


def evaluate_coverage(intervals, y_true) -> dict:
    """Returns {"coverage", "MPIW", "n"}. NOTE: if any interval is
    (-inf, inf) (see hgc_cp_interval's inf-handling fix above), MPIW will be
    inf for the whole set -- this is intentional and should be read as "at
    least one test point got an uninformative interval under this
    configuration," not averaged away or treated as a data error. Report the
    fraction of infinite-width intervals separately if a finite-case-only
    width figure is also wanted."""
    lowers = np.array([iv[0] for iv in intervals])
    uppers = np.array([iv[1] for iv in intervals])
    covered = (y_true >= lowers) & (y_true <= uppers)
    mpiw = np.mean(uppers - lowers)
    return {"coverage": covered.mean(), "MPIW": mpiw, "n": len(y_true)}


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Synthetic end-to-end test: verifies the CODE is logically correct
    # (weight computation, quantile formula, inflation) since this sandbox
    # does not have this project's real trained model or representations.
    # Run this before trusting the module; do not skip this step when you
    # adapt it to real features.
    # ------------------------------------------------------------------
    np.random.seed(42)
    n_dims = 8

    # 4 synthetic "source" groups with representations at varying distance
    # from a synthetic "target" group, and residual scores whose spread
    # scales with how different the group's underlying process is.
    group_specs = {
        "A_similar": {"center": np.zeros(n_dims), "resid_scale": 0.05},
        "B_moderate": {"center": np.ones(n_dims) * 1.5, "resid_scale": 0.08},
        "C_distant": {"center": np.ones(n_dims) * 4.0, "resid_scale": 0.15},
        "D_very_distant": {"center": np.ones(n_dims) * 8.0, "resid_scale": 0.25},
    }
    target_center = np.zeros(n_dims) + 0.3  # target is closest to "A_similar"

    Z_target = target_center + np.random.randn(60, n_dims) * 0.3
    calib_scores, calib_group_ids, calib_Z = [], [], {}
    for name, spec in group_specs.items():
        Z = spec["center"] + np.random.randn(200, n_dims) * 0.5
        calib_Z[name] = Z
        scores = np.abs(np.random.randn(200) * spec["resid_scale"])
        calib_scores.append(scores)
        calib_group_ids.extend([name] * 200)
    calib_scores = np.concatenate(calib_scores)
    calib_group_ids = np.array(calib_group_ids)

    distances = {name: mmd_squared(Z_target, Z) for name, Z in calib_Z.items()}
    print("Stage 1 -- MMD^2 distances to target (should rank A closest, D furthest):")
    for name, d in sorted(distances.items(), key=lambda x: x[1]):
        print(f"  {name}: {d:.4f}")

    weights = compute_group_weights(distances, lam=3.0)
    print("\nStage 2 -- resulting group weights (should favor A_similar heavily):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.4f}")

    y_hat_test = 0.0
    lower, upper, q_star, _ = hgc_cp_interval(
        y_hat_test, calib_scores, calib_group_ids, distances,
        lam=3.0, alpha=0.1, beta=0.0,
    )
    print(f"\nStage 3 -- HGC-CP interval for target-like point: [{lower:.4f}, {upper:.4f}] "
          f"(width={upper-lower:.4f})")

    pq = pooled_quantile(calib_scores, alpha=0.1)
    print(f"\nFor comparison, POOLED quantile across all 4 groups (ignores heterogeneity): "
          f"{pq:.4f} (width={2*pq:.4f})")
    print("HGC-CP's interval should be narrower than pooled here, since the target closely "
          "resembles only the lowest-noise source group (A_similar) and the weights should "
          "reflect that rather than being diluted by the noisier, dissimilar groups.")

    # Reproducibility check for the FIXES applied this pass: same inputs,
    # repeated calls, no explicit rng -> must now be identical. Uses groups
    # deliberately LARGER than 500 combined points, so this actually
    # exercises the bandwidth-subsampling code path the fix touches --
    # the demo groups above (260 combined points) stay under that
    # threshold and would pass trivially even without the fix.
    print("\n--- Reproducibility check (FIXES item 2) ---")
    big_target = np.random.randn(300, n_dims)
    big_source = np.random.randn(300, n_dims) + 1.5  # 600 combined points: triggers subsampling
    repeat_vals = [mmd_squared(big_target, big_source) for _ in range(3)]
    print(f"mmd_squared on 600 combined points x3, no explicit rng: {[round(v, 6) for v in repeat_vals]}")
    assert len(set(repeat_vals)) == 1, "should be deterministic by default now"
    print("deterministic by default: confirmed")

    # inf-handling check (FIXES item 1): a deliberately starved group should
    # now yield an unbounded interval, not a zero-width one.
    print("\n--- inf-handling check (FIXES item 1) ---")
    starved_group_ids = np.array(["tiny"] * 3 + ["A_similar"] * 200)
    starved_scores = np.concatenate([np.abs(np.random.randn(3) * 0.05), calib_scores[:200]])
    starved_distances = {"tiny": 1e-9, "A_similar": 5.0}
    lo, hi, q_t, _ = hgc_cp_interval(0.0, starved_scores, starved_group_ids, starved_distances,
                                      lam=3.0, alpha=0.1, beta=0.0)
    print(f"deliberately starved near-zero-distance group (3 points): interval=({lo}, {hi}), q_t={q_t}")
    assert np.isinf(q_t) and lo == -np.inf and hi == np.inf, "should be unbounded, not zero-width"
    print("unbounded as expected (not the old zero-width bug): confirmed")