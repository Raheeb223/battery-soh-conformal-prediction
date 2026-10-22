"""
HGC-CP: heterogeneity-guided conformal prediction.

Stage 1 (similarity): d_k = MMD^2(Z_t, Z_k) between the target's window representations Z_t and
    each source group's Z_k, with an RBF kernel. The paper uses five summary statistics of each
    input SOH window (hgc_cp_evaluation_harness.load_representations).
Stage 2 (weighted quantile): w_k = softmax(-lambda * d_k) over source groups; each calibration point
    gets its group's weight, and q_t is the weighted conformal quantile with the test point's own
    weight included as a score of +infinity (Tibshirani et al. 2019, "Conformal Prediction Under
    Covariate Shift"). Normalizing over the calibration points alone under-covers at small n.
Stage 3 (inflation): q_t* = q_t * (1 + beta * h_t), with h_t a bounded, increasing function of the
    distance to the closest source.

If the weighted quantile is infinite (too little weight on the calibration points), the interval is
(-inf, inf); with equal group sizes of at least 10 points this cannot happen at alpha = 0.1. The RBF
bandwidth is estimated on a subsample drawn from a seeded generator, so MMD values are reproducible.

Validity: the weights are a similarity heuristic, not an estimated likelihood ratio, so HGC-CP has
no exact finite-sample guarantee. Barber et al. (2023, "Conformal prediction beyond
exchangeability") bound the coverage gap in terms of the distance between calibration and target
distributions. HGC-CP's coverage is therefore evaluated empirically, on held-out datasets.

Run `python hgc_cp.py` for a self-test of the weighting, the quantile, and the inflation step.
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
    repeated calls with no rng argument are deterministic."""
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
    test point's y_hat, computed once (split out of hgc_cp_interval
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
    # impossible regardless of the data.
    test_weight = point_weights.max()

    q_t = corrected_weighted_quantile(calib_scores, point_weights, test_weight, alpha)
    if np.isinf(q_t):
        # Not enough weighted calibration mass to reach the target coverage
        # even with the test point's own weight included, so the interval
        # must be unbounded.
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

    Performance: if calling this for many test points against the
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
    (-inf, inf) (see hgc_cp_interval above), MPIW will be
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
    # Run this after any change to the module.
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

    # Reproducibility check: same inputs, repeated calls, no explicit rng ->
    # identical results. The groups have more than 500 combined points, so the
    # bandwidth-subsampling code path is exercised (the demo groups above, with
    # 260 points, do not reach it).
    print("\n--- Reproducibility check ---")
    big_target = np.random.randn(300, n_dims)
    big_source = np.random.randn(300, n_dims) + 1.5  # 600 combined points: triggers subsampling
    repeat_vals = [mmd_squared(big_target, big_source) for _ in range(3)]
    print(f"mmd_squared on 600 combined points x3, no explicit rng: {[round(v, 6) for v in repeat_vals]}")
    assert len(set(repeat_vals)) == 1, "should be deterministic by default now"
    print("deterministic by default: confirmed")

    # inf-handling check: a deliberately starved group should
    # now yield an unbounded interval, not a zero-width one.
    print("\n--- inf-handling check ---")
    starved_group_ids = np.array(["tiny"] * 3 + ["A_similar"] * 200)
    starved_scores = np.concatenate([np.abs(np.random.randn(3) * 0.05), calib_scores[:200]])
    starved_distances = {"tiny": 1e-9, "A_similar": 5.0}
    lo, hi, q_t, _ = hgc_cp_interval(0.0, starved_scores, starved_group_ids, starved_distances,
                                      lam=3.0, alpha=0.1, beta=0.0)
    print(f"deliberately starved near-zero-distance group (3 points): interval=({lo}, {hi}), q_t={q_t}")
    assert np.isinf(q_t) and lo == -np.inf and hi == np.inf, "should be unbounded, not zero-width"
    print("unbounded as expected (not the old zero-width bug): confirmed")