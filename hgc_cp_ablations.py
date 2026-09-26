"""
HGC-CP component ablations.

This script isolates each component:

  (a) Equal weights (lam=0, every source weighted identically regardless of
      distance) vs. heterogeneity-guided weights (lam=3, the validated
      setting) -- if HGC-CP's improvement over pooled calibration survives
      at lam=0 too, the improvement is coming from something OTHER than
      heterogeneity guidance (e.g. just the corrected weighted-quantile
      formula itself, or the per-point/group-size normalization), not from
      distance-based weighting specifically.
  (b) A small lam grid at beta=0, to show how sensitive results are to the
      specific heterogeneity threshold, beyond the single validated point.
  (c) beta=0 vs. beta=0.5 at lam=3, isolating whether Stage 3's inflation
      term does anything (Section 4.5 reports beta=0 was selected by the
      nested search; this shows why, side by side with beta>0, rather than
      asserting it).

Both empirical coverage AND mean interval width (MPIW) are reported for
every configuration, since coverage alone does not establish practical
usefulness if achieved via very wide intervals.

Pooled and groupwise (the "target-specific baseline," using the target's
own calibration data where available) are included as fixed reference
points in every table for direct comparison.

Hierarchical partial-pooling vs. no pooling is NOT re-run here -- that
comparison already exists in the manuscript (Section 5.3.6, Table 14:
n-weighted pooled approx. vs. prior-only vs. prior+100 vs. true groupwise)
using a separate, already-validated script (hierarchical_calibration.py).
Re-deriving it here would duplicate existing, correct results rather than
add new ones.

For every target dataset below, only the OTHER six datasets' calibration
data and representations are used to compute weights and quantiles -- no
information from the target's own test set is used to select lam, beta, or
any weighting decision. This mirrors exactly how a genuinely new,
uncalibrated deployment source would be handled in practice.

USAGE: python hgc_cp_ablations.py
"""

import numpy as np

from hgc_cp_evaluation_harness import (
    load_representations, load_calibration_residuals,
    load_test_predictions_and_truth, DATASETS,
)
from hgc_cp import mmd_squared, hgc_cp_interval, pooled_quantile, groupwise_quantile, evaluate_coverage

MMD_SAMPLE_SIZE = 300
CALIB_SIZE = 100
ALPHA = 0.1
SEED = 0


def evaluate_target(target: str, lam: float, beta: float, rng, precomputed: dict) -> dict:
    """Evaluate one (target, lam, beta) combination using precomputed
    distances/calibration data (shared across the whole lam/beta grid for
    this target, so only the cheap weighting/quantile step is repeated)."""
    p = precomputed[target]
    target_test_yhat, y_true = p["yhat"], p["y_true"]

    intervals_hgc = []
    for yhat in target_test_yhat:
        lo, hi, _, _ = hgc_cp_interval(yhat, p["calib_scores"], p["calib_group_ids"],
                                        p["distances"], lam, ALPHA, beta)
        intervals_hgc.append((lo, hi))
    return evaluate_coverage(intervals_hgc, y_true)


def precompute_all_targets(rng) -> dict:
    """Distances, calibration data, and test predictions depend only on the
    target (not on lam/beta), so compute each ONCE and reuse across the
    whole ablation grid below."""
    precomputed = {}
    for target in DATASETS:
        sources = [d for d in DATASETS if d != target]
        Z_target_full = load_representations(target)
        Z_target_mmd = Z_target_full[rng.choice(len(Z_target_full), size=min(len(Z_target_full), MMD_SAMPLE_SIZE), replace=False)]

        source_scores, source_ids, distances = [], [], {}
        for s in sources:
            Zs_full = load_representations(s)
            Zs_mmd = Zs_full[rng.choice(len(Zs_full), size=min(len(Zs_full), MMD_SAMPLE_SIZE), replace=False)]
            distances[s] = mmd_squared(Z_target_mmd, Zs_mmd)
            scores_s = load_calibration_residuals(s)
            if len(scores_s) > CALIB_SIZE:
                idx = rng.choice(len(scores_s), size=CALIB_SIZE, replace=False)
                scores_s = scores_s[idx]
            source_scores.append(scores_s)
            source_ids.extend([s] * len(scores_s))
        calib_scores = np.concatenate(source_scores)
        calib_group_ids = np.array(source_ids)

        yhat, y_true = load_test_predictions_and_truth(target)

        precomputed[target] = {
            "distances": distances, "calib_scores": calib_scores,
            "calib_group_ids": calib_group_ids, "yhat": yhat, "y_true": y_true,
        }
        print(f"  Precomputed for target '{target}'.")
    return precomputed


def print_table(title: str, rows: list):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"{'Target':<8}{'Setting':<20}{'Coverage':<12}{'MPIW':<10}{'Pooled cov':<12}{'Groupwise cov'}")
    for r in rows:
        print(f"{r['target']:<8}{r['setting']:<20}{r['coverage']:<12.3f}{r['MPIW']:<10.4f}"
              f"{r['pooled_cov']:<12.3f}{r['groupwise_cov']:.3f}")


def main():
    rng = np.random.default_rng(SEED)
    print("Precomputing distances and calibration data for all 7 targets "
          f"(calib_size={CALIB_SIZE}, shared across the whole ablation grid)...")
    precomputed = precompute_all_targets(rng)

    # Fixed pooled/groupwise reference points, computed once per target.
    fixed_baselines = {}
    for target in DATASETS:
        p = precomputed[target]
        pq = pooled_quantile(p["calib_scores"], ALPHA)
        pooled_intervals = [(yh - pq, yh + pq) for yh in p["yhat"]]
        pooled_cov = evaluate_coverage(pooled_intervals, p["y_true"])["coverage"]

        best_source = min(p["distances"], key=p["distances"].get)
        gq = groupwise_quantile(p["calib_scores"], p["calib_group_ids"], best_source, ALPHA)
        if np.isfinite(gq):
            groupwise_intervals = [(yh - gq, yh + gq) for yh in p["yhat"]]
            groupwise_cov = evaluate_coverage(groupwise_intervals, p["y_true"])["coverage"]
        else:
            groupwise_cov = float("nan")
        fixed_baselines[target] = {"pooled_cov": pooled_cov, "groupwise_cov": groupwise_cov}

    # --- Ablation (a): equal weights (lam=0) vs. heterogeneity-guided (lam=3) ---
    rows_a = []
    for target in DATASETS:
        for lam, label in [(0.0, "equal weights"), (3.0, "heterogeneity-guided")]:
            result = evaluate_target(target, lam, beta=0.0, rng=rng, precomputed=precomputed)
            rows_a.append({"target": target, "setting": label, **result, **fixed_baselines[target]})
    print_table("Ablation (a): Equal weights (lam=0) vs. heterogeneity-guided weights (lam=3)", rows_a)

    # --- Ablation (b): lam sensitivity grid at beta=0 ---
    rows_b = []
    for target in DATASETS:
        for lam in [1.0, 3.0, 5.0, 10.0, 20.0]:
            result = evaluate_target(target, lam, beta=0.0, rng=rng, precomputed=precomputed)
            rows_b.append({"target": target, "setting": f"lam={lam}", **result, **fixed_baselines[target]})
    print_table("Ablation (b): Heterogeneity-threshold (lam) sensitivity at beta=0", rows_b)

    # --- Ablation (c): inflation on/off at lam=3 ---
    rows_c = []
    for target in DATASETS:
        for beta, label in [(0.0, "beta=0 (no inflation)"), (0.5, "beta=0.5 (inflation on)")]:
            result = evaluate_target(target, 3.0, beta, rng=rng, precomputed=precomputed)
            rows_c.append({"target": target, "setting": label, **result, **fixed_baselines[target]})
    print_table("Ablation (c): Stage 3 inflation on vs. off (lam=3)", rows_c)


if __name__ == "__main__":
    main()
