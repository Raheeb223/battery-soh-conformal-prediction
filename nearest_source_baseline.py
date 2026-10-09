"""
Additional HGC-CP baseline: nearest-source-only calibration.

Uses only the single closest source (by the same representation-space MMD
distance HGC-CP already computes) as if it were the target's own
calibration data, ignoring all other sources entirely -- a simpler
alternative to HGC-CP's weighted blend of all six sources. This tests
whether picking the single best match outperforms, matches, or underperforms
a full weighted combination, directly addressing whether HGC-CP's blending
across multiple sources adds value beyond simply identifying the closest one.

USAGE: python nearest_source_baseline.py
"""

import numpy as np

from hgc_cp_evaluation_harness import (
    load_representations, load_calibration_residuals,
    load_test_predictions_and_truth, DATASETS,
)
from hgc_cp import mmd_squared, groupwise_quantile, evaluate_coverage

MMD_SAMPLE_SIZE = 300
CALIB_SIZE = 100
ALPHA = 0.10
SEED = 0


def evaluate_nearest_source_only(target: str, rng) -> dict:
    sources = [d for d in DATASETS if d != target]
    Z_target_full = load_representations(target)
    Z_target_mmd = Z_target_full[rng.choice(len(Z_target_full), size=min(len(Z_target_full), MMD_SAMPLE_SIZE), replace=False)]

    distances, source_scores, source_ids = {}, [], []
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

    all_scores = np.concatenate(source_scores)
    all_group_ids = np.array(source_ids)
    nearest = min(distances, key=distances.get)

    q = groupwise_quantile(all_scores, all_group_ids, target_group=nearest, alpha=ALPHA)
    yhat_test, y_true = load_test_predictions_and_truth(target)
    if not np.isfinite(q):
        return {"nearest_source": nearest, "coverage": float("nan"), "MPIW": float("nan")}
    intervals = [(yh - q, yh + q) for yh in yhat_test]
    result = evaluate_coverage(intervals, y_true)
    return {"nearest_source": nearest, **result}


def main():
    rng = np.random.default_rng(SEED)
    print(f"{'Target':<8}{'Nearest source':<16}{'Coverage':<11}{'MPIW'}")
    for target in DATASETS:
        r = evaluate_nearest_source_only(target, rng)
        print(f"{target:<8}{r['nearest_source']:<16}{r['coverage']:<11.3f}{r['MPIW']:.4f}")


if __name__ == "__main__":
    main()
