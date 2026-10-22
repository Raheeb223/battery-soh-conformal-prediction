"""
Nested hyperparameter search for HGC-CP's (lambda, beta), without using the target's test data.

To choose (lambda, beta) for a real target (for example CALCE), the script:
  1. takes the other six datasets only;
  2. holds each of them out in turn as a pseudo-target, calibrating on the remaining five;
  3. computes, for each candidate (lambda, beta), the average distance of coverage from the 90%
     target over the six pseudo-targets;
  4. selects the candidate with the smallest average;
  5. applies it once to the real target's test set.
With seven datasets the selection is itself noisy. MMD distances are computed once per pseudo-target
and the (lambda, beta) grid is swept on top of them.

Usage: python nested_hyperparameter_search.py --target CALCE
"""

import argparse

import numpy as np

from hgc_cp_evaluation_harness import (
    load_representations, load_calibration_residuals,
    load_test_predictions_and_truth, DATASETS,
)
from hgc_cp import mmd_squared, compute_hgc_cp_qstar, apply_hgc_cp_interval, evaluate_coverage

MMD_SAMPLE_SIZE = 300
LAM_GRID = [1.0, 3.0, 5.0, 10.0, 20.0]
BETA_GRID = [0.0, 0.25, 0.5, 1.0]
ALPHA = 0.1
CALIB_SIZE = 100  # fixed calibration budget for the search itself, for consistency


def evaluate_one_setting(pseudo_target: str, sources: list, distances: dict,
                          calib_scores: np.ndarray, calib_group_ids: np.ndarray,
                          lam: float, beta: float, rng) -> float:
    """Returns |coverage - (1-alpha)| for one pseudo-target under one (lam, beta)."""
    target_test_yhat, y_true = load_test_predictions_and_truth(pseudo_target)
    intervals = hgc_cp_intervals(target_test_yhat, calib_scores, calib_group_ids, distances, lam, beta)
    result = evaluate_coverage(intervals, y_true)
    return abs(result["coverage"] - (1 - ALPHA))


def hgc_cp_intervals(y_hats, calib_scores, calib_group_ids, distances, lam, beta):
    """HGC-CP intervals for many test points against one calibration set. The quantile does not
    depend on the test point, so it is computed once (compute_hgc_cp_qstar) and applied to each
    point; tests/check_hgc_cp_split_equivalence.py shows this is bit-identical to calling
    hgc_cp_interval per point."""
    q_star, q_t, weights = compute_hgc_cp_qstar(calib_scores, calib_group_ids, distances, lam, ALPHA, beta)
    return [apply_hgc_cp_interval(y, q_star, q_t, weights)[:2] for y in y_hats]


def nested_search(real_target: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    other_six = [d for d in DATASETS if d != real_target]

    print(f"Selecting (lam, beta) for evaluating '{real_target}', using ONLY "
          f"the other 6 datasets ({', '.join(other_six)}) as pseudo-targets. "
          f"'{real_target}' is not touched during this selection.\n")

    # Precompute each pseudo-target's distances and calibration data ONCE.
    precomputed = {}
    for pseudo_target in other_six:
        pseudo_sources = [d for d in other_six if d != pseudo_target]
        Z_pt_full = load_representations(pseudo_target)
        Z_pt_mmd = Z_pt_full[rng.choice(len(Z_pt_full), size=min(len(Z_pt_full), MMD_SAMPLE_SIZE), replace=False)]

        source_scores, source_ids, distances = [], [], {}
        for s in pseudo_sources:
            Zs_full = load_representations(s)
            Zs_mmd = Zs_full[rng.choice(len(Zs_full), size=min(len(Zs_full), MMD_SAMPLE_SIZE), replace=False)]
            distances[s] = mmd_squared(Z_pt_mmd, Zs_mmd)

            scores_s = load_calibration_residuals(s)
            if len(scores_s) > CALIB_SIZE:
                idx = rng.choice(len(scores_s), size=CALIB_SIZE, replace=False)
                scores_s = scores_s[idx]
            source_scores.append(scores_s)
            source_ids.extend([s] * len(scores_s))

        precomputed[pseudo_target] = {
            "distances": distances,
            "calib_scores": np.concatenate(source_scores),
            "calib_group_ids": np.array(source_ids),
        }
        print(f"  Precomputed distances for pseudo-target '{pseudo_target}'.")

    print(f"\nSweeping {len(LAM_GRID)}x{len(BETA_GRID)} grid across 6 pseudo-target rotations...")
    grid_results = {}
    for lam in LAM_GRID:
        for beta in BETA_GRID:
            deviations = []
            for pseudo_target in other_six:
                p = precomputed[pseudo_target]
                dev = evaluate_one_setting(
                    pseudo_target, [d for d in other_six if d != pseudo_target],
                    p["distances"], p["calib_scores"], p["calib_group_ids"],
                    lam, beta, rng,
                )
                deviations.append(dev)
            grid_results[(lam, beta)] = np.mean(deviations)
            print(f"  lam={lam:<6} beta={beta:<6} mean |coverage-90%| across 6 pseudo-targets: "
                  f"{grid_results[(lam, beta)]:.4f}")

    best_setting = min(grid_results, key=grid_results.get)
    print(f"\nBest setting by nested search: lam={best_setting[0]}, beta={best_setting[1]} "
          f"(mean deviation {grid_results[best_setting]:.4f})")

    # ---- Apply the SELECTED setting to the REAL target's ACTUAL test set ----
    print(f"\nApplying this setting to the REAL target '{real_target}' "
          f"(its test set was never used to choose lam/beta above)...")
    Z_target_full = load_representations(real_target)
    Z_target_mmd = Z_target_full[rng.choice(len(Z_target_full), size=min(len(Z_target_full), MMD_SAMPLE_SIZE), replace=False)]
    source_scores, source_ids, distances = [], [], {}
    for s in other_six:
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

    target_test_yhat, y_true = load_test_predictions_and_truth(real_target)
    intervals = hgc_cp_intervals(target_test_yhat, calib_scores, calib_group_ids, distances,
                                 best_setting[0], best_setting[1])
    result = evaluate_coverage(intervals, y_true)
    print(f"\n{real_target} coverage under nested-selected (lam={best_setting[0]}, "
          f"beta={best_setting[1]}): {result['coverage']:.3f} (target: {1-ALPHA:.0%})")
    print(f"{real_target} MPIW: {result['MPIW']:.4f}")

    return {
        "real_target": real_target,
        "pseudo_target_rotation": other_six,
        "grid_results_mean_deviation": {f"lam={l},beta={b}": v for (l, b), v in grid_results.items()},
        "selected_lam": best_setting[0],
        "selected_beta": best_setting[1],
        "selected_mean_deviation_on_pseudo_targets": grid_results[best_setting],
        "real_target_test_coverage": result["coverage"],
        "real_target_test_mpiw": result["MPIW"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="CALCE", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    nested_search(args.target, seed=args.seed)
