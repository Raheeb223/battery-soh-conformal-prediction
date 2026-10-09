"""
Nested hyperparameter search for HGC-CP's (lam, beta), designed specifically
to avoid tuning on the same data being reported on.

WHY THIS IS STRUCTURED AS A NESTED SEARCH:
Simply sweeping (lam, beta) and picking whatever gives CALCE the best test
coverage would be tuning-on-test -- the hyperparameters would be chosen
using the exact numbers being reported, silently inflating how good the
result looks. Instead, to choose (lam, beta) for evaluating a given REAL
target (e.g. CALCE), this script:
  1. Takes the other 6 datasets only.
  2. Rotates through them as PSEUDO-targets (holding each out in turn,
     calibrating on the remaining 5).
  3. For each candidate (lam, beta), computes average coverage-deviation-
     from-90%-target across all 6 pseudo-target rotations.
  4. Picks whichever (lam, beta) performs best on average across those 6 --
     CALCE's own test set is never touched during this selection.
  5. Only THEN applies the selected (lam, beta) to CALCE's actual held-out
     test set, exactly once.

This is a genuine, if computationally heavier, generalization estimate --
not a perfect one (7 datasets is still a small sample for this kind of
nested design), but a meaningfully more honest one than picking whatever
looks best on the real target directly.

COMPUTATIONAL NOTE: MMD/distance computation is the expensive part; the
weighting and quantile steps are cheap. This script computes each pseudo-
target's distances ONCE and sweeps the (lam, beta) grid cheaply on top of
that, rather than recomputing distances per grid point.

USAGE: python nested_hyperparameter_search.py --target CALCE
"""

import argparse

import numpy as np

from hgc_cp_evaluation_harness import (
    load_representations, load_calibration_residuals,
    load_test_predictions_and_truth, DATASETS,
)
from hgc_cp import mmd_squared, hgc_cp_interval, evaluate_coverage

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
    intervals = []
    for yhat in target_test_yhat:
        lo, hi, _, _ = hgc_cp_interval(yhat, calib_scores, calib_group_ids, distances, lam, ALPHA, beta)
        intervals.append((lo, hi))
    result = evaluate_coverage(intervals, y_true)
    return abs(result["coverage"] - (1 - ALPHA))


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
    intervals = []
    for yhat in target_test_yhat:
        lo, hi, _, _ = hgc_cp_interval(yhat, calib_scores, calib_group_ids, distances,
                                        best_setting[0], ALPHA, best_setting[1])
        intervals.append((lo, hi))
    result = evaluate_coverage(intervals, y_true)
    print(f"\n{real_target} coverage under nested-selected (lam={best_setting[0]}, "
          f"beta={best_setting[1]}): {result['coverage']:.3f} (target: {1-ALPHA:.0%})")
    print(f"{real_target} MPIW: {result['MPIW']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="CALCE", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    nested_search(args.target, seed=args.seed)
