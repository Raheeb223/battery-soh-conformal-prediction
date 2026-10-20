"""
HGC-CP with every design choice made from training and validation data only.

Why. HGC-CP's original design (its representation, weighting, widening term and search grid) was
settled in development rounds that looked at main-split test results, and its nested
(lambda, beta) search (nested_hyperparameter_search.py) scores each stand-in dataset on that
dataset's TEST windows. Neither reads the real target's test data, but together they mean the
original HGC-CP numbers are not fully held-out. This script removes both problems.

  1. The design space below is declared in advance and lets the search switch each HGC-CP
     component off: lambda = 0 gives every other dataset the same weight (no similarity
     weighting), beta = 0 gives no widening. Three window representations are offered: the five
     summary statistics of the original method, mean and slope only, and the raw 20-cycle window.
  2. The choice within that space uses the same rotation as the nested search (each other
     dataset in turn is a stand-in target, calibrated from the remaining ones), but every
     stand-in is scored on its VALIDATION windows. Calibration residuals come from validation
     windows and representations from training windows, as in the original method. No test
     window of any dataset is read until the chosen design is applied, once, to the real
     target's test windows.

The quantile code is unchanged (hgc_cp.compute_hgc_cp_qstar), as are the per-source calibration
subsample (100 residuals) and the MMD sample (300 windows). For comparison it also computes each
target's own-data (groupwise) quantile from all of its validation residuals, in the same units.
It runs on the main split and on the four further splits (43-46); nothing is retrained.

Requires: cache/processed.npz and outputs/conformal_groupwise_predictions{,_val}.npz (main split);
          cache/processed_split{43..46}.npz and outputs/split{43..46}_groupwise_predictions{,_val}.npz
          (repeated_split_experiment.py, split_replication.py).
Output:   outputs/hgc_cp_validation_only.json
Run:      python hgc_cp_validation_only.py
"""
import json
import os

import numpy as np

import config
import hgc_cp_evaluation_harness as harness
from hgc_cp import compute_hgc_cp_qstar, mmd_squared, pooled_quantile

ALPHA = 0.1
MMD_SAMPLE_SIZE = 300  # as in nested_hyperparameter_search.py
CALIB_SIZE = 100       # as in nested_hyperparameter_search.py
REPRESENTATIONS = ["summary5", "mean_slope", "window"]
LAM_GRID = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0]
BETA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
DATASETS = harness.DATASETS
FIVE = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]
SPLITS = ["main", 43, 44, 45, 46]


def split_paths(s):
    if s == "main":
        return (os.path.join(config.CACHE_DIR, "processed.npz"),
                os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"),
                os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"))
    return (os.path.join(config.CACHE_DIR, f"processed_split{s}.npz"),
            os.path.join(config.OUTPUT_DIR, f"split{s}_groupwise_predictions_val.npz"),
            os.path.join(config.OUTPUT_DIR, f"split{s}_groupwise_predictions.npz"))


def representations(cache, d, cache_path):
    """The three candidate representations of dataset d's training windows, in real SOH units."""
    summary5 = harness.load_representations(d, npz_path=cache_path)  # the original method's features
    m, s = cache["soh_mean"].item()[d], cache["soh_std"].item()[d]
    window = cache["X_train"][cache["lab_train"] == d][:, :, 0] * s + m
    return {"summary5": summary5, "mean_slope": summary5[:, [0, 2]], "window": window}


def real_units(pred, key_y, key_lab, cache, d):
    m, s = cache["soh_mean"].item()[d], cache["soh_std"].item()[d]
    mask = pred[key_lab] == d
    return pred["mean_pred"][mask] * s + m, pred[key_y][mask] * s + m


def coverage(yhat, y, q):
    return float(np.mean(np.abs(y - yhat) <= q))


def run_split(s):
    cache_path, val_path, test_path = split_paths(s)
    cache = np.load(cache_path, allow_pickle=True)
    val = np.load(val_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    usable = [d for d in DATASETS if (val["lab_val"] == d).any() and (test["lab_test"] == d).any()]
    rng = np.random.default_rng(0)

    val_pred, test_pred, calib, own_scores, Z = {}, {}, {}, {}, {}
    for d in usable:
        val_pred[d] = real_units(val, "y_val", "lab_val", cache, d)
        test_pred[d] = real_units(test, "y_test", "lab_test", cache, d)
        own_scores[d] = np.abs(val_pred[d][1] - val_pred[d][0])
        calib[d] = (own_scores[d] if len(own_scores[d]) <= CALIB_SIZE
                    else own_scores[d][rng.choice(len(own_scores[d]), CALIB_SIZE, replace=False)])
        reps = representations(cache, d, cache_path)
        n = len(reps["summary5"])
        idx = rng.choice(n, size=min(n, MMD_SAMPLE_SIZE), replace=False)
        Z[d] = {k: v[idx] for k, v in reps.items()}

    # MMD^2 from every (stand-in or real) target to every source, for each representation.
    mmd = {k: {(a, b): mmd_squared(Z[a][k], Z[b][k]) for a in usable for b in usable if a != b}
           for k in REPRESENTATIONS}

    def q_star(target, sources, rep, lam, beta):
        scores = np.concatenate([calib[x] for x in sources])
        ids = np.concatenate([[x] * len(calib[x]) for x in sources])
        q, q_t, _ = compute_hgc_cp_qstar(scores, ids, {x: mmd[rep][(target, x)] for x in sources},
                                         lam, ALPHA, beta)
        return np.inf if np.isinf(q_t) else q

    per_target = {}
    for g in usable:
        others = [d for d in usable if d != g]
        best = None
        for rep in REPRESENTATIONS:
            for lam in LAM_GRID:
                for beta in BETA_GRID:
                    devs = []
                    for p in others:
                        q = q_star(p, [x for x in others if x != p], rep, lam, beta)
                        devs.append(abs(coverage(*val_pred[p], q) - (1 - ALPHA)))
                    score = float(np.mean(devs))
                    # Strict improvement only, so ties go to the earlier (simpler) setting.
                    if best is None or score < best[0] - 1e-12:
                        best = (score, rep, lam, beta)
        score, rep, lam, beta = best
        # The real target's test windows are read here, once, with the design fixed.
        q = q_star(g, others, rep, lam, beta)
        q_own = pooled_quantile(own_scores[g], ALPHA)
        per_target[g] = {
            "representation": rep, "lam": lam, "beta": beta, "validation_mean_abs_dev": score,
            "coverage": coverage(*test_pred[g], q), "half_width": float(q),
            "own_coverage": coverage(*test_pred[g], q_own), "own_half_width": float(q_own),
            "n_test": int(len(test_pred[g][0]))}
        t = per_target[g]
        print(f"  {g:7s} chose {rep:10s} lam={lam:<5} beta={beta:<5} -> coverage {100 * t['coverage']:5.1f}% "
              f"(own {100 * t['own_coverage']:5.1f}%), width x{t['half_width'] / t['own_half_width']:.2f} of own")

    five = [d for d in FIVE if d in per_target]
    summary = {"mean_abs_dev_five_validation_only_hgc_cp": float(np.mean([abs(per_target[d]["coverage"] - 0.9) for d in five])),
               "mean_abs_dev_five_own": float(np.mean([abs(per_target[d]["own_coverage"] - 0.9) for d in five]))}
    return {"datasets_used": usable, "per_target": per_target, **summary}


def main():
    results = {"design_space": {"representations": REPRESENTATIONS, "lam": LAM_GRID, "beta": BETA_GRID,
                                "calib_size": CALIB_SIZE, "mmd_sample_size": MMD_SAMPLE_SIZE},
               "per_split": {}}
    for s in SPLITS:
        print(f"\nsplit {s}")
        results["per_split"][str(s)] = run_split(s)
    path = os.path.join(config.OUTPUT_DIR, "hgc_cp_validation_only.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nMean |coverage - 90%| (points) on {', '.join(FIVE)}")
    print(f"{'split':8s}{'own data':>12s}{'HGC-CP (val only)':>20s}")
    for s in SPLITS:
        r = results["per_split"][str(s)]
        print(f"{str(s):8s}{100 * r['mean_abs_dev_five_own']:12.2f}"
              f"{100 * r['mean_abs_dev_five_validation_only_hgc_cp']:20.2f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
