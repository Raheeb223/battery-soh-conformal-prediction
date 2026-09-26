"""
Few-shot target calibration.

Section 5.3.6 (hierarchical calibration) and Section 5.3.7 (HGC-CP) both
report the two extremes -- zero target-specific calibration data (pure
cross-dataset borrowing) and full target calibration data (standard
groupwise). This script fills in the range between them, since a genuinely
new deployment source is more likely to have a FEW cells of its own data
than none at all.

DESIGN: for each target dataset, at each calibration budget (0, 1, 3, 5, 10
target-owned cells, then the full available calibration set), the target's
own sampled cells are added as an ADDITIONAL pseudo-source with distance=0
(trivially the closest possible match to itself) into HGC-CP's EXISTING
weighting machinery -- this reuses hgc_cp_interval() exactly as already
validated, rather than inventing new blending logic. At budget=0, this
pseudo-source is simply absent, exactly matching the existing genuinely-
unseen-source evaluation. At full budget, weight naturally concentrates
almost entirely on the target's own (zero-distance) data, recovering
something close to standard groupwise calibration.

REPEATED CELL-LEVEL SPLITS: which specific cells happen to be drawn at a
small budget (e.g. 1 cell) can matter a lot -- a single draw could show a
misleadingly good or bad result. Each budget is evaluated across
N_REPEATS independent random draws of which cells are selected, reporting
mean and standard deviation, not a single run.

Both coverage AND MPIW are reported at every budget, matching the same
practice already used in hgc_cp_ablations.py.

USAGE: python few_shot_calibration.py
"""

import numpy as np

from hgc_cp_evaluation_harness import (
    load_representations, load_calibration_residuals,
    load_test_predictions_and_truth, DATASETS,
    PROCESSED_NPZ_PATH, VAL_PREDICTIONS_NPZ_PATH,
)
from hgc_cp import mmd_squared, hgc_cp_interval, pooled_quantile, evaluate_coverage

MMD_SAMPLE_SIZE = 300
CALIB_SIZE_SOURCES = 100  # calibration budget for the 6 OTHER (source) datasets, held fixed
ALPHA = 0.1
LAM = 3.0   # the validated setting from Section 4.5
BETA = 0.0
CELL_BUDGETS = [0, 1, 3, 5, 10, "all"]
N_REPEATS = 10
SEED = 0


def get_target_cells_and_residuals(target: str, val_pred_path=VAL_PREDICTIONS_NPZ_PATH):
    """Returns (unique_cell_ids, per-cell residual arrays) for the target's
    OWN validation data -- this is what a genuinely new deployment source
    calibrating with a few of its own cells would actually have."""
    val_pred = np.load(val_pred_path, allow_pickle=True)
    processed = np.load(PROCESSED_NPZ_PATH, allow_pickle=True)
    soh_mean = processed["soh_mean"].item()
    soh_std = processed["soh_std"].item()

    mask = val_pred["lab_val"] == target
    cell_ids = val_pred["cell_val"][mask]
    m, s = soh_mean[target], soh_std[target]
    y_real = val_pred["y_val"][mask] * s + m
    pred_real = val_pred["mean_pred"][mask] * s + m
    residuals = np.abs(y_real - pred_real)

    unique_cells = np.unique(cell_ids)
    cell_to_residuals = {c: residuals[cell_ids == c] for c in unique_cells}
    return unique_cells, cell_to_residuals


def precompute_sources(target: str, rng) -> dict:
    """Distances and calibration data for the 6 OTHER datasets -- identical
    in spirit to the main harness's precompute step, held fixed across all
    cell-budget conditions for this target."""
    sources = [d for d in DATASETS if d != target]
    Z_target_full = load_representations(target)
    Z_target_mmd = Z_target_full[rng.choice(len(Z_target_full), size=min(len(Z_target_full), MMD_SAMPLE_SIZE), replace=False)]

    source_scores, source_ids, distances = [], [], {}
    for s in sources:
        Zs_full = load_representations(s)
        Zs_mmd = Zs_full[rng.choice(len(Zs_full), size=min(len(Zs_full), MMD_SAMPLE_SIZE), replace=False)]
        distances[s] = mmd_squared(Z_target_mmd, Zs_mmd)
        scores_s = load_calibration_residuals(s)
        if len(scores_s) > CALIB_SIZE_SOURCES:
            idx = rng.choice(len(scores_s), size=CALIB_SIZE_SOURCES, replace=False)
            scores_s = scores_s[idx]
        source_scores.append(scores_s)
        source_ids.extend([s] * len(scores_s))

    return {
        "distances": distances,
        "source_scores": np.concatenate(source_scores),
        "source_ids": np.array(source_ids),
    }


def evaluate_budget(target: str, budget, unique_cells, cell_to_residuals,
                     source_data: dict, yhat_test, y_true, rng) -> dict:
    """Adds `budget` randomly-sampled target-owned cells as a distance=0
    pseudo-source (or uses ALL available target cells if budget='all'),
    then evaluates HGC-CP and a simple pooled-with-target-data baseline."""
    if budget == "all" or (isinstance(budget, int) and budget >= len(unique_cells)):
        sampled_cells = unique_cells
    elif budget == 0:
        sampled_cells = np.array([])
    else:
        sampled_cells = rng.choice(unique_cells, size=budget, replace=False)

    if len(sampled_cells) > 0:
        target_residuals = np.concatenate([cell_to_residuals[c] for c in sampled_cells])
        distances = dict(source_data["distances"])
        distances["__target_self__"] = 1e-9  # effectively zero distance to itself
        calib_scores = np.concatenate([source_data["source_scores"], target_residuals])
        calib_group_ids = np.concatenate([
            source_data["source_ids"],
            np.array(["__target_self__"] * len(target_residuals)),
        ])
    else:
        distances = source_data["distances"]
        calib_scores = source_data["source_scores"]
        calib_group_ids = source_data["source_ids"]

    intervals_hgc = []
    for yh in yhat_test:
        lo, hi, _, _ = hgc_cp_interval(yh, calib_scores, calib_group_ids, distances, LAM, ALPHA, BETA)
        intervals_hgc.append((lo, hi))
    hgc_result = evaluate_coverage(intervals_hgc, y_true)

    pq = pooled_quantile(calib_scores, ALPHA)
    pooled_intervals = [(yh - pq, yh + pq) for yh in yhat_test]
    pooled_result = evaluate_coverage(pooled_intervals, y_true)

    return {"hgc_coverage": hgc_result["coverage"], "hgc_MPIW": hgc_result["MPIW"],
            "pooled_coverage": pooled_result["coverage"], "pooled_MPIW": pooled_result["MPIW"],
            "n_cells_used": len(sampled_cells)}


def main():
    rng = np.random.default_rng(SEED)
    print(f"{'Target':<8}{'Budget':<10}{'HGC cov (mean±sd)':<22}{'HGC MPIW':<12}{'Pooled cov':<14}{'Pooled MPIW'}")

    for target in DATASETS:
        unique_cells, cell_to_residuals = get_target_cells_and_residuals(target)
        source_data = precompute_sources(target, rng)
        yhat_test, y_true = load_test_predictions_and_truth(target)

        for budget in CELL_BUDGETS:
            if budget != 0 and budget != "all" and budget > len(unique_cells):
                continue  # skip budgets exceeding what this dataset actually has

            n_repeats_here = 1 if (budget == 0 or budget == "all") else N_REPEATS
            hgc_covs, hgc_mpiws, pool_covs, pool_mpiws = [], [], [], []
            for _ in range(n_repeats_here):
                r = evaluate_budget(target, budget, unique_cells, cell_to_residuals,
                                     source_data, yhat_test, y_true, rng)
                hgc_covs.append(r["hgc_coverage"])
                hgc_mpiws.append(r["hgc_MPIW"])
                pool_covs.append(r["pooled_coverage"])
                pool_mpiws.append(r["pooled_MPIW"])

            label = f"{budget} cells" if isinstance(budget, int) else f"all ({len(unique_cells)})"
            cov_str = f"{np.mean(hgc_covs):.3f}±{np.std(hgc_covs):.3f}" if n_repeats_here > 1 else f"{np.mean(hgc_covs):.3f}"
            print(f"{target:<8}{label:<10}{cov_str:<22}{np.mean(hgc_mpiws):<12.4f}"
                  f"{np.mean(pool_covs):<14.3f}{np.mean(pool_mpiws):.4f}")
        print()


if __name__ == "__main__":
    main()
