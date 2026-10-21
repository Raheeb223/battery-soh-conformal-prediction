"""
HGC-CP evaluation harness: pooled vs. nearest-source vs. HGC-CP,
across shrinking calibration-sample-size budgets, via a calibration-level
leave-one-dataset-out protocol -- the same validation style already used for
hierarchical_calibration.py (Section 5.3.6) in this project.

The paper's HGC-CP results come from nested_hyperparameter_search.py and
hgc_cp_validation_only.py, which use this file's three loaders; the
run_calibration_level_lodo() comparison below is an earlier exploratory
harness. The hgc_cp.py functions it imports are checked by
tests/check_hgc_cp.py and tests/check_hgc_cp_split_equivalence.py.

--- CHANGES IN THIS VERSION ---
1. RENAMED the third comparison arm from "groupwise" to "nearest_source"
   throughout (results dict keys, printed/saved columns). This is a
   calibration-level LODO protocol: the target dataset contributes ZERO of
   its own calibration data, by design -- that's the point of testing
   genuine unseen-source generalization. So this arm was never computing
   "the target's own quantile" the way predictor_consistent_recalibration.py
   or cell_clustered_bootstrap.py do; it picks whichever of the 6 OTHER
   datasets is closest by MMD distance and uses THAT dataset's own quantile
   as a stand-in for the unseen target (target_group=min(distances,
   key=distances.get)). Calling that "groupwise" caused a real, confusing
   mismatch when its numbers were compared against this project's actual
   groupwise calibration (target calibrated on its own data) -- they are
   answering different questions and were never expected to agree. The
   computation itself is unchanged; only the label is corrected. If
   nearest_source_baseline.py already computes this same quantity
   elsewhere in the project, cross-check the two against each other.
2. EFFICIENCY: pooled_quantile() and groupwise_quantile() [now
   nearest_source_quantile in spirit] depend only on all_scores /
   all_group_ids / distances, none of which change across a target's test
   windows -- they were being recomputed identically on every single
   window inside the per-window loop. Moved outside the loop so each is
   computed once per (target, calib_size) instead of once per window (for
   MIT's 17,513 test windows, this was 17,513 redundant identical calls
   per calib_size setting). Verified behavior-preserving against the
   original per-window version on synthetic data before applying here --
   same coverage, same MPIW, byte-identical. hgc_cp_interval() is NOT
   moved, since it genuinely depends on each window's own yhat.
"""

import os

import numpy as np
import pandas as pd

import config
from hgc_cp import (
    mmd_squared,
    compute_group_weights,
    hgc_cp_interval,
    pooled_quantile,
    groupwise_quantile,
    evaluate_coverage,
)


DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
CALIB_SIZES_TO_TEST = [10, 25, 50, 100, 250, None]  # None = use all available

# Inputs, located through config.py (set SOH_DATA_DIR to move the data folder).
PROCESSED_NPZ_PATH = os.path.join(config.CACHE_DIR, "processed.npz")
PREDICTIONS_NPZ_PATH = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz")
VAL_PREDICTIONS_NPZ_PATH = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz")


def load_representations(dataset_name: str, npz_path: str = PROCESSED_NPZ_PATH,
                          split: str = "train") -> np.ndarray:
    """Build a per-cell/window representation directly from processed.npz's
    raw input windows (X), since no trained-model feature extractor is
    available in this template. Returns (n_windows, 5) summary-statistic
    features per window: [mean, std, slope, start_value, end_value], all in
    DE-NORMALIZED (real SOH) units.

    IMPORTANT: X in processed.npz is z-scored SEPARATELY per dataset using
    that dataset's own soh_mean/soh_std. Computing MMD directly on this
    normalized data would make every dataset look artificially similar
    (each individually centered at mean~0, std~1) and defeat the entire
    purpose of measuring real cross-dataset heterogeneity. This function
    de-normalizes back to real SOH units first -- verified against the
    actual uploaded processed.npz (BIT window de-normalizes to a plausible
    0.74-1.07 SOH range, not a z-score range).

    If your trained model's penultimate-layer features are available and
    preferred over these summary statistics, replace this function's body
    with a call to your model's forward pass up to that layer -- the rest
    of this harness only depends on the (n, d) shape of what's returned.
    """
    data = np.load(npz_path, allow_pickle=True)
    soh_mean = data["soh_mean"].item()
    soh_std = data["soh_std"].item()

    X = data[f"X_{split}"]
    lab = data[f"lab_{split}"]
    mask = lab == dataset_name
    X_sub = X[mask][:, :, 0]  # (n, 20) -- drop the trailing singleton feature dim

    m, s = soh_mean[dataset_name], soh_std[dataset_name]
    X_real = X_sub * s + m  # de-normalize to real SOH units

    n_windows, window_len = X_real.shape
    t = np.arange(window_len)
    slopes = np.array([np.polyfit(t, row, 1)[0] for row in X_real])

    features = np.column_stack([
        X_real.mean(axis=1),
        X_real.std(axis=1),
        slopes,
        X_real[:, 0],
        X_real[:, -1],
    ])
    return features


def load_calibration_residuals(dataset_name: str, npz_path: str = PROCESSED_NPZ_PATH,
                                val_pred_path: str = VAL_PREDICTIONS_NPZ_PATH) -> np.ndarray:
    """Return |y - yhat| for this dataset's VALIDATION set, de-normalized to
    real SOH units -- these are the calibration residuals used to compute
    conformal quantiles (pooled/nearest-source/HGC-CP all draw from this
    same underlying set, just weighted/grouped differently). Uses
    conformal_groupwise_predictions_val.npz (confirmed: rows matching
    X_val exactly, same schema as the already-verified test-set predictions
    file, same z-scored-per-dataset normalization requiring the same
    de-normalization step)."""
    val_pred = np.load(val_pred_path, allow_pickle=True)
    data = np.load(npz_path, allow_pickle=True)
    soh_mean = data["soh_mean"].item()
    soh_std = data["soh_std"].item()

    mask = val_pred["lab_val"] == dataset_name
    m, s = soh_mean[dataset_name], soh_std[dataset_name]
    y_real = val_pred["y_val"][mask] * s + m
    pred_real = val_pred["mean_pred"][mask] * s + m
    return np.abs(y_real - pred_real)


def load_test_predictions_and_truth(dataset_name: str, npz_path: str = PROCESSED_NPZ_PATH,
                                     pred_path: str = PREDICTIONS_NPZ_PATH) -> tuple:
    """Return (yhat_test, y_true_test) for this dataset, de-normalized to
    real SOH units. conformal_groupwise_predictions.npz's y_test/mean_pred
    are still per-dataset z-scored (confirmed: raw range -8.6 to 1.7, not a
    plausible SOH range) -- de-normalizes using processed.npz's soh_mean/
    soh_std, same as load_representations() already does."""
    pred = np.load(pred_path, allow_pickle=True)
    data = np.load(npz_path, allow_pickle=True)
    soh_mean = data["soh_mean"].item()
    soh_std = data["soh_std"].item()

    mask = pred["lab_test"] == dataset_name
    m, s = soh_mean[dataset_name], soh_std[dataset_name]
    yhat = pred["mean_pred"][mask] * s + m
    y_true = pred["y_test"][mask] * s + m
    return yhat, y_true


def run_calibration_level_lodo(alpha: float = 0.1, lam: float = 3.0, beta: float = 0.0,
                                calib_size: int = None, seed: int = 0,
                                mmd_sample_size: int = 300):
    """For each dataset held out as 'target', use the other six as sources.
    Subsample each source's calibration RESIDUALS to `calib_size` (or use
    all available if None) to test performance under a shrinking
    calibration budget -- but MMD distance estimation uses a SEPARATE,
    independently-sized sample of representations (`mmd_sample_size`),
    regardless of calib_size. These must not be conflated: MMD's kernel
    matrix is O(n^2) in memory (a full 76,022-window dataset would need a
    76,022 x 76,022 matrix, ~43 GiB -- this crashed on first use), so it
    needs subsampling for tractability; separately, using only 10-50 points
    to estimate distributional similarity (if this were tied to a small
    calib_size) would make the distance estimate too noisy to trust,
    independent of the memory issue. mmd_sample_size=300 matches what was
    validated on this project's real representations earlier in this
    conversation (stable, interpretable distances; e.g. MIT/XJTU correctly
    identified as closest to each other).

    lam=3.0, beta=0.0 are the defaults as of the nested, held-out
    hyperparameter search (see nested_hyperparameter_search.py): selected
    using only 6 of the 7 datasets as pseudo-targets (never the one being
    evaluated), giving a genuine, non-tuned-on-test improvement over the
    earlier lam=5.0/beta=0.5 defaults (mean |coverage-90%| across the 6
    held-out pseudo-target rotations: 0.1453 vs 0.1714). This did NOT fix
    CALCE specifically (still ~23-28% coverage under either setting) --
    see hgc_cp.py's module docstring and this project's CALCE hypothesis
    test for why that failure is a specific, understood limitation (BIT is
    a good residual proxy for CALCE; RWTH, its second-closest match by
    representation, is a poor one and dilutes the signal) rather than a
    tuning problem. This was all validated on the data available before
    the CALCE loader fix -- treat it as provisional until re-checked.

    The 'nearest_source' comparison arm (see module docstring, CHANGES
    item 1) uses whichever of the 6 sources is closest to the target by
    MMD distance, NOT the target's own data -- the target contributes no
    calibration data at all in this protocol, by design."""
    rng = np.random.default_rng(seed)
    results = []

    for target in DATASETS:
        sources = [d for d in DATASETS if d != target]
        Z_target_full = load_representations(target)
        Z_target_mmd = Z_target_full[rng.choice(len(Z_target_full),
                                                 size=min(len(Z_target_full), mmd_sample_size),
                                                 replace=False)]

        source_Z_mmd, source_scores, source_group_ids = {}, [], []
        for s in sources:
            Z_s_full = load_representations(s)
            Z_s_mmd = Z_s_full[rng.choice(len(Z_s_full),
                                           size=min(len(Z_s_full), mmd_sample_size),
                                           replace=False)]
            source_Z_mmd[s] = Z_s_mmd

            scores_s = load_calibration_residuals(s)
            if calib_size is not None and len(scores_s) > calib_size:
                idx = rng.choice(len(scores_s), size=calib_size, replace=False)
                scores_s = scores_s[idx]
            source_scores.append(scores_s)
            source_group_ids.extend([s] * len(scores_s))
        all_scores = np.concatenate(source_scores)
        all_group_ids = np.array(source_group_ids)

        distances = {s: mmd_squared(Z_target_mmd, Z) for s, Z in source_Z_mmd.items()}

        # Target's own TEST set: need yhat and y_true as SEPARATE arrays (not
        # a residual score) to actually check whether y_true falls inside
        # each constructed interval.
        target_test_yhat, y_true = load_test_predictions_and_truth(target)

        # pq and the nearest-source quantile depend only on all_scores /
        # all_group_ids / distances -- none of which vary across this
        # target's test windows -- so compute each ONCE here rather than
        # redundantly inside the per-window loop below (verified
        # behavior-preserving: see module docstring, CHANGES item 2).
        pq = pooled_quantile(all_scores, alpha)
        nearest_source_group = min(distances, key=distances.get)
        nq = groupwise_quantile(all_scores, all_group_ids,
                                 target_group=nearest_source_group, alpha=alpha)

        intervals_hgc, intervals_pooled, intervals_nearest_source = [], [], []
        for yhat in target_test_yhat:
            lo, hi, _, _ = hgc_cp_interval(yhat, all_scores, all_group_ids, distances, lam, alpha, beta)
            intervals_hgc.append((lo, hi))
            intervals_pooled.append((yhat - pq, yhat + pq))
            intervals_nearest_source.append((yhat - nq, yhat + nq) if np.isfinite(nq) else (yhat, yhat))

        results.append({
            "target": target,
            "calib_size": calib_size if calib_size else "all",
            "nearest_source_group": nearest_source_group,
            **{f"hgc_{k}": v for k, v in evaluate_coverage(intervals_hgc, y_true).items()},
            **{f"pooled_{k}": v for k, v in evaluate_coverage(intervals_pooled, y_true).items()},
            **{f"nearest_source_{k}": v for k, v in evaluate_coverage(intervals_nearest_source, y_true).items()},
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    all_results = []
    for size in CALIB_SIZES_TO_TEST:
        print(f"Running calibration-level LODO at calib_size={size}...")
        df = run_calibration_level_lodo(calib_size=size)
        all_results.append(df)
    final = pd.concat(all_results, ignore_index=True)
    final.to_csv("hgc_cp_comparison_results.csv", index=False)
    print(final.to_string())