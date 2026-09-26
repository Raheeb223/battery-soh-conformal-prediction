"""
HGC-CP evaluation harness: pooled vs. groupwise (Mondrian) vs. HGC-CP,
across shrinking calibration-sample-size budgets, via a calibration-level
leave-one-dataset-out protocol -- the same validation style already used for
hierarchical_calibration.py (Section 5.3.6).

Representations (Z) are built directly from cache/processed.npz's raw input
windows rather than a trained-model feature extractor, since a compact,
label-free, five-number summary (mean, std, slope, start, end value, all in
de-normalized SOH units) is sufficient to characterize a degradation window's
shape for MMD purposes and needs no access to model internals. Calibration
residuals and test-set predictions are read from this project's existing
conformal_groupwise_predictions.npz (test set) and
conformal_groupwise_predictions_val.npz (validation set, produced by
generate_val_predictions.py) -- the same underlying predictions
pooled/groupwise calibration already use elsewhere in this project.

Run this file directly to reproduce the calibration-level LODO sweep across
CALIB_SIZES_TO_TEST.
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

PROCESSED_NPZ_PATH = os.path.join(config.CACHE_DIR, "processed.npz")
PREDICTIONS_NPZ_PATH = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz")
VAL_PREDICTIONS_NPZ_PATH = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz")


def load_representations(dataset_name: str, npz_path: str = PROCESSED_NPZ_PATH,
                          split: str = "train") -> np.ndarray:
    """Build a per-window representation directly from processed.npz's raw
    input windows (X). Returns (n_windows, 5) summary-statistic features per
    window: [mean, std, slope, start_value, end_value], all in DE-NORMALIZED
    (real SOH) units.

    X in processed.npz is z-scored SEPARATELY per dataset using that
    dataset's own soh_mean/soh_std. Computing MMD directly on this normalized
    data would make every dataset look artificially similar (each
    individually centered at mean~0, std~1) and defeat the purpose of
    measuring real cross-dataset heterogeneity, so this function de-
    normalizes back to real SOH units first.
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
    conformal quantiles (pooled/groupwise/HGC-CP all draw from this same
    underlying set, just weighted/grouped differently)."""
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
    are stored per-dataset z-scored, so this de-normalizes using
    processed.npz's soh_mean/soh_std, the same as load_representations()."""
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
    matrix is O(n^2) in memory (a full ~76,000-window dataset would need a
    correspondingly large matrix), so it needs subsampling for tractability;
    separately, using only 10-50 points to estimate distributional
    similarity (if this were tied to a small calib_size) would make the
    distance estimate too noisy to trust, independent of the memory
    constraint. mmd_sample_size=300 gives stable, interpretable distances
    (e.g. MIT and XJTU are correctly identified as closest to each other).

    lam=3.0, beta=0.0 are the defaults selected by the nested, held-out
    hyperparameter search (see nested_hyperparameter_search.py): chosen
    using only 6 of the 7 datasets as pseudo-targets (never the one being
    evaluated), giving a genuine, non-tuned-on-test improvement over an
    untuned lam=5.0/beta=0.5 (mean |coverage-90%| across the 6 held-out
    pseudo-target rotations: 0.1453 vs. 0.1714). This does NOT fix CALCE
    specifically (still ~21-40% coverage under either setting) -- see
    hgc_cp.py's module docstring and this project's CALCE hypothesis test
    for why that failure is a specific, understood limitation (BIT is a good
    residual proxy for CALCE; RWTH, its second-closest match by
    representation, is a poor one and dilutes the signal) rather than a
    tuning problem."""
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

        intervals_hgc, intervals_pooled, intervals_group = [], [], []
        for yhat in target_test_yhat:
            lo, hi, _, _ = hgc_cp_interval(yhat, all_scores, all_group_ids, distances, lam, alpha, beta)
            intervals_hgc.append((lo, hi))
            pq = pooled_quantile(all_scores, alpha)
            intervals_pooled.append((yhat - pq, yhat + pq))
            gq = groupwise_quantile(all_scores, all_group_ids,
                                     target_group=min(distances, key=distances.get), alpha=alpha)
            intervals_group.append((yhat - gq, yhat + gq) if np.isfinite(gq) else (yhat, yhat))

        results.append({
            "target": target,
            "calib_size": calib_size if calib_size else "all",
            **{f"hgc_{k}": v for k, v in evaluate_coverage(intervals_hgc, y_true).items()},
            **{f"pooled_{k}": v for k, v in evaluate_coverage(intervals_pooled, y_true).items()},
            **{f"groupwise_{k}": v for k, v in evaluate_coverage(intervals_group, y_true).items()},
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
