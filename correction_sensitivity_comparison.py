"""
Direct sensitivity comparison: final MAE and conformal coverage for XJTU and
RWTH, WITH the data-quality corrections (Section on data-quality artifacts)
versus WITHOUT them -- the specific numeric comparison requested to
complement the existing row-count/reference-capacity sensitivity analysis
already in the manuscript.

REQUIRES: two already-existing prediction files, one from each condition.
Based on this project's folder structure, these should already exist as:
  WITH correction:    outputs\\conformal_groupwise_predictions.npz
  WITHOUT correction: outputs_WITHOUT_fix\\conformal_groupwise_predictions.npz
along with the matching cache files each was produced against:
  WITH correction:    cache\\processed.npz
  WITHOUT correction: cache_WITHOUT_fix\\processed.npz
(needed for de-normalization -- each cache has its OWN soh_mean/soh_std,
since removing different rows shifts each dataset's per-dataset statistics;
using one cache's normalization on the other condition's predictions would
silently produce wrong numbers.)

If your actual WITHOUT-fix prediction file has a different name or path,
edit WITHOUT_FIX_PRED_PATH and WITHOUT_FIX_CACHE_PATH below to match --
everything else in this script is generic given correct paths.

USAGE: python correction_sensitivity_comparison.py
"""

import numpy as np

WITH_FIX_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs\conformal_groupwise_predictions.npz"
WITH_FIX_CACHE_PATH = r"C:\Users\User\PycharmProjects\cache\processed.npz"

WITHOUT_FIX_PRED_PATH = r"C:\Users\User\PycharmProjects\outputs_WITHOUT_fix\conformal_groupwise_predictions.npz"
WITHOUT_FIX_CACHE_PATH = r"C:\Users\User\PycharmProjects\cache_WITHOUT_fix\processed.npz"

DATASETS_OF_INTEREST = ["XJTU", "RWTH"]  # the two datasets with documented corrections


def compute_dataset_metrics(pred_path: str, cache_path: str, dataset: str) -> dict:
    """Returns MAE and conformal coverage for one dataset, de-normalized
    using that specific run's own cache statistics."""
    pred = np.load(pred_path, allow_pickle=True)
    cache = np.load(cache_path, allow_pickle=True)
    soh_mean = cache["soh_mean"].item()
    soh_std = cache["soh_std"].item()

    mask = pred["lab_test"] == dataset
    if mask.sum() == 0:
        return None

    m, s = soh_mean[dataset], soh_std[dataset]
    y_true = pred["y_test"][mask] * s + m
    y_pred = pred["mean_pred"][mask] * s + m
    lower = pred["lower"][mask] * s + m
    upper = pred["upper"][mask] * s + m

    mae = np.mean(np.abs(y_true - y_pred))
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    mpiw = np.mean(upper - lower)
    n = mask.sum()
    n_cells = len(np.unique(pred["cell_test"][mask])) if "cell_test" in pred.files else None

    return {"n_windows": n, "n_cells": n_cells, "MAE": mae, "coverage": coverage, "MPIW": mpiw}


def main():
    print(f"{'Dataset':<8}{'Condition':<18}{'n windows':<11}{'n cells':<9}{'MAE':<10}{'Coverage':<12}{'MPIW'}")

    for dataset in DATASETS_OF_INTEREST:
        with_fix = compute_dataset_metrics(WITH_FIX_PRED_PATH, WITH_FIX_CACHE_PATH, dataset)
        without_fix = compute_dataset_metrics(WITHOUT_FIX_PRED_PATH, WITHOUT_FIX_CACHE_PATH, dataset)

        if with_fix is None or without_fix is None:
            print(f"{dataset}: could not find this dataset in one or both prediction files -- check paths/labels.")
            continue

        print(f"{dataset:<8}{'WITH correction':<18}{with_fix['n_windows']:<11}{with_fix['n_cells']:<9}"
              f"{with_fix['MAE']:<10.5f}{with_fix['coverage']:<12.3f}{with_fix['MPIW']:.4f}")
        print(f"{dataset:<8}{'WITHOUT correction':<18}{without_fix['n_windows']:<11}{without_fix['n_cells']:<9}"
              f"{without_fix['MAE']:<10.5f}{without_fix['coverage']:<12.3f}{without_fix['MPIW']:.4f}")

        mae_change = without_fix["MAE"] - with_fix["MAE"]
        cov_change = without_fix["coverage"] - with_fix["coverage"]
        print(f"{'':<8}{'Change (w/o - with)':<18}"
              f"{without_fix['n_windows']-with_fix['n_windows']:<11}{'':<9}"
              f"{mae_change:+.5f}   {cov_change:+.3f}")
        print()


if __name__ == "__main__":
    main()
