"""
Predictor-consistent recalibration of pooled and groupwise conformal
intervals.

WHY THIS EXISTS. In train.py, conformal quantiles are calibrated on
DETERMINISTIC validation predictions (model.eval(), predict_in_batches), but
the reported test intervals are centred on the MC-Dropout MEAN prediction
(mc_dropout_predict). Calibration scores and test scores therefore come from
different predictors, which departs from the split-conformal assumption that
calibration and test nonconformity scores are exchangeable. Here both use the
MC-Dropout-mean predictor: conformal_groupwise_predictions_val.npz (validation,
MC-Dropout mean) is used to calibrate, and conformal_groupwise_predictions.npz
(test, MC-Dropout mean) is used to evaluate.

"POOLED" HAS TWO MEANINGS AND BOTH ARE REPORTED. The pipeline z-scores SOH per
dataset (training-cell mean/std), so a pooled quantile computed on z-scored
residuals is a single number in per-dataset standard-deviation units; the same
z-quantile corresponds to a different physical half-width in each dataset.
"Pooled (real units)" pools residuals in physical SOH units instead, so one
physical half-width applies to every dataset. Results can differ substantially.

Nothing here retrains a model. Inputs: cache/processed.npz (normalization
statistics), outputs/conformal_groupwise_predictions_val.npz,
outputs/conformal_groupwise_predictions.npz.

Run from the project root:
    python predictor_consistent_recalibration.py
Output:
    outputs/predictor_consistent_recalibration.json
"""

import json
import os

import numpy as np
from scipy.stats import beta

import config

ALPHA = getattr(config, "CONFORMAL_ALPHA", 0.10)
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]


def finite_sample_quantile(residuals, alpha=ALPHA):
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level))


def clopper_pearson(k, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    val = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
    test = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    std = cache["soh_std"].item()

    r_val = np.abs(val["y_val"] - val["mean_pred"])
    lab_val = val["lab_val"]
    r_test = np.abs(test["y_test"] - test["mean_pred"])
    lab_test = test["lab_test"]

    r_val_real = np.array([r_val[i] * std[lab_val[i]] for i in range(len(r_val))])
    q_pooled_z = finite_sample_quantile(r_val)
    q_pooled_real = finite_sample_quantile(r_val_real)

    out = {"alpha": ALPHA, "q_pooled_z": q_pooled_z, "q_pooled_real": q_pooled_real, "per_dataset": {}}
    print(f"pooled quantile: z-units = {q_pooled_z:.4f}, real SOH units = {q_pooled_real:.5f}\n")
    print(f"{'Dataset':8s}{'n_test':>8s}{'groupwise':>11s}{'95% CP CI':>20s}{'pooled(z)':>11s}{'pooled(real)':>14s}")
    for d in DATASETS:
        mt, mv = lab_test == d, lab_val == d
        n = int(mt.sum())
        q_g = finite_sample_quantile(r_val[mv])
        cov_g = float(np.mean(r_test[mt] <= q_g))
        k = int(round(cov_g * n))
        lo, hi = clopper_pearson(k, n)
        cov_pz = float(np.mean(r_test[mt] <= q_pooled_z))
        cov_pr = float(np.mean(r_test[mt] * std[d] <= q_pooled_real))
        out["per_dataset"][d] = {"n_test": n, "n_val": int(mv.sum()), "q_groupwise_z": q_g,
                                 "cov_groupwise": cov_g, "cp95": [lo, hi],
                                 "cov_pooled_z": cov_pz, "cov_pooled_real": cov_pr}
        print(f"{d:8s}{n:8d}{100*cov_g:10.1f}%{'[%5.1f, %5.1f]' % (100*lo, 100*hi):>20s}"
              f"{100*cov_pz:10.1f}%{100*cov_pr:13.1f}%")

    path = os.path.join(config.OUTPUT_DIR, "predictor_consistent_recalibration.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
