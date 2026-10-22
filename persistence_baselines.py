"""
Naive one-step-ahead baselines versus the trained model.

The model input is the previous `INPUT_WINDOW` (=20) SOH values of one cell
and the target is the next recorded SOH value. For that task the natural
zero-training baselines are:
  persistence        : yhat = x[-1]
  lag-2              : yhat = x[-2]
  linear extrapolation: yhat = 2*x[-1] - x[-2]
None of these was included in the baseline comparison of the manuscript.

UNITS. The pipeline z-scores SOH per dataset, and the pooled MAE
and RMSE are in those z-units, not in physical SOH units. This script reports
both: `z` (per-dataset z-units, pooled over windows) and `real` (physical SOH
units, used for the per-dataset persistence comparison).

Model predictions are the MC-Dropout means stored in
outputs/conformal_groupwise_predictions.npz (the same predictions used for
every interval in the paper).

Requires: cache/processed.npz, outputs/conformal_groupwise_predictions.npz,
config.py.
Run:  python persistence_baselines.py
Out:  outputs/persistence_baselines.json
"""

import json
import os

import numpy as np

import config

DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]


def metrics(err):
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def main():
    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    test = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    std = cache["soh_std"].item()

    X, y, lab = cache["X_test"][:, :, 0], cache["y_test"], cache["lab_test"]
    assert np.allclose(test["y_test"], y, atol=1e-6), "prediction file rows do not match cache test rows"
    preds = {
        "model": test["mean_pred"],
        "persistence": X[:, -1],
        "lag2": X[:, -2],
        "linear_extrapolation": 2 * X[:, -1] - X[:, -2],
    }
    scale = np.array([std[d] for d in lab])

    out = {"per_dataset": {}, "pooled": {}, "macro_over_datasets": {}}
    print(f"{'Dataset':8s}{'n':>7s} | real-unit MAE:{'model':>9s}{'persist':>9s}{'lag2':>9s}{'linext':>9s} | real-unit RMSE:{'model':>9s}{'persist':>9s}")
    per_real = {k: [] for k in preds}
    for d in DATASETS:
        m = lab == d
        row = {"n": int(m.sum())}
        for k, p in preds.items():
            mae, rmse = metrics((y[m] - p[m]) * std[d])
            row[k] = {"mae_real": mae, "rmse_real": rmse}
            per_real[k].append(mae)
        out["per_dataset"][d] = row
        print(f"{d:8s}{m.sum():7d} |{'':15s}{row['model']['mae_real']:9.5f}{row['persistence']['mae_real']:9.5f}"
              f"{row['lag2']['mae_real']:9.5f}{row['linear_extrapolation']['mae_real']:9.5f} |{'':16s}"
              f"{row['model']['rmse_real']:9.5f}{row['persistence']['rmse_real']:9.5f}")

    print("\nPooled over all test windows")
    print(f"{'predictor':22s}{'MAE z':>9s}{'RMSE z':>9s}{'MAE real':>10s}{'RMSE real':>11s}")
    for k, p in preds.items():
        mz, rz = metrics(y - p)
        mr, rr = metrics((y - p) * scale)
        out["pooled"][k] = {"mae_z": mz, "rmse_z": rz, "mae_real": mr, "rmse_real": rr}
        out["macro_over_datasets"][k] = {"mae_real_7": float(np.mean(per_real[k])),
                                         "mae_real_6_excl_nasa": float(np.mean(per_real[k][:6]))}
        print(f"{k:22s}{mz:9.4f}{rz:9.4f}{mr:10.5f}{rr:11.5f}")

    path = os.path.join(config.OUTPUT_DIR, "persistence_baselines.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
