"""
Deterministic forward pass vs the MC-Dropout mean, as the centre of groupwise conformal intervals.

Every interval in the paper is centred on the mean of MC_DROPOUT_SAMPLES stochastic passes. The
same checkpoint's deterministic pass (dropout off) has a different point accuracy. This script
calibrates the deterministic predictor with groupwise conformal on the validation set, exactly as
train.py does for the MC mean (calibration and test predictions from the SAME predictor), and
compares accuracy, coverage, width, and the persistence baseline per dataset.

Requires: cache/processed.npz, outputs/conformal_groupwise_model.pt,
          outputs/conformal_groupwise_predictions.npz
Run:  python deterministic_predictor_comparison.py
Out:  outputs/deterministic_predictor_results.json
"""

import json
import os

import numpy as np
import torch

import config
from conformal import GroupwiseConformal
from model import build_model, predict_in_batches

DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
FIVE = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]


def summarize(name, y, pred, lower, upper, lab, std):
    out = {
        "mae_z": float(np.mean(np.abs(y - pred))),
        "rmse_z": float(np.sqrt(np.mean((y - pred) ** 2))),
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mpiw": float(np.mean(upper - lower)),
        "per_dataset": {},
    }
    for d in DATASETS:
        m = lab == d
        out["per_dataset"][d] = {
            "coverage": float(np.mean((y[m] >= lower[m]) & (y[m] <= upper[m]))),
            "mpiw": float(np.mean(upper[m] - lower[m])),
            "mae_real": float(np.mean(np.abs(y[m] - pred[m])) * std[d]),
        }
    out["mean_abs_dev_five"] = float(np.mean([abs(out["per_dataset"][d]["coverage"] - (1 - config.CONFORMAL_ALPHA))
                                              for d in FIVE]))
    print(f"{name:14s} MAE_z={out['mae_z']:.5f} RMSE_z={out['rmse_z']:.5f} "
          f"cov={100 * out['coverage']:.1f}% MPIW={out['mpiw']:.5f} meanDev5={100 * out['mean_abs_dev_five']:.2f}")
    return out


def run_split(seed, cache_name, run_name, device):
    print(f"\n===== split seed {seed} =====")
    cache = np.load(os.path.join(config.CACHE_DIR, cache_name), allow_pickle=True)
    std = cache["soh_std"].item()

    model = build_model("full")
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, f"{run_name}_model.pt"),
                                     map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    y_val, lab_val = cache["y_val"], cache["lab_val"]
    y_test, lab_test = cache["y_test"], cache["lab_test"]
    det_val = predict_in_batches(model, torch.from_numpy(cache["X_val"]).to(device)).cpu().numpy()
    det_test = predict_in_batches(model, torch.from_numpy(cache["X_test"]).to(device)).cpu().numpy()

    gc = GroupwiseConformal(alpha=config.CONFORMAL_ALPHA)
    gc.calibrate(y_val, det_val, lab_val)
    det_lo, det_hi = gc.predict_interval(det_test, lab_test)

    mc = np.load(os.path.join(config.OUTPUT_DIR, f"{run_name}_predictions.npz"), allow_pickle=True)
    assert np.allclose(mc["y_test"], y_test, atol=1e-6), "prediction file rows do not match cache test rows"

    results = {
        "deterministic": summarize("deterministic", y_test, det_test, det_lo, det_hi, lab_test, std),
        "mc_mean": summarize("mc_mean", y_test, mc["mean_pred"], mc["lower"], mc["upper"], lab_test, std),
        "persistence_mae_real": {},
    }
    persist = cache["X_test"][:, -1, 0]
    print(f"\n{'dataset':8s}{'det':>10s}{'mc_mean':>10s}{'persist':>10s}  det beats persistence?")
    for d in DATASETS:
        m = lab_test == d
        p = float(np.mean(np.abs(y_test[m] - persist[m])) * std[d])
        results["persistence_mae_real"][d] = p
        det_mae = results["deterministic"]["per_dataset"][d]["mae_real"]
        print(f"{d:8s}{det_mae:10.5f}{results['mc_mean']['per_dataset'][d]['mae_real']:10.5f}{p:10.5f}  "
              f"{'yes' if det_mae < p else 'no'}")
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {config.RANDOM_SEED: ("processed.npz", "conformal_groupwise")}
    for seed in (43, 44, 45, 46):
        splits[seed] = (f"processed_split{seed}.npz", f"split{seed}_groupwise")

    out = {"per_split": {str(s): run_split(s, c, r, device) for s, (c, r) in splits.items()}}
    per = out["per_split"].values()
    det = [p["deterministic"] for p in per]
    mc = [p["mc_mean"] for p in per]
    out["summary"] = {
        "n_splits": len(det),
        "det_lower_mae_count": sum(d["mae_z"] < m["mae_z"] for d, m in zip(det, mc)),
        "det_narrower_count": sum(d["mpiw"] < m["mpiw"] for d, m in zip(det, mc)),
        "mae_reduction_pct_mean": float(np.mean([100 * (1 - d["mae_z"] / m["mae_z"]) for d, m in zip(det, mc)])),
        "mae_reduction_pct_min": float(np.min([100 * (1 - d["mae_z"] / m["mae_z"]) for d, m in zip(det, mc)])),
        "width_reduction_pct_mean": float(np.mean([100 * (1 - d["mpiw"] / m["mpiw"]) for d, m in zip(det, mc)])),
        "width_reduction_pct_min": float(np.min([100 * (1 - d["mpiw"] / m["mpiw"]) for d, m in zip(det, mc)])),
        "det_coverage_min": float(min(d["coverage"] for d in det)),
        "det_coverage_max": float(max(d["coverage"] for d in det)),
        "mc_coverage_min": float(min(m["coverage"] for m in mc)),
        "mc_coverage_max": float(max(m["coverage"] for m in mc)),
        "det_mean_dev_five_mean": float(np.mean([d["mean_abs_dev_five"] for d in det])),
        "mc_mean_dev_five_mean": float(np.mean([m["mean_abs_dev_five"] for m in mc])),
        "det_beats_persistence_count": {
            ds: sum(p["deterministic"]["per_dataset"][ds]["mae_real"] < p["persistence_mae_real"][ds] for p in per)
            for ds in DATASETS},
    }
    print("\n" + json.dumps(out["summary"], indent=2))

    path = os.path.join(config.OUTPUT_DIR, "deterministic_predictor_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
