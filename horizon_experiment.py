"""
Longer forecast horizons: does the calibration finding hold when the forecast is not trivial?

The main experiments predict the SOH of the next cycle. There, consecutive readings are so close
that a persistence forecast ("next SOH = last SOH") is hard to beat on densely sampled datasets.
This script repeats the core comparison at horizons of 10 and 50 cycles on the main split: the
same cells, the same 20-cycle window, the same model and training loop; only the target moves
h cycles ahead (preprocessing.run(horizon=h)).

For each horizon (and for h = 1 from the main run's saved predictions) it reports, per dataset:
  - the MAE in SOH units of the model (MC-Dropout mean), of persistence, and of a drift forecast
    that fits a straight line to the window and extends it h cycles;
  - pooled, groupwise and normalized-groupwise coverage of the model (deep_ensemble.calibrate_all);
  - pooled and groupwise coverage of the persistence and drift forecasts themselves, to show that
    the calibration finding does not depend on the forecaster beating the naive ones.

Requires: raw data (parsed once), outputs/conformal_groupwise_predictions{,_val}.npz (h = 1).
Output:   cache/processed_h{h}.npz, outputs/horizon{h}_groupwise_*, outputs/horizon_experiment.json
Run:      python horizon_experiment.py
"""
import json
import os

import numpy as np
import torch

import config
import preprocessing
from conformal import GroupwiseConformal, SplitConformal
from deep_ensemble import FIVE, calibrate_all
from model import build_model, mc_dropout_predict
from train import train_one

HORIZONS = [10, 50]
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
OUTPUT = os.path.join(config.OUTPUT_DIR, "horizon_experiment.json")


def naive_forecasts(X, h):
    """Persistence and straight-line drift over the input window, h cycles past its last point."""
    x = X[:, :, 0].astype(np.float64)
    w = x.shape[1]
    t = np.arange(w, dtype=np.float64)
    slope = ((t - t.mean()) * (x - x.mean(1, keepdims=True))).sum(1) / ((t - t.mean()) ** 2).sum()
    intercept = x.mean(1) - slope * t.mean()
    return {"persistence": x[:, -1], "drift": intercept + slope * (w - 1 + h)}


def coverage_pooled_groupwise(y_val, pv, lab_val, y_test, pt, lab_test):
    out = {}
    pooled = SplitConformal(alpha=config.CONFORMAL_ALPHA)
    pooled.calibrate(y_val, pv)
    gw = GroupwiseConformal(alpha=config.CONFORMAL_ALPHA)
    gw.calibrate(y_val, pv, lab_val)
    for name, (lo, hi) in (("pooled", pooled.predict_interval(pt)),
                           ("groupwise", gw.predict_interval(pt, lab_test))):
        cov = {d: float(np.mean((y_test[lab_test == d] >= lo[lab_test == d]) &
                                (y_test[lab_test == d] <= hi[lab_test == d]))) for d in np.unique(lab_test)}
        out[name] = {"per_dataset_coverage": cov,
                     "mean_abs_dev_five": float(np.mean([abs(cov[d] - 0.9) for d in FIVE if d in cov]))}
    return out


def evaluate(h, cache, mc_val, mc_val_std, mc_test, mc_test_std):
    std = cache["soh_std"].item()
    y_val, lab_val, X_val = cache["y_val"], cache["lab_val"], cache["X_val"]
    y_test, lab_test, X_test = cache["y_test"], cache["lab_test"], cache["X_test"]
    naive_val, naive_test = naive_forecasts(X_val, h), naive_forecasts(X_test, h)

    mae = {}
    for d in DATASETS:
        m = lab_test == d
        if not m.any():
            continue
        mae[d] = {"n_test_windows": int(m.sum()), "n_test_cells": int(len(np.unique(cache["cell_test"][m]))),
                  "model": float(np.mean(np.abs(y_test[m] - mc_test[m])) * std[d])}
        for k, p in naive_test.items():
            mae[d][k] = float(np.mean(np.abs(y_test[m] - p[m])) * std[d])
        mae[d]["model_beats_persistence"] = bool(mae[d]["model"] < mae[d]["persistence"])
        mae[d]["model_beats_drift"] = bool(mae[d]["model"] < mae[d]["drift"])

    scale = np.array([std[d] for d in lab_test])
    pooled_mae = {"model": float(np.mean(np.abs(y_test - mc_test) * scale))}
    for k, p in naive_test.items():
        pooled_mae[k] = float(np.mean(np.abs(y_test - p) * scale))

    return {
        "horizon": h,
        "mae_real": mae,
        "pooled_mae_real": pooled_mae,
        "model_calibration": calibrate_all(y_val, mc_val, mc_val_std, lab_val, y_test, mc_test, mc_test_std, lab_test),
        "naive_calibration": {k: coverage_pooled_groupwise(y_val, naive_val[k], lab_val, y_test, naive_test[k], lab_test)
                              for k in naive_test},
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    v = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
    t = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    results["1"] = evaluate(1, cache, v["mean_pred"], v["mc_std"], t["mean_pred"], t["mc_std"])

    raw_df = None
    for h in HORIZONS:
        cache_path = os.path.join(config.CACHE_DIR, f"processed_h{h}.npz")
        if not os.path.exists(cache_path):
            if raw_df is None:
                raw_df = preprocessing.add_soh(preprocessing.load_all_raw())
            preprocessing.run(raw_df=raw_df, out_path=cache_path, horizon=h)
        run_name = f"horizon{h}_groupwise"
        ckpt = os.path.join(config.OUTPUT_DIR, f"{run_name}_model.pt")
        if not os.path.exists(ckpt):
            r = train_one(cache_path=cache_path, model_variant="full", seed=config.RANDOM_SEED,
                          conformal_mode="groupwise", run_name=run_name, verbose=False)
            print(f"[h={h}] trained: test MAE {r['test_mae']:.5f}, best epoch {r['best_epoch']}")
        cache = np.load(cache_path, allow_pickle=True)
        model = build_model("full").to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        val_mean, val_std = mc_dropout_predict(model, torch.tensor(cache["X_val"]).to(device))
        test = np.load(os.path.join(config.OUTPUT_DIR, f"{run_name}_predictions.npz"), allow_pickle=True)
        assert np.array_equal(test["y_test"], cache["y_test"])
        results[str(h)] = evaluate(h, cache, val_mean.cpu().numpy(), val_std.cpu().numpy(),
                                   test["mean_pred"], test["mc_std"])

    with open(OUTPUT, "w") as fh:
        json.dump(results, fh, indent=2)

    for h, r in results.items():
        print(f"\n=== horizon {h} ===")
        print(f"{'dataset':8s}{'cells':>6s}{'model':>9s}{'persist':>9s}{'drift':>9s}")
        for d, m in r["mae_real"].items():
            print(f"{d:8s}{m['n_test_cells']:6d}{m['model']:9.5f}{m['persistence']:9.5f}{m['drift']:9.5f}")
        p = r["pooled_mae_real"]
        print(f"{'pooled':8s}{'':6s}{p['model']:9.5f}{p['persistence']:9.5f}{p['drift']:9.5f}")
        for mode in ("pooled", "groupwise", "normalized_groupwise"):
            c = r["model_calibration"][mode]
            print(f"  model {mode:21s} dev5 {c['mean_abs_dev_five'] * 100:5.2f}  BIT "
                  f"{c['per_dataset_coverage'].get('BIT', float('nan')) * 100:5.1f}")
        for k, c in r["naive_calibration"].items():
            print(f"  {k:11s} pooled dev5 {c['pooled']['mean_abs_dev_five'] * 100:5.2f}  "
                  f"groupwise dev5 {c['groupwise']['mean_abs_dev_five'] * 100:5.2f}")
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
