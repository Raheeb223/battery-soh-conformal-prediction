"""
K7 follow-up to E9/xjtu_bridging_isolation.py.

xjtu_bridging_isolation.py established that 376 of XJTU's 2,605 test
windows (all from the 8 Sim_satellite_* cells) bridge a removed
sentinel-value block on every single window, while the other 2,229 XJTU
test windows show zero bridging. This script quantifies the practical
effect: does excluding those 376 windows change XJTU's reported MAE/RMSE/
coverage materially?

No retrain needed -- this re-slices the ALREADY-COMPUTED per-window test
predictions (outputs/conformal_groupwise_predictions.npz) using the
bridging mask, which is just a filter over existing rows.

Run:  python xjtu_bridging_sensitivity.py
Out:  outputs/xjtu_bridging_sensitivity.json
"""
import json
import os

import numpy as np

import config


def main():
    cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    pred = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)

    lab_test = cache["lab_test"]
    cell_test = cache["cell_test"]
    window_cycles = cache["window_cycle_indices_test"]
    soh_mean = cache["soh_mean"].item()["XJTU"]
    soh_std = cache["soh_std"].item()["XJTU"]

    assert np.array_equal(lab_test, pred["lab_test"]), "cache/predictions row order mismatch"

    xjtu_mask = lab_test == "XJTU"
    is_satellite = np.array(["Sim_satellite" in str(c) for c in cell_test])
    diffs = np.diff(window_cycles, axis=1)
    bridging = np.any(diffs != 1, axis=1)

    bridging_mask = xjtu_mask & bridging
    clean_mask = xjtu_mask & ~bridging

    y = pred["y_test"]
    yhat = pred["mean_pred"]
    lower, upper = pred["lower"], pred["upper"]

    def stats(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        err_z = y[mask] - yhat[mask]
        err_real = err_z * soh_std
        covered = (y[mask] >= lower[mask]) & (y[mask] <= upper[mask])
        return {
            "n": n,
            "mae_z": float(np.mean(np.abs(err_z))),
            "rmse_z": float(np.sqrt(np.mean(err_z ** 2))),
            "mae_real": float(np.mean(np.abs(err_real))),
            "rmse_real": float(np.sqrt(np.mean(err_real ** 2))),
            "coverage": float(np.mean(covered)),
        }

    result = {
        "all_xjtu": stats(xjtu_mask),
        "bridging_windows_only": stats(bridging_mask),
        "clean_windows_only": stats(clean_mask),
        "bridging_fraction": float(bridging_mask.sum()) / float(xjtu_mask.sum()),
    }

    print(f"{'subset':24s}{'n':>7s}{'MAE z':>9s}{'RMSE z':>9s}{'MAE real':>10s}{'coverage':>10s}")
    for key, label in [("all_xjtu", "all XJTU"), ("clean_windows_only", "clean (no bridge)"),
                        ("bridging_windows_only", "bridging only")]:
        r = result[key]
        print(f"{label:24s}{r['n']:7d}{r['mae_z']:9.4f}{r['rmse_z']:9.4f}{r['mae_real']:10.5f}{r['coverage']:10.3f}")

    delta_mae_z = result["all_xjtu"]["mae_z"] - result["clean_windows_only"]["mae_z"]
    result["headline_mae_z_shift_from_excluding_bridging_windows"] = float(delta_mae_z)
    print(f"\nExcluding the 376 bridging windows shifts pooled-XJTU MAE(z) by {delta_mae_z:+.5f} "
          f"({result['all_xjtu']['mae_z']:.5f} -> {result['clean_windows_only']['mae_z']:.5f}).")

    out_path = os.path.join(config.OUTPUT_DIR, "xjtu_bridging_sensitivity.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
