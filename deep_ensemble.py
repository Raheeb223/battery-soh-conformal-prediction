"""
A deep ensemble as a second source of uncertainty, calibrated with the same conformal code.

Every other interval in the paper is centred on the MC-Dropout mean of one network, or on its
single deterministic pass. This script replaces that network with a deep ensemble
(Lakshminarayanan et al., 2017): five copies of the full model trained from different seeds on the
same cells, each used with one deterministic pass. The ensemble mean is the point prediction and
the spread between members is the per-sample uncertainty.

The seed-42, 123 and 2024 members are the multi-seed runs of experiments.py; the two further
members (seeds 7 and 99) are trained here with the same settings if their checkpoints are missing.

For both the ensemble and the MC-Dropout model, it reports
  - the raw Gaussian interval mean +/- z_0.95 * spread, with no conformal step, to show how well
    each source of uncertainty is calibrated on its own;
  - pooled, groupwise, and normalized-groupwise conformal intervals (the last scaled by the
    ensemble spread or the MC-Dropout standard deviation).

Requires: cache/processed.npz, outputs/multiseed_{42,123,2024}_model.pt,
          outputs/conformal_groupwise_predictions{,_val}.npz (MC-Dropout reference).
Output:   outputs/deep_ensemble.json, outputs/deep_ensemble_predictions.npz

Run:  python deep_ensemble.py
"""
import json
import os

import numpy as np
import torch
from scipy.stats import norm

import config
from conformal import GroupwiseConformal, NormalizedGroupwiseConformal, SplitConformal
from model import build_model, predict_in_batches
from train import train_one

SEEDS = [42, 123, 2024, 7, 99]
EXISTING = {42, 123, 2024}
FIVE = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]
SIX = FIVE + ["Oxford"]
OUTPUT = os.path.join(config.OUTPUT_DIR, "deep_ensemble.json")


def member_path(seed):
    name = f"multiseed_{seed}" if seed in EXISTING else f"ensemble_seed{seed}"
    return name, os.path.join(config.OUTPUT_DIR, f"{name}_model.pt")


def member_predictions(seed, X_val, X_test, device):
    name, path = member_path(seed)
    if not os.path.exists(path):
        print(f"[ensemble] training member {name}")
        train_one(model_variant="full", seed=seed, run_name=name)
    model = build_model("full").to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    pv = predict_in_batches(model, torch.tensor(X_val).to(device)).cpu().numpy()
    pt = predict_in_batches(model, torch.tensor(X_test).to(device)).cpu().numpy()
    return pv, pt


def summarize(y, lower, upper, labels):
    per = {d: float(np.mean((y[labels == d] >= lower[labels == d]) & (y[labels == d] <= upper[labels == d])))
           for d in np.unique(labels)}
    return {
        "overall_coverage": float(np.mean((y >= lower) & (y <= upper))),
        "per_dataset_coverage": per,
        "mean_width": float(np.mean(upper - lower)),
        "mean_abs_dev_five": float(np.mean([abs(per[d] - 0.9) for d in FIVE])),
        "mean_abs_dev_six": float(np.mean([abs(per[d] - 0.9) for d in SIX])),
    }


def calibrate_all(y_val, pv, sv, lab_val, y_test, pt, st, lab_test):
    alpha = config.CONFORMAL_ALPHA
    z = norm.ppf(1 - alpha / 2)
    out = {"raw_gaussian": summarize(y_test, pt - z * st, pt + z * st, lab_test)}
    pooled = SplitConformal(alpha=alpha)
    pooled.calibrate(y_val, pv)
    lo, hi = pooled.predict_interval(pt)
    out["pooled"] = summarize(y_test, lo, hi, lab_test)
    gw = GroupwiseConformal(alpha=alpha)
    gw.calibrate(y_val, pv, lab_val)
    lo, hi = gw.predict_interval(pt, lab_test)
    out["groupwise"] = summarize(y_test, lo, hi, lab_test)
    ngw = NormalizedGroupwiseConformal(alpha=alpha)
    ngw.calibrate(y_val, pv, sv, lab_val)
    lo, hi = ngw.predict_interval(pt, st, lab_test)
    out["normalized_groupwise"] = summarize(y_test, lo, hi, lab_test)
    out["test_mae"] = float(np.mean(np.abs(y_test - pt)))
    out["test_rmse"] = float(np.sqrt(np.mean((y_test - pt) ** 2)))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    X_val, y_val, lab_val = data["X_val"], data["y_val"], data["lab_val"]
    X_test, y_test, lab_test = data["X_test"], data["y_test"], data["lab_test"]

    members_val, members_test = [], []
    for seed in SEEDS:
        pv, pt = member_predictions(seed, X_val, X_test, device)
        members_val.append(pv)
        members_test.append(pt)
        print(f"[ensemble] member seed {seed}: test MAE {np.mean(np.abs(y_test - pt)):.5f}")
    members_val, members_test = np.stack(members_val), np.stack(members_test)
    ens_pv, ens_sv = members_val.mean(0), members_val.std(0)
    ens_pt, ens_st = members_test.mean(0), members_test.std(0)

    mc_val = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz"), allow_pickle=True)
    mc_test = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    assert np.array_equal(mc_test["y_test"], y_test) and np.array_equal(mc_val["y_val"], y_val)

    results = {
        "seeds": SEEDS,
        "n_members": len(SEEDS),
        "alpha": config.CONFORMAL_ALPHA,
        "deep_ensemble": calibrate_all(y_val, ens_pv, ens_sv, lab_val, y_test, ens_pt, ens_st, lab_test),
        "mc_dropout": calibrate_all(y_val, mc_val["mean_pred"], mc_val["mc_std"], lab_val,
                                    y_test, mc_test["mean_pred"], mc_test["mc_std"], lab_test),
        "member_test_mae": [float(np.mean(np.abs(y_test - p))) for p in members_test],
        "spread_test_median": {"deep_ensemble": float(np.median(ens_st)),
                               "mc_dropout": float(np.median(mc_test["mc_std"]))},
    }
    with open(OUTPUT, "w") as fh:
        json.dump(results, fh, indent=2)
    np.savez(os.path.join(config.OUTPUT_DIR, "deep_ensemble_predictions.npz"),
             y_val=y_val, mean_val=ens_pv, std_val=ens_sv, lab_val=lab_val,
             y_test=y_test, mean_test=ens_pt, std_test=ens_st, lab_test=lab_test,
             cell_test=data["cell_test"])

    for src in ("deep_ensemble", "mc_dropout"):
        r = results[src]
        print(f"\n{src}: test MAE {r['test_mae']:.5f}")
        for mode in ("raw_gaussian", "pooled", "groupwise", "normalized_groupwise"):
            m = r[mode]
            per = "  ".join(f"{d} {m['per_dataset_coverage'][d] * 100:5.1f}" for d in SIX)
            print(f"  {mode:21s} overall {m['overall_coverage'] * 100:5.1f}  width {m['mean_width']:.4f}  "
                  f"dev5 {m['mean_abs_dev_five'] * 100:5.2f}  | {per}")
    print(f"\nsaved {OUTPUT}")


if __name__ == "__main__":
    main()
