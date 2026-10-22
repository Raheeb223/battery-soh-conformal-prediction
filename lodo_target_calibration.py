"""
Leave-one-dataset-out with calibration cells from the left-out dataset.

leave_one_out.py shows that a model never trained on a dataset is badly calibrated on it when its
interval comes from the other datasets (coverage as low as 21.2%). That test mixes two failures: the
model has not learned the new dataset, and the interval width was learned elsewhere. Split
conformal needs only calibration data exchangeable with the test data, not a model trained on
them, so this script separates the two. For each dataset D it retrains the LODO model (the same
training loop as leave_one_out.py, now saving the model) and then calibrates it in three ways:

  - others:   one quantile from the other datasets' validation cells (what leave_one_out.py does);
  - own_val:  D's own validation cells of the main split (the cells groupwise calibration uses);
  - own_all:  every non-test cell of D in the main split (training and validation cells; the LODO
              model has seen none of them);

and evaluates all three on D's test cells of the main split, the same test cells as every
calibration-level result in the paper. A budget sweep then draws b cells at random from D's
non-test cells (500 draws per budget) to find how many target cells restore coverage. Widths and
errors are compared, in SOH units, with the main model (trained on every dataset) on the same test
windows.

Requires: the raw datasets (as leave_one_out.py), cache/processed.npz,
          outputs/conformal_groupwise_predictions.npz.
Output:   outputs/lodo_target_calibration.json, outputs/lodo_target_<D>_model.pt

Run:  python lodo_target_calibration.py [--only MIT BIT ...]
"""
import argparse
import json
import os
import tempfile

import numpy as np
import torch

import config
import preprocessing
from calibration_size_sensitivity import budget_sweep, fsq
from leave_one_out import build_lodo_cache, train_lodo_model
from model import build_model, mc_dropout_predict

DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]
OUTPUT = os.path.join(config.OUTPUT_DIR, "lodo_target_calibration.json")
SUFFICIENT_P05 = 0.85


def main_split_cells(main, d):
    cells = lambda part: set(np.unique(main[f"cell_{part}"][main[f"lab_{part}"] == d]).tolist())  # noqa: E731
    return cells("train"), cells("val"), cells("test")


def run_one(d, raw_df, main, ref, device):
    data = build_lodo_cache(d, raw_df=raw_df)
    path = os.path.join(config.OUTPUT_DIR, f"lodo_target_{d}_model.pt")
    model = build_model("full").to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    else:
        model = train_lodo_model(data, f"lodo_target_{d}", "full", device)
        torch.save(model.state_dict(), path)
    model.eval()

    pv, _ = mc_dropout_predict(model, torch.tensor(data["X_val"]).to(device))
    pt, _ = mc_dropout_predict(model, torch.tensor(data["X_test"]).to(device))
    r_others = np.abs(data["y_val"] - pv.cpu().numpy())
    pred_d = pt.cpu().numpy()
    y_d, cell_d = data["y_test"], data["cell_test"]
    res_d = np.abs(y_d - pred_d)

    train_c, val_c, test_c = main_split_cells(main, d)
    in_test = np.isin(cell_d, list(test_c))
    in_val = np.isin(cell_d, list(val_c))
    in_pool = np.isin(cell_d, list(train_c | val_c))
    std_lodo = data["held_out_soh_std"]
    std_main = float(main["soh_std"].item()[d])

    def evaluate(q):
        return {"coverage": float(np.mean(res_d[in_test] <= q)), "half_width_soh": float(q * std_lodo)}

    out = {
        "n_test_cells": len(test_c), "n_test_windows": int(in_test.sum()),
        "n_val_cells": len(val_c), "n_pool_cells": int(len(np.unique(cell_d[in_pool]))),
        "others": evaluate(fsq(r_others)),
        "own_val": evaluate(fsq(res_d[in_val])) if in_val.any() else None,
        "own_all": evaluate(fsq(res_d[in_pool])),
        "all_windows_others_coverage": float(np.mean(res_d <= fsq(r_others))),
        "lodo_test_mae_soh": float(np.mean(res_d[in_test]) * std_lodo),
    }

    # The main model (trained on every dataset) on the same test windows, groupwise calibrated.
    m = ref["lab_test"] == d
    out["in_training"] = {
        "coverage": float(np.mean((ref["y_test"][m] >= ref["lower"][m]) & (ref["y_test"][m] <= ref["upper"][m]))),
        "half_width_soh": float(np.mean(ref["upper"][m] - ref["lower"][m]) / 2 * std_main),
        "test_mae_soh": float(np.mean(np.abs(ref["y_test"][m] - ref["mean_pred"][m])) * std_main),
    }
    assert out["n_test_windows"] == int(m.sum()), f"{d}: test windows differ from the main split"

    # Budget sweep over D's non-test cells, with the same code as the laboratory sweep.
    if len(np.unique(cell_d[in_pool])) > 1:
        with tempfile.TemporaryDirectory() as tmp:
            vp, tp, cp = (os.path.join(tmp, f) for f in ("val.npz", "test.npz", "cache.npz"))
            lab = np.array([d] * len(y_d), dtype=object)
            np.savez(vp, y_val=y_d[in_pool], mean_pred=pred_d[in_pool], lab_val=lab[in_pool], cell_val=cell_d[in_pool])
            np.savez(tp, y_test=y_d[in_test], mean_pred=pred_d[in_test], lab_test=lab[in_test], cell_test=cell_d[in_test])
            np.savez(cp, soh_std=np.array({d: std_lodo}, dtype=object))
            sweep = budget_sweep(cp, vp, tp, datasets=[d])["per_dataset"][d]
        n_full = sweep["n_cells_available"]
        ok = sorted(int(b) for b, r in sweep["per_budget"].items()
                    if int(b) < n_full and r["coverage_p05"] >= SUFFICIENT_P05)
        out["budget"] = sweep
        out["cells_needed"] = ok[0] if ok else None
    print(f"[lodo_target] {d}: others {out['others']['coverage']:.1%}  own_val "
          f"{(out['own_val'] or {}).get('coverage', float('nan')):.1%}  own_all {out['own_all']['coverage']:.1%}  "
          f"in-training {out['in_training']['coverage']:.1%}  cells needed {out.get('cells_needed')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=DATASETS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_cache = np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)
    ref = np.load(os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions.npz"), allow_pickle=True)
    raw_df = preprocessing.add_soh(preprocessing.load_all_raw())

    results = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as fh:
            results = json.load(fh).get("per_dataset", {})
    for d in args.only:
        results[d] = run_one(d, raw_df, main_cache, ref, device)
        with open(OUTPUT, "w") as fh:
            json.dump({"alpha": config.CONFORMAL_ALPHA, "sufficient_p05": SUFFICIENT_P05,
                       "per_dataset": results}, fh, indent=2)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
