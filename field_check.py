"""
Field check: do the laboratory findings hold for field data from electric vehicles?

Data: the EVBattery charging records (field_preprocessing.py), three manufacturers (EV1, EV2, EV3),
each vehicle a "cell" and each labelled charging event a "cycle". Every step reuses the laboratory
code unchanged; nothing here is tuned on field test data.

For each of five vehicle-level splits (split seeds 42-46, as for the laboratory data):
  1. Builds the field cache (field_preprocessing.run).
  2. Trains the same model with the same settings and seed as the main laboratory model
     (train.train_one, model 'full'), on field training vehicles only.
  3. One MC-Dropout run gives validation and test predictions (saved in the same format as the
     laboratory prediction files).
  4. Pooled and groupwise calibration in z-units (as for the laboratory data), the hierarchical
     analysis (hierarchical_calibration_v2.analyse: own data, empirical-Bayes shrinkage, prior only,
     other datasets pooled), and HGC-CP with its design chosen on validation data only
     (hgc_cp_validation_only.run_split).
On the main split it also runs:
  5. The calibration budget sweep (calibration_size_sensitivity.budget_sweep), in vehicles.
  6. Laboratory model on field data: the main laboratory model (outputs/conformal_groupwise_model.pt),
     never trained on field data, predicts the field test vehicles. Its intervals are calibrated
     either with the laboratory pooled quantile or on the field validation vehicles (own data).

Requires: EVBattery archives; outputs/conformal_groupwise_model.pt and
          outputs/conformal_groupwise_predictions_val.npz (laboratory main model).
Output:   outputs/field_check.json (and outputs/field*_model.pt, field*_{val,test}.npz)
Run:      python field_check.py
"""
import json
import os

import numpy as np
import torch

import calibration_size_sensitivity as css
import config
import field_preprocessing
import hgc_cp_validation_only as hv
import hierarchical_calibration_v2 as hier
import preprocessing
from model import build_model, mc_dropout_predict
from train import train_one

SPLITS = [42, 43, 44, 45, 46]
FIELD = ["EV1", "EV2", "EV3"]
ALPHA = config.CONFORMAL_ALPHA
LAB_MODEL = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_model.pt")
LAB_VAL = os.path.join(config.OUTPUT_DIR, "conformal_groupwise_predictions_val.npz")


def fsq(r):
    n = len(r)
    return float(np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))


def tag(s):
    return "field" if s == 42 else f"field_split{s}"


def files(s):
    t = tag(s)
    return (field_preprocessing.cache_path(None if s == 42 else s),
            os.path.join(config.OUTPUT_DIR, f"{t}_model.pt"),
            os.path.join(config.OUTPUT_DIR, f"{t}_val.npz"),
            os.path.join(config.OUTPUT_DIR, f"{t}_test.npz"))


def mc_predictions(model_path, cache_path, val_path, test_path, seed):
    """One MC-Dropout run for validation and test, saved like the laboratory prediction files."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("full").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    cache = np.load(cache_path, allow_pickle=True)
    torch.manual_seed(seed)
    for split, path in (("val", val_path), ("test", test_path)):
        mean, std = mc_dropout_predict(model, torch.tensor(cache[f"X_{split}"]).to(device))
        np.savez(path, **{f"y_{split}": cache[f"y_{split}"], "mean_pred": mean.cpu().numpy(),
                          "mc_std": std.cpu().numpy(), f"lab_{split}": cache[f"lab_{split}"],
                          f"cell_{split}": cache[f"cell_{split}"]})


def prepare(s, df):
    cache_path, model_path, val_path, test_path = files(s)
    if not os.path.exists(cache_path):
        field_preprocessing.run(split_seed=None if s == 42 else s, df=df)
    if not os.path.exists(model_path):
        train_one(cache_path=cache_path, model_variant="full", run_name=tag(s))
    if not (os.path.exists(val_path) and os.path.exists(test_path)):
        mc_predictions(model_path, cache_path, val_path, test_path, seed=s)
    return cache_path, val_path, test_path


def pooled_vs_groupwise(val_path, test_path, datasets):
    """Coverage and mean width in z-units, as in the laboratory main results."""
    val, test = np.load(val_path, allow_pickle=True), np.load(test_path, allow_pickle=True)
    rv = np.abs(val["y_val"] - val["mean_pred"])
    rt = np.abs(test["y_test"] - test["mean_pred"])
    q_pool = fsq(rv)
    out = {"pooled": {}, "groupwise": {}, "n_test_windows": {}, "n_test_vehicles": {}, "n_val_vehicles": {}}
    half_g = np.empty_like(rt)
    for d in datasets:
        mv, mt = val["lab_val"] == d, test["lab_test"] == d
        if not (mv.any() and mt.any()):
            continue
        q_g = fsq(rv[mv])
        half_g[mt] = q_g
        out["pooled"][d] = float(np.mean(rt[mt] <= q_pool))
        out["groupwise"][d] = float(np.mean(rt[mt] <= q_g))
        out["n_test_windows"][d] = int(mt.sum())
        out["n_test_vehicles"][d] = int(len(np.unique(test["cell_test"][mt])))
        out["n_val_vehicles"][d] = int(len(np.unique(val["cell_val"][mv])))
    out["pooled_overall"] = float(np.mean(rt <= q_pool))
    out["groupwise_overall"] = float(np.mean(rt <= half_g))
    out["pooled_mean_width"] = 2 * q_pool
    out["groupwise_mean_width"] = float(np.mean(2 * half_g))
    out["test_mae"] = float(np.mean(rt))
    return out


def borrowing(cache_path, val_path, test_path):
    hier.ALL, hier.TARGETS = FIELD, FIELD
    h = hier.analyse(cache_path, val_path, test_path)
    # HGC-CP-val's design search rotates stand-in targets among the other datasets, so it needs a
    # target and at least two others with validation and test windows. A split in which EV3's only
    # validation vehicle is too short for a window leaves two, and HGC-CP-val is not run there.
    val, test = np.load(val_path, allow_pickle=True), np.load(test_path, allow_pickle=True)
    usable = [d for d in FIELD if (val["lab_val"] == d).any() and (test["lab_test"] == d).any()]
    if len(usable) >= 3:
        hv.split_paths = lambda _: (cache_path, val_path, test_path)
        hv.DATASETS, hv.FIVE = FIELD, FIELD
        g = hv.run_split("field")
    else:
        g = {"per_target": {}}
    per = {}
    for d in FIELD:
        t = h["per_target"].get(d)
        if t is None:
            continue
        per[d] = {"own": t["cov_own_quantile"], "hierarchical": t["cov_eb_shrunk"],
                  "prior_only": t["cov_prior_only"], "others_pooled": t["cov_cross_source_pooled"],
                  "w_eb": t["w_eb"], "hgc_cp_val": g["per_target"].get(d, {}).get("coverage"),
                  "hgc_cp_val_design": {k: g["per_target"][d][k] for k in ("representation", "lam", "beta")}
                  if d in g["per_target"] else None}
    return {"fit_groups": h["fit_groups"], "usable_datasets": usable,
            "hgc_cp_val_run": len(usable) >= 3, "per_dataset": per}


def persistence_comparison(cache_path, test_path):
    """Model against one-step persistence (next SOH = last SOH), MAE in SOH units, main split."""
    cache, test = np.load(cache_path, allow_pickle=True), np.load(test_path, allow_pickle=True)
    std = cache["soh_std"].item()
    out = {}
    for d in FIELD:
        m = cache["lab_test"] == d
        if not m.any():
            continue
        out[d] = {"model_mae_soh": float(np.mean(np.abs(test["y_test"][m] - test["mean_pred"][m])) * std[d]),
                  "persistence_mae_soh": float(np.mean(np.abs(cache["y_test"][m] - cache["X_test"][m][:, -1, 0])) * std[d])}
    return out


def lab_model_on_field(cache_path):
    """The laboratory main model, never trained on field data, on the field vehicles."""
    lab_cache_val = np.load(LAB_VAL, allow_pickle=True)
    q_lab_pooled = fsq(np.abs(lab_cache_val["y_val"] - lab_cache_val["mean_pred"]))
    tmp_val = os.path.join(config.OUTPUT_DIR, "field_labmodel_val.npz")
    tmp_test = os.path.join(config.OUTPUT_DIR, "field_labmodel_test.npz")
    if not (os.path.exists(tmp_val) and os.path.exists(tmp_test)):
        mc_predictions(LAB_MODEL, cache_path, tmp_val, tmp_test, seed=0)
    val, test = np.load(tmp_val, allow_pickle=True), np.load(tmp_test, allow_pickle=True)
    rv = np.abs(val["y_val"] - val["mean_pred"])
    rt = np.abs(test["y_test"] - test["mean_pred"])
    out = {"lab_pooled_quantile": {}, "field_own_quantile": {}, "test_mae": float(np.mean(rt))}
    for d in FIELD:
        mv, mt = val["lab_val"] == d, test["lab_test"] == d
        if not (mv.any() and mt.any()):
            continue
        out["lab_pooled_quantile"][d] = float(np.mean(rt[mt] <= q_lab_pooled))
        out["field_own_quantile"][d] = float(np.mean(rt[mt] <= fsq(rv[mv])))
    return out


def main():
    df = preprocessing.add_soh(field_preprocessing.load_events())
    results = {"splits": SPLITS, "datasets": FIELD, "per_split": {}}
    for s in SPLITS:
        print(f"\n{'=' * 70}\nfield split {s}\n{'=' * 70}")
        cache_path, val_path, test_path = prepare(s, df)
        results["per_split"][str(s)] = {"pooled_vs_groupwise": pooled_vs_groupwise(val_path, test_path, FIELD),
                                        "borrowing": borrowing(cache_path, val_path, test_path)}
    main_files = files(42)
    val_main, test_main = (np.load(p, allow_pickle=True) for p in main_files[2:])
    with_val = [d for d in FIELD if (val_main["lab_val"] == d).any() and (test_main["lab_test"] == d).any()]
    results["budget"] = css.budget_sweep(main_files[0], main_files[2], main_files[3], with_val)
    results["lab_model_on_field"] = lab_model_on_field(main_files[0])
    results["persistence"] = persistence_comparison(main_files[0], main_files[3])
    cache = np.load(main_files[0], allow_pickle=True)
    results["data"] = {
        "vehicles_with_windows": {d: int(len(np.unique(np.concatenate(
            [cache[f"cell_{p}"][cache[f"lab_{p}"] == d] for p in ("train", "val", "test")])))) for d in FIELD},
        "windows": {p: int(len(cache[f"y_{p}"])) for p in ("train", "val", "test")},
        "labelled_events": {d: int((df["dataset"] == d).sum()) for d in FIELD},
        "vehicles_with_labels": {d: int(df.loc[df["dataset"] == d, "cell_id"].nunique()) for d in FIELD}}

    path = os.path.join(config.OUTPUT_DIR, "field_check.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nCoverage (%) per split: pooled / groupwise")
    for s in SPLITS:
        pg = results["per_split"][str(s)]["pooled_vs_groupwise"]
        print(f"  {s}: " + "  ".join(f"{d} {100 * pg['pooled'][d]:.1f}/{100 * pg['groupwise'][d]:.1f}"
                                     for d in pg["groupwise"]))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
