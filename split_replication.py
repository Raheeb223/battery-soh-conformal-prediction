"""
Replicates the calibration-strategy comparison (Table 3 of the manuscript) on the four further
cell-level splits (split seeds 43-46), with every method's design frozen.

Why. HGC-CP's design (its MMD weighting, widening term and search grid) was settled on the main
split while looking at that split's test results, so its main-split numbers are not fully
held-out. The method (hgc_cp.py) and its search grid (nested_hyperparameter_search.py) were fixed
in the repository before splits 43-46 were generated, so these splits' test sets played no part
in designing it. Running the unchanged method on them is a held-out check. The hierarchical
analysis is replicated alongside it, to see whether its failure on BIT is a property of one split.

What it does, for each split s in 43-46 (nothing is retrained):
  1. MC-Dropout validation predictions of that split's groupwise model
     (outputs/split{s}_groupwise_model.pt), saved as outputs/split{s}_groupwise_predictions_val.npz
     if not already present (torch seed s, so reruns give the same file).
  2. hierarchical_calibration_v2.analyse() on that split's cache, validation and test predictions.
  3. HGC-CP's nested (lambda, beta) search, unchanged, for every real target, reading that split's
     files. A dataset with no validation or no test windows in a split (NASA under splits 43 and
     44, where its validation cell is too short for a window) is left out of that split's run.
It also records how many of each split's test cells were test cells of the main split.

Requires: cache/processed_split{43..46}.npz, outputs/split{43..46}_groupwise_model.pt and
_predictions.npz (repeated_split_experiment.py).
Output:   outputs/split_replication.json
Run:      python split_replication.py
"""
import functools
import json
import os

import numpy as np
import torch

import config
import hgc_cp_evaluation_harness as harness
import hierarchical_calibration_v2 as hier
import nested_hyperparameter_search as nhs
from model import build_model, mc_dropout_predict

SPLITS = [43, 44, 45, 46]
FIVE = ["MIT", "RWTH", "XJTU", "BIT", "CALCE"]


def paths(s):
    return (os.path.join(config.CACHE_DIR, f"processed_split{s}.npz"),
            os.path.join(config.OUTPUT_DIR, f"split{s}_groupwise_predictions_val.npz"),
            os.path.join(config.OUTPUT_DIR, f"split{s}_groupwise_predictions.npz"))


def validation_predictions(s):
    cache_path, val_path, test_path = paths(s)
    cache = np.load(cache_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    # The saved test predictions must line up with this split's cache, window for window.
    assert np.allclose(cache["y_test"], test["y_test"]) and (cache["cell_test"] == test["cell_test"]).all()
    if os.path.exists(val_path):
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("full").to(device)
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, f"split{s}_groupwise_model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()
    torch.manual_seed(s)
    mean_pred, mc_std = mc_dropout_predict(model, torch.tensor(cache["X_val"]).to(device))
    np.savez(val_path, y_val=cache["y_val"], mean_pred=mean_pred.cpu().numpy(),
             mc_std=mc_std.cpu().numpy(), lab_val=cache["lab_val"], cell_val=cache["cell_val"])
    print(f"saved {val_path}")


def hgc_cp(s):
    cache_path, val_path, test_path = paths(s)
    val = np.load(val_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    usable = [d for d in harness.DATASETS if (val["lab_val"] == d).any() and (test["lab_test"] == d).any()]
    # Point the unchanged search at this split's files.
    nhs.load_representations = functools.partial(harness.load_representations, npz_path=cache_path)
    nhs.load_calibration_residuals = functools.partial(harness.load_calibration_residuals,
                                                       npz_path=cache_path, val_pred_path=val_path)
    nhs.load_test_predictions_and_truth = functools.partial(harness.load_test_predictions_and_truth,
                                                            npz_path=cache_path, pred_path=test_path)
    nhs.DATASETS = usable
    out = {}
    for target in usable:
        r = nhs.nested_search(target, seed=0)
        out[target] = {"lam": r["selected_lam"], "beta": r["selected_beta"],
                       "coverage": r["real_target_test_coverage"]}
    return {"datasets_used": usable, "per_target": out}


def main():
    main_test_cells = set(np.load(os.path.join(config.CACHE_DIR, "processed.npz"), allow_pickle=True)["cell_test"])
    results = {"splits": SPLITS, "per_split": {}}
    for s in SPLITS:
        print(f"\n{'=' * 70}\nsplit {s}\n{'=' * 70}")
        validation_predictions(s)
        cache_path, val_path, test_path = paths(s)
        h = hier.analyse(cache_path, val_path, test_path)
        g = hgc_cp(s)
        test_cells = set(np.load(cache_path, allow_pickle=True)["cell_test"])
        per = {}
        for d in FIVE:
            t = h["per_target"].get(d)
            per[d] = None if t is None else {
                "own": t["cov_own_quantile"], "hierarchical": t["cov_eb_shrunk"],
                "prior_only": t["cov_prior_only"], "others_pooled": t["cov_cross_source_pooled"],
                "w_eb": t["w_eb"], "hgc_cp": g["per_target"].get(d, {}).get("coverage")}
        results["per_split"][str(s)] = {
            "per_dataset": per, "hgc_cp_selected": g["per_target"], "hgc_cp_datasets_used": g["datasets_used"],
            "test_cells": len(test_cells), "test_cells_also_in_main_split_test": len(test_cells & main_test_cells)}

    # Mean |coverage - 90%| over the five datasets, per split and method.
    methods = ["own", "hierarchical", "prior_only", "others_pooled", "hgc_cp"]
    for s in SPLITS:
        per = results["per_split"][str(s)]["per_dataset"]
        results["per_split"][str(s)]["mean_abs_dev_five"] = {
            m: float(np.mean([abs(per[d][m] - 0.9) for d in FIVE])) for m in methods}
    path = os.path.join(config.OUTPUT_DIR, "split_replication.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nMean |coverage - 90%| (points) on {', '.join(FIVE)}")
    print("split  " + "".join(f"{m:>15s}" for m in methods))
    for s in SPLITS:
        dev = results["per_split"][str(s)]["mean_abs_dev_five"]
        print(f"{s:<7d}" + "".join(f"{100 * dev[m]:15.2f}" for m in methods))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
