"""
K6: repeated cell-level splits.

The main pipeline uses a single, fixed cell-level split (config.SPLIT_SEED,
default = config.RANDOM_SEED = 42). This script builds 4 ADDITIONAL splits
(different split_seed, same model-init seed) end to end -- preprocessing,
training the full model under all three conformal modes, persistence
baselines, and cell-clustered bootstrap coverage CIs -- and reports
between-split variability for the central calibration comparison.

Model-init seed (config.RANDOM_SEED) is held fixed across all 5 splits, so
variability reported here is attributable to WHICH CELLS landed in which
split, not to training-seed variance (that is a separate, already-reported
source of uncertainty: outputs/multiseed_*_results.json).

Raw data is parsed ONCE and reused across all 4 new splits (see E10) --
re-parsing CALCE/RWTH/etc. 4 more times would be the same unnecessary cost
that made the pre-fix LODO run impractically slow.

New artifacts only -- nothing from the original seed-42 run is touched or
overwritten (rule: write new artifacts to new filenames/subfolders).
Per-split cache: cache/processed_split{seed}.npz
Per-split results: outputs/split{seed}_{mode}_results.json / _predictions.npz
Summary: outputs/repeated_split_summary.json

Run:  python repeated_split_experiment.py
"""
import json
import os

import numpy as np

import config
import preprocessing
from train import train_one
from cell_clustered_bootstrap import cell_clustered_bootstrap_coverage, clopper_pearson

NEW_SPLIT_SEEDS = [43, 44, 45, 46]
MODES = ["groupwise", "pooled", "normalized_groupwise"]
DATASETS = ["MIT", "RWTH", "XJTU", "BIT", "CALCE", "Oxford", "NASA"]


def persistence_for_split(cache_path, pred_path):
    """Parametrized version of persistence_baselines.py's core computation,
    for an arbitrary (cache, predictions) pair rather than the hardcoded
    seed-42 paths -- same formulas, not re-derived."""
    cache = np.load(cache_path, allow_pickle=True)
    test = np.load(pred_path, allow_pickle=True)
    std = cache["soh_std"].item()
    X, y, lab = cache["X_test"][:, :, 0], cache["y_test"], cache["lab_test"]
    model_pred = test["mean_pred"]
    persistence_pred = X[:, -1]

    out = {}
    for d in DATASETS:
        m = lab == d
        if not m.any():
            continue
        model_mae = float(np.mean(np.abs((y[m] - model_pred[m]) * std[d])))
        persist_mae = float(np.mean(np.abs((y[m] - persistence_pred[m]) * std[d])))
        out[d] = {"model_mae_real": model_mae, "persistence_mae_real": persist_mae,
                  "model_beats_persistence": model_mae < persist_mae}
    return out


def bootstrap_for_split(pred_path):
    data = np.load(pred_path, allow_pickle=True)
    out = {}
    for d in np.unique(data["lab_test"]):
        mask = data["lab_test"] == d
        y_true, lower, upper = data["y_test"][mask], data["lower"][mask], data["upper"][mask]
        cell_ids = data["cell_test"][mask]
        n_windows = int(mask.sum())
        covered = (y_true >= lower) & (y_true <= upper)
        k = int(covered.sum())
        cp_lo, cp_hi = clopper_pearson(k, n_windows)
        n_cells = len(np.unique(cell_ids))
        entry = {"n_windows": n_windows, "n_cells": n_cells, "coverage": k / n_windows if n_windows else float("nan"),
                 "clopper_pearson_95ci": [cp_lo, cp_hi]}
        if n_cells >= 2:
            boot = cell_clustered_bootstrap_coverage(y_true, lower, upper, cell_ids)
            entry["cell_bootstrap_95ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        else:
            entry["cell_bootstrap_95ci"] = None
        out[str(d)] = entry
    return out


def main():
    print(f"Repeated-split experiment: {len(NEW_SPLIT_SEEDS)} new splits "
          f"({NEW_SPLIT_SEEDS}) + existing seed-{config.RANDOM_SEED} baseline, "
          f"{len(MODES)} conformal modes each -- {len(NEW_SPLIT_SEEDS) * len(MODES)} new training runs.")

    print("Loading and SOH-processing raw data once for all new splits...")
    raw = preprocessing.load_all_raw()
    raw_df = preprocessing.add_soh(raw)

    summary = {"split_seeds": [config.RANDOM_SEED] + NEW_SPLIT_SEEDS, "per_mode": {}}

    # --- Baseline (existing seed-42) results, read not recomputed ---
    for mode in MODES:
        path = os.path.join(config.OUTPUT_DIR, f"conformal_{mode}_results.json")
        if not os.path.exists(path):
            print(f"  [!] {path} missing -- seed-42 baseline for mode={mode} unavailable, skipping in summary.")
            continue
        with open(path) as fh:
            r = json.load(fh)
        summary["per_mode"].setdefault(mode, {})[str(config.RANDOM_SEED)] = {
            "per_dataset_coverage": r["per_dataset_coverage"], "overall_coverage": r["overall_coverage"],
        }

    for seed in NEW_SPLIT_SEEDS:
        print(f"\n{'='*70}\nSplit seed {seed}\n{'='*70}")
        cache_path = preprocessing.run(split_seed=seed, raw_df=raw_df)
        print(f"  cache: {cache_path}")

        for mode in MODES:
            run_name = f"split{seed}_{mode}"
            print(f"  training model_variant=full, conformal_mode={mode}, run_name={run_name} "
                  f"(model-init seed={config.RANDOM_SEED}, unchanged)...")
            result = train_one(cache_path=cache_path, model_variant="full", seed=config.RANDOM_SEED,
                               conformal_mode=mode, run_name=run_name, verbose=False)
            summary["per_mode"].setdefault(mode, {})[str(seed)] = {
                "per_dataset_coverage": result["per_dataset_coverage"],
                "overall_coverage": result["overall_coverage"],
            }
            print(f"    overall_coverage={result['overall_coverage']:.4f}  test_mae={result['test_mae']:.5f}")

        pred_path = os.path.join(config.OUTPUT_DIR, f"split{seed}_groupwise_predictions.npz")
        summary.setdefault("persistence", {})[str(seed)] = persistence_for_split(cache_path, pred_path)
        summary.setdefault("cell_clustered_bootstrap", {})[str(seed)] = bootstrap_for_split(pred_path)

    # --- Between-split variability summary, groupwise mode (the central comparison) ---
    print(f"\n{'='*70}\nBetween-split coverage variability (groupwise, {len(summary['split_seeds'])} splits)\n{'='*70}")
    print(f"{'Dataset':8s}{'mean':>8s}{'min':>8s}{'max':>8s}{'range':>8s}")
    variability = {}
    for d in DATASETS:
        vals = []
        for seed_str, entry in summary["per_mode"].get("groupwise", {}).items():
            if d in entry["per_dataset_coverage"]:
                vals.append(entry["per_dataset_coverage"][d])
        if vals:
            variability[d] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                              "max": float(np.max(vals)), "range": float(np.max(vals) - np.min(vals)), "n_splits": len(vals)}
            print(f"{d:8s}{variability[d]['mean']:8.4f}{variability[d]['min']:8.4f}"
                  f"{variability[d]['max']:8.4f}{variability[d]['range']:8.4f}")
    summary["groupwise_coverage_variability_across_splits"] = variability

    out_path = os.path.join(config.OUTPUT_DIR, "repeated_split_summary.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
